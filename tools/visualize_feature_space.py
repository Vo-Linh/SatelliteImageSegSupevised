#!/usr/bin/env python
"""Qual 4: Feature Space Analysis (t-SNE/UMAP).

Compares pixel-level feature embeddings between baseline and +DAPCN models.
Shows that DAPCN produces more compact intra-class and more separated
inter-class clusters.

Usage:
    python tools/visualize_feature_space.py \
        --config configs/unetformer/unetformer_openearthmap_train1000_40k_resnext101_32x16d.py \
        --checkpoint work_dirs/.../best_mIoU.pth \
        --baseline-config configs/unetformer/... \
        --baseline-checkpoint work_dirs/.../baseline_best.pth \
        --img-dir data/OEM_edit/Train/Images \
        --ann-dir data/OEM_edit/Train/labels \
        --out-dir work_dirs/qual4_features \
        --num-images 5 --sample-ratio 0.02
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmseg.datasets.pipelines import Compose
from mmseg.models.builder import build_segmentor


# ---------------------------------------------------------------
# OEM class names and colors
# ---------------------------------------------------------------
OEM_CLASSES = [
    'bareland', 'rangeland', 'developed space', 'road',
    'tree', 'water', 'agriculture', 'building', 'nodata'
]
OEM_PALETTE = np.array([
    [128, 0, 0],      # bareland
    [0, 255, 36],     # rangeland
    [148, 148, 148],  # developed space
    [255, 255, 255],  # road
    [34, 97, 38],     # tree
    [0, 69, 255],     # water
    [75, 181, 73],    # agriculture
    [222, 31, 7],     # building
    [0, 0, 0],        # nodata
], dtype=np.float32) / 255.0


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def build_model(cfg, checkpoint_path, device='cuda:0'):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def build_pipeline(cfg):
    test_pipeline_cfg = cfg.data.test.pipeline
    if test_pipeline_cfg[1]['type'] == 'MultiScaleFlipAug':
        inner = test_pipeline_cfg[1]['transforms']
        pipeline_cfg = [test_pipeline_cfg[0]] + inner
    else:
        pipeline_cfg = test_pipeline_cfg
    return Compose(pipeline_cfg)


def extract_features(model, img_path, pipeline, device='cuda:0'):
    """Extract fused decoder features (before cls_seg) for a single image.

    Returns:
        features: (H, W, C) numpy array
    """
    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)
    img_tensor = data['img'][0].unsqueeze(0).to(device)

    with torch.no_grad():
        x = model.extract_feat(img_tensor)
        head = model.decode_head

        # Get fused feature before classification
        if hasattr(head, '_decode'):
            # UNetFormerHead
            fused = head._decode(x)
        elif hasattr(head, '_get_fused_feature'):
            # SegFormerDAPCNHead, KNetDAPCNHead
            fused = head._get_fused_feature(x)
        elif hasattr(head, 'bottleneck'):
            # DDRNetHead
            feat = head._transform_inputs(x)
            fused = head.bottleneck(feat)
        elif hasattr(head, 'fuse'):
            # PIDNetHead
            feat = head._transform_inputs(x)
            fused = head.fuse(feat, feat)
        elif hasattr(head, 'feat_proj'):
            # SegMenterHead
            feat = head._transform_inputs(x)
            B, C, H, W = feat.shape
            x_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            fused = head.feat_proj(x_flat).view(B, H, W, -1)
            fused = fused.permute(0, 3, 1, 2)
        else:
            raise RuntimeError("Cannot extract fused features from this head type")

    # (B, C, H, W) -> (H, W, C)
    return fused.squeeze(0).permute(1, 2, 0).cpu().numpy()


def sample_pixels(features, labels, sample_ratio=0.02, ignore_index=255,
                  max_samples=10000):
    """Randomly sample pixels from features/labels.

    Returns:
        sampled_feats: (N, C)
        sampled_labels: (N,)
    """
    H, W, C = features.shape
    feats_flat = features.reshape(-1, C)
    labels_flat = labels.reshape(-1)

    valid = labels_flat != ignore_index
    valid_idx = np.where(valid)[0]

    n_sample = min(int(len(valid_idx) * sample_ratio), max_samples)
    n_sample = max(n_sample, 100)

    chosen = np.random.choice(valid_idx, size=min(n_sample, len(valid_idx)),
                              replace=False)
    return feats_flat[chosen], labels_flat[chosen]


def reduce_dimensions(embeddings, method='tsne', perplexity=30):
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2,
                       perplexity=min(perplexity, len(embeddings) - 1),
                       random_state=42, init='pca', learning_rate='auto')
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(embeddings)


def compute_cluster_metrics(features_2d, labels):
    """Compute intra-class compactness and inter-class separation."""
    classes = np.unique(labels)

    centroids = {}
    intra_dists = {}
    for c in classes:
        mask = labels == c
        pts = features_2d[mask]
        centroid = pts.mean(axis=0)
        centroids[c] = centroid
        intra_dists[c] = np.mean(np.linalg.norm(pts - centroid, axis=1))

    avg_intra = np.mean(list(intra_dists.values()))

    # Inter-class: mean distance between centroids
    cent_arr = np.stack(list(centroids.values()))
    from scipy.spatial.distance import pdist
    avg_inter = np.mean(pdist(cent_arr))

    return dict(
        avg_intra_distance=float(avg_intra),
        avg_inter_distance=float(avg_inter),
        separation_ratio=float(avg_inter / max(avg_intra, 1e-8)),
    )


def plot_feature_space(feats_2d, labels, title, ax, class_names=None,
                       palette=None):
    """Plot 2D feature embeddings colored by class."""
    classes = np.unique(labels)
    for c in classes:
        mask = labels == c
        color = palette[c] if palette is not None and c < len(palette) else None
        name = class_names[c] if class_names and c < len(class_names) else f'cls_{c}'
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                   c=[color] if color is not None else None,
                   s=6, alpha=0.5, label=name, edgecolors='none')
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Qual 4: Feature Space Visualization')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline-config', default=None)
    parser.add_argument('--baseline-checkpoint', default=None)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--seg-map-suffix', default='.tif')
    parser.add_argument('--out-dir', default='work_dirs/qual4_features')
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--sample-ratio', type=float, default=0.02)
    parser.add_argument('--method', default='tsne', choices=['tsne', 'umap'])
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(42)

    # Build models
    cfg = Config.fromfile(args.config)
    model = build_model(cfg, args.checkpoint, args.device)
    pipeline = build_pipeline(cfg)

    baseline_model = None
    if args.baseline_checkpoint:
        base_cfg = Config.fromfile(
            args.baseline_config if args.baseline_config else args.config)
        baseline_model = build_model(base_cfg, args.baseline_checkpoint,
                                     args.device)

    # Collect images
    img_files = sorted([f for f in os.listdir(args.img_dir)
                        if f.endswith(args.img_suffix)])[:args.num_images]

    # Accumulate features across images
    all_dapcn_feats, all_dapcn_labels = [], []
    all_base_feats, all_base_labels = [], []

    for img_name in img_files:
        stem = img_name.replace(args.img_suffix, '')
        img_path = osp.join(args.img_dir, img_name)
        ann_path = osp.join(args.ann_dir, stem + args.seg_map_suffix)

        print(f'Processing: {img_name}')

        # DAPCN features
        feats = extract_features(model, img_path, pipeline, args.device)
        H, W, _ = feats.shape
        label = cv2.imread(ann_path, cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)

        sf, sl = sample_pixels(feats, label, args.sample_ratio)
        all_dapcn_feats.append(sf)
        all_dapcn_labels.append(sl)

        # Baseline features
        if baseline_model is not None:
            base_feats = extract_features(baseline_model, img_path,
                                          pipeline, args.device)
            bH, bW, _ = base_feats.shape
            base_label = cv2.resize(
                cv2.imread(ann_path, cv2.IMREAD_UNCHANGED),
                (bW, bH), interpolation=cv2.INTER_NEAREST)
            bsf, bsl = sample_pixels(base_feats, base_label, args.sample_ratio)
            all_base_feats.append(bsf)
            all_base_labels.append(bsl)

    dapcn_feats = np.concatenate(all_dapcn_feats)
    dapcn_labels = np.concatenate(all_dapcn_labels)

    print(f'\nTotal DAPCN samples: {len(dapcn_feats)}')

    # --- Plot 1: DAPCN-only feature space ---
    print(f'Running {args.method} on DAPCN features...')
    dapcn_2d = reduce_dimensions(dapcn_feats, method=args.method)
    dapcn_metrics = compute_cluster_metrics(dapcn_2d, dapcn_labels)

    if baseline_model is not None:
        base_feats_all = np.concatenate(all_base_feats)
        base_labels_all = np.concatenate(all_base_labels)
        print(f'Total baseline samples: {len(base_feats_all)}')

        print(f'Running {args.method} on baseline features...')
        base_2d = reduce_dimensions(base_feats_all, method=args.method)
        base_metrics = compute_cluster_metrics(base_2d, base_labels_all)

        # Side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_feature_space(
            base_2d, base_labels_all,
            f"Baseline ({args.method.upper()})\n"
            f"Intra={base_metrics['avg_intra_distance']:.2f}, "
            f"Inter={base_metrics['avg_inter_distance']:.2f}, "
            f"Ratio={base_metrics['separation_ratio']:.2f}",
            ax1, OEM_CLASSES, OEM_PALETTE)
        plot_feature_space(
            dapcn_2d, dapcn_labels,
            f"+DAPCN ({args.method.upper()})\n"
            f"Intra={dapcn_metrics['avg_intra_distance']:.2f}, "
            f"Inter={dapcn_metrics['avg_inter_distance']:.2f}, "
            f"Ratio={dapcn_metrics['separation_ratio']:.2f}",
            ax2, OEM_CLASSES, OEM_PALETTE)

        # Shared legend
        handles, leg_labels = ax2.get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc='lower center',
                   ncol=min(len(handles), 5), fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle('Feature Space: Baseline vs +DAPCN', fontsize=14)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        plot_feature_space(
            dapcn_2d, dapcn_labels,
            f"+DAPCN ({args.method.upper()})\n"
            f"Intra={dapcn_metrics['avg_intra_distance']:.2f}, "
            f"Inter={dapcn_metrics['avg_inter_distance']:.2f}, "
            f"Ratio={dapcn_metrics['separation_ratio']:.2f}",
            ax, OEM_CLASSES, OEM_PALETTE)
        ax.legend(fontsize=8, markerscale=3)

    plt.tight_layout()
    out_path = osp.join(args.out_dir, f'feature_space_{args.method}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

    # --- Print metrics ---
    print('\n=== Cluster Quality Metrics ===')
    print(f"  +DAPCN: {dapcn_metrics}")
    if baseline_model is not None:
        print(f"  Baseline: {base_metrics}")
        delta = (dapcn_metrics['separation_ratio'] -
                 base_metrics['separation_ratio'])
        print(f"  Separation ratio improvement: {delta:+.3f}")

    # Save metrics
    import json
    metrics = {'dapcn': dapcn_metrics}
    if baseline_model is not None:
        metrics['baseline'] = base_metrics
    with open(osp.join(args.out_dir, 'feature_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
