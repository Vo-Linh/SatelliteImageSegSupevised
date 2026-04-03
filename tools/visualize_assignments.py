#!/usr/bin/env python
"""Qual 5: Assignment Map Visualization.

Visualizes the soft assignment maps from DynamicAnchorModule — which pixels
attend to which prototypes. Shows that different prototypes capture different
semantic structures (boundaries, textures, regions).

Usage:
    python tools/visualize_assignments.py \
        --config configs/unetformer/unetformer_openearthmap_train1000_40k_resnext101_32x16d.py \
        --checkpoint work_dirs/.../best_mIoU.pth \
        --img-dir data/OEM_edit/Train/Images \
        --ann-dir data/OEM_edit/Train/labels \
        --out-dir work_dirs/qual5_assignments \
        --num-images 5 --top-k 8
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


OEM_CLASSES = [
    'bareland', 'rangeland', 'developed space', 'road',
    'tree', 'water', 'agriculture', 'building', 'nodata'
]
OEM_PALETTE = np.array([
    [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
    [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7], [0, 0, 0],
], dtype=np.uint8)


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


def get_da_input_feature(model, img_tensor):
    """Get the feature tensor that feeds into DynamicAnchorModule.

    Respects da_position to return the correct feature.
    """
    head = model.decode_head
    x = model.extract_feat(img_tensor)
    da_position = getattr(head, 'da_position', 'after_fusion')

    if da_position == 'before_fusion':
        if hasattr(head, 'in_index') and isinstance(head.in_index, (list, tuple)):
            return x[head.in_index[-1]]
        return x[-1]
    else:
        # after_fusion: need fused decoder feature
        if hasattr(head, '_decode'):
            return head._decode(x)
        elif hasattr(head, '_get_fused_feature'):
            return head._get_fused_feature(x)
        elif hasattr(head, 'bottleneck'):
            feat = head._transform_inputs(x)
            return head.bottleneck(feat)
        elif hasattr(head, 'fuse'):
            feat = head._transform_inputs(x)
            return head.fuse(feat, feat)
        else:
            raise RuntimeError("Cannot derive fused feature for this head")


def extract_assignments(model, img_path, pipeline, device='cuda:0'):
    """Run DynamicAnchorModule forward and return assignment maps + metadata.

    Returns:
        assign_map: (H, W, K') assignment weights per pixel
        quality: (K',) quality scores
        seg_pred: (H, W) argmax segmentation prediction
        feature_shape: (H, W) spatial dims of the DA feature
    """
    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)
    img_tensor = data['img'][0].unsqueeze(0).to(device)

    head = model.decode_head
    if not hasattr(head, 'dynamic_anchor'):
        raise ValueError("Model decode_head has no dynamic_anchor module")

    with torch.no_grad():
        # Get DA input feature
        da_feat = get_da_input_feature(model, img_tensor)
        B, C, H, W = da_feat.shape

        # Run DynamicAnchorModule
        assign, proto, quality = head.dynamic_anchor(da_feat)
        # assign: (B*H*W, K'), proto: (K', C), quality: (K',)

        assign_map = assign.reshape(B, H, W, -1).squeeze(0).cpu().numpy()
        quality = quality.cpu().numpy()

        # Also get segmentation prediction
        x = model.extract_feat(img_tensor)
        logits = head(x)
        seg_pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return assign_map, quality, seg_pred, (H, W)


def select_top_k_prototypes(assign_map, quality, top_k=8):
    """Select top-K prototypes by total assignment mass (most activated).

    Returns indices sorted by activation mass.
    """
    # Total activation per prototype: sum of assignments across all pixels
    mass = assign_map.sum(axis=(0, 1))  # (K',)
    top_indices = np.argsort(mass)[::-1][:top_k]
    return top_indices


def compute_prototype_class_correlation(assign_map, seg_pred, num_classes=9):
    """For each prototype, compute which classes it predominantly covers.

    Returns:
        correlation: (K', num_classes) — fraction of assignment mass per class
    """
    H, W, K = assign_map.shape
    correlation = np.zeros((K, num_classes), dtype=np.float32)

    for c in range(num_classes):
        mask = (seg_pred == c).astype(np.float32)  # (H, W)
        for k in range(K):
            correlation[k, c] = (assign_map[:, :, k] * mask).sum()

    # Normalize per prototype
    row_sums = correlation.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    correlation = correlation / row_sums

    return correlation


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_assignment_grid(img_rgb, gt_label, seg_pred, assign_map, quality,
                         top_indices, stem, out_dir):
    """Plot image + GT + pred + top-K assignment maps."""
    K = len(top_indices)
    ncols = 3 + K
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.5))

    H, W = assign_map.shape[:2]
    img_show = cv2.resize(img_rgb, (W, H))

    # Image
    axes[0].imshow(img_show)
    axes[0].set_title('Image', fontsize=9)
    axes[0].axis('off')

    # GT
    gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
    gt_resized = cv2.resize(gt_label, (W, H), interpolation=cv2.INTER_NEAREST)
    for c in range(len(OEM_PALETTE)):
        gt_vis[gt_resized == c] = OEM_PALETTE[c]
    axes[1].imshow(gt_vis)
    axes[1].set_title('GT', fontsize=9)
    axes[1].axis('off')

    # Prediction
    pred_vis = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(len(OEM_PALETTE)):
        pred_vis[seg_pred == c] = OEM_PALETTE[c]
    axes[2].imshow(pred_vis)
    axes[2].set_title('Prediction', fontsize=9)
    axes[2].axis('off')

    # Assignment maps for top-K prototypes
    for i, k in enumerate(top_indices):
        a = assign_map[:, :, k]
        a_norm = a / max(a.max(), 1e-8)
        axes[3 + i].imshow(a_norm, cmap='hot', vmin=0, vmax=1)
        axes[3 + i].set_title(f'Proto {k}\nQ={quality[k]:.2f}', fontsize=8)
        axes[3 + i].axis('off')

    plt.suptitle(f'{stem}', fontsize=11)
    plt.tight_layout()
    fig.savefig(osp.join(out_dir, f'{stem}_assignments.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_prototype_class_heatmap(correlation, quality, top_indices,
                                 class_names, out_path):
    """Plot heatmap showing which classes each prototype covers."""
    sub_corr = correlation[top_indices]  # (top_k, num_classes)
    sub_q = quality[top_indices]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(top_indices))))
    im = ax.imshow(sub_corr, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f'Proto {k} (Q={sub_q[i]:.2f})'
                        for i, k in enumerate(top_indices)], fontsize=8)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)

    # Annotate cells
    for i in range(len(top_indices)):
        for j in range(len(class_names)):
            val = sub_corr[i, j]
            if val > 0.05:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Assignment Fraction')
    ax.set_title('Prototype-to-Class Correlation', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_hard_assignment_segmentation(assign_map, quality, img_rgb, out_path):
    """Color each pixel by its dominant prototype (hard assignment)."""
    H, W, K = assign_map.shape
    hard = assign_map.argmax(axis=2)  # (H, W)

    # Generate distinct colors for each prototype
    cmap = plt.cm.get_cmap('tab20', K)
    colors = (cmap(np.arange(K))[:, :3] * 255).astype(np.uint8)

    vis = colors[hard]  # (H, W, 3)

    img_show = cv2.resize(img_rgb, (W, H))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    ax1.imshow(img_show)
    ax1.set_title('Image', fontsize=10)
    ax1.axis('off')

    ax2.imshow(vis)
    ax2.set_title(f'Hard Assignment ({K} prototypes)', fontsize=10)
    ax2.axis('off')

    # Overlay
    overlay = (img_show.astype(np.float32) * 0.4 +
               vis.astype(np.float32) * 0.6).astype(np.uint8)
    ax3.imshow(overlay)
    ax3.set_title('Overlay', fontsize=10)
    ax3.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Qual 5: Assignment Map Visualization')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--seg-map-suffix', default='.tif')
    parser.add_argument('--out-dir', default='work_dirs/qual5_assignments')
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=8,
                        help='Number of top prototypes to visualize')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = build_model(cfg, args.checkpoint, args.device)
    pipeline = build_pipeline(cfg)

    img_files = sorted([f for f in os.listdir(args.img_dir)
                        if f.endswith(args.img_suffix)])[:args.num_images]

    all_correlations = []

    for img_name in img_files:
        stem = img_name.replace(args.img_suffix, '')
        img_path = osp.join(args.img_dir, img_name)
        ann_path = osp.join(args.ann_dir, stem + args.seg_map_suffix)

        print(f'Processing: {img_name}')

        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt_label = cv2.imread(ann_path, cv2.IMREAD_UNCHANGED)

        assign_map, quality, seg_pred, (H, W) = extract_assignments(
            model, img_path, pipeline, args.device)

        K = assign_map.shape[2]
        top_indices = select_top_k_prototypes(
            assign_map, quality, min(args.top_k, K))

        print(f'  Feature size: {H}x{W}, Active prototypes: {K}, '
              f'Quality range: [{quality.min():.3f}, {quality.max():.3f}]')

        # Plot 1: Assignment grid (image + GT + pred + top-K maps)
        plot_assignment_grid(img_rgb, gt_label, seg_pred, assign_map,
                             quality, top_indices, stem, args.out_dir)

        # Plot 2: Hard assignment coloring
        plot_hard_assignment_segmentation(
            assign_map, quality, img_rgb,
            osp.join(args.out_dir, f'{stem}_hard_assign.png'))

        # Compute prototype-class correlation
        corr = compute_prototype_class_correlation(
            assign_map, seg_pred, num_classes=len(OEM_CLASSES))
        all_correlations.append(corr)

        # Plot 3: Per-image correlation heatmap
        plot_prototype_class_heatmap(
            corr, quality, top_indices, OEM_CLASSES,
            osp.join(args.out_dir, f'{stem}_proto_class_corr.png'))

    # Aggregate correlation across images
    if len(all_correlations) > 1:
        # Average correlation (need same K for all images)
        Ks = [c.shape[0] for c in all_correlations]
        if len(set(Ks)) == 1:
            avg_corr = np.mean(all_correlations, axis=0)
            avg_quality = np.ones(Ks[0])  # placeholder
            top_idx = select_top_k_prototypes(
                np.ones((1, 1, Ks[0])),  # dummy
                avg_quality,
                min(args.top_k, Ks[0]))
            # Sort by max class fraction for better visualization
            max_frac = avg_corr.max(axis=1)
            top_idx = np.argsort(max_frac)[::-1][:args.top_k]

            plot_prototype_class_heatmap(
                avg_corr, avg_quality, top_idx, OEM_CLASSES,
                osp.join(args.out_dir, 'aggregate_proto_class_corr.png'))

    print(f'\nAll outputs saved to {args.out_dir}')


if __name__ == '__main__':
    main()
