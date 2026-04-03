#!/usr/bin/env python
"""Qual 3: Prototype Evolution Visualization (t-SNE/UMAP).

Visualizes how DynamicAnchorModule prototypes evolve during training
by loading checkpoints at different iterations and plotting their
embeddings with quality scores.

Usage:
    python tools/visualize_prototypes.py \
        --config configs/unetformer/unetformer_openearthmap_train1000_40k_resnext101_32x16d.py \
        --checkpoints iter_4000.pth iter_16000.pth iter_32000.pth iter_40000.pth \
        --labels "4K" "16K" "32K" "40K" \
        --out-dir work_dirs/qual3_prototypes \
        --method tsne
"""

import argparse
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmseg.models.builder import build_segmentor


def extract_prototypes(model):
    """Extract prototype vectors and quality scores from a model.

    Returns:
        prototypes: (K, C) numpy array
        quality: (K,) numpy array or None
    """
    head = model.decode_head

    # DynamicAnchorModule prototypes
    if not hasattr(head, 'dynamic_anchor'):
        raise ValueError("Model does not have DynamicAnchorModule")

    da = head.dynamic_anchor
    prototypes = da.prototypes.detach().cpu().numpy()  # (K, C)

    # Quality scores (if quality gate is enabled)
    quality = None
    if hasattr(da, 'quality_net'):
        with torch.no_grad():
            proto_tensor = da.prototypes.detach()
            q = da.quality_net(proto_tensor).squeeze(-1)
            quality = q.cpu().numpy()

    return prototypes, quality


def reduce_dimensions(embeddings, method='tsne', perplexity=15, n_neighbors=10):
    """Reduce high-dimensional embeddings to 2D."""
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                       random_state=42, init='pca', learning_rate='auto')
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("pip install umap-learn")
        reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(embeddings) - 1),
                            random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)


def plot_prototype_evolution(all_protos, all_quality, labels, method, out_path):
    """Plot prototype embeddings across training stages in a single figure.

    Each subplot shows prototypes at a different checkpoint, colored by
    quality score.
    """
    n = len(all_protos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    # Combine all prototypes for joint dimensionality reduction
    combined = np.concatenate(all_protos, axis=0)
    combined_2d = reduce_dimensions(combined, method=method)

    # Split back
    splits = np.cumsum([0] + [p.shape[0] for p in all_protos])

    vmin = min(q.min() for q in all_quality if q is not None)
    vmax = max(q.max() for q in all_quality if q is not None)

    for i, ax in enumerate(axes):
        s, e = splits[i], splits[i + 1]
        pts = combined_2d[s:e]
        q = all_quality[i]

        if q is not None:
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=q, cmap='RdYlGn',
                            s=60, edgecolors='k', linewidths=0.5,
                            vmin=vmin, vmax=vmax, alpha=0.85)
            # Mark low-quality prototypes
            low_q = q < 0.1
            if low_q.any():
                ax.scatter(pts[low_q, 0], pts[low_q, 1],
                           marker='x', c='red', s=40, linewidths=1.5,
                           label=f'Low quality ({low_q.sum()})')
                ax.legend(fontsize=8, loc='lower right')
        else:
            sc = ax.scatter(pts[:, 0], pts[:, 1], c='steelblue',
                            s=60, edgecolors='k', linewidths=0.5)

        ax.set_title(f'{labels[i]}\n({all_protos[i].shape[0]} prototypes)',
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    if all_quality[0] is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label('Quality Score', fontsize=10)

    fig.suptitle(f'Prototype Evolution ({method.upper()})', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_prototype_similarity_matrix(all_protos, labels, out_path):
    """Plot inter-prototype cosine similarity heatmap at each stage."""
    n = len(all_protos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        protos = all_protos[i]
        # L2 normalize
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        protos_norm = protos / norms
        sim = protos_norm @ protos_norm.T

        im = ax.imshow(sim, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f'{labels[i]}', fontsize=11)
        ax.set_xlabel('Prototype')
        ax.set_ylabel('Prototype')

    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label='Cosine Similarity')
    fig.suptitle('Inter-Prototype Similarity', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_quality_histogram(all_quality, labels, out_path):
    """Plot quality score distribution across training stages."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_quality)))

    for i, (q, label) in enumerate(zip(all_quality, labels)):
        if q is not None:
            ax.hist(q, bins=20, alpha=0.5, color=colors[i], label=label,
                    range=(0, 1), edgecolor='k', linewidth=0.5)

    ax.set_xlabel('Quality Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Quality Score Distribution Over Training', fontsize=13)
    ax.legend()
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7,
               label='Threshold (0.1)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Qual 3: Prototype Evolution Visualization')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint paths at different training stages')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each checkpoint (e.g. "4K" "16K" "40K")')
    parser.add_argument('--out-dir', default='work_dirs/qual3_prototypes')
    parser.add_argument('--method', default='tsne', choices=['tsne', 'umap'])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labels = args.labels or [f'ckpt_{i}' for i in range(len(args.checkpoints))]
    assert len(labels) == len(args.checkpoints)

    cfg = Config.fromfile(args.config)

    all_protos = []
    all_quality = []

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f'Loading {label}: {ckpt_path}')
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, ckpt_path, map_location='cpu')
        model.eval()

        protos, quality = extract_prototypes(model)
        all_protos.append(protos)
        all_quality.append(quality)

        print(f'  Prototypes: {protos.shape}, '
              f'Quality: [{quality.min():.3f}, {quality.max():.3f}]'
              if quality is not None else '  (no quality gate)')

    # Plot 1: t-SNE/UMAP evolution
    plot_prototype_evolution(
        all_protos, all_quality, labels, args.method,
        osp.join(args.out_dir, f'proto_evolution_{args.method}.png'))

    # Plot 2: Similarity matrices
    plot_prototype_similarity_matrix(
        all_protos, labels,
        osp.join(args.out_dir, 'proto_similarity.png'))

    # Plot 3: Quality histograms
    if all_quality[0] is not None:
        plot_quality_histogram(
            all_quality, labels,
            osp.join(args.out_dir, 'quality_histogram.png'))

    print(f'\nAll outputs saved to {args.out_dir}')


if __name__ == '__main__':
    main()
