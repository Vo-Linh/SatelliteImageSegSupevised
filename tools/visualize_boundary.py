#!/usr/bin/env python
"""Qual 2: Boundary Quality Visualization.

Generates side-by-side boundary comparison:
    GT boundary | Baseline predicted boundary | +DAPCN predicted boundary

Also computes boundary IoU and boundary F1 as quantitative metrics.

Usage:
    python tools/visualize_boundary.py \
        --config configs/unetformer/unetformer_openearthmap_train1000_40k_resnext101_32x16d.py \
        --checkpoint work_dirs/.../best_mIoU.pth \
        --baseline-checkpoint work_dirs/.../baseline_best_mIoU.pth \
        --img-dir data/OEM_edit/Train/Images \
        --ann-dir data/OEM_edit/Train/labels \
        --out-dir work_dirs/qual2_boundary \
        --num-images 10
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmseg.datasets.pipelines import Compose
from mmseg.models.builder import build_segmentor
from mmseg.models.utils.dapcn_utils import compute_boundary_gt


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def build_model(cfg, checkpoint_path, device='cuda:0'):
    """Build model and load checkpoint."""
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = checkpoint['meta'].get('CLASSES', None)
    model.PALETTE = checkpoint['meta'].get('PALETTE', None)
    model.to(device)
    model.eval()
    return model


def build_pipeline(cfg):
    """Build test pipeline from config."""
    test_pipeline_cfg = cfg.data.test.pipeline
    # Handle MultiScaleFlipAug wrapper
    if test_pipeline_cfg[1]['type'] == 'MultiScaleFlipAug':
        inner = test_pipeline_cfg[1]['transforms']
        pipeline_cfg = [test_pipeline_cfg[0]] + inner
    else:
        pipeline_cfg = test_pipeline_cfg
    return Compose(pipeline_cfg)


def inference_single(model, img_path, pipeline, device='cuda:0'):
    """Run inference on a single image, return logits."""
    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)
    img_tensor = data['img'][0].unsqueeze(0).to(device)
    img_meta = [data['img_metas'][0].data]

    with torch.no_grad():
        # Get raw logits from decode head
        x = model.extract_feat(img_tensor)
        logits = model.decode_head(x)
    return logits


def pred_boundary_from_logits(logits):
    """Extract boundary map from segmentation logits using Sobel."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=logits.dtype, device=logits.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=logits.dtype, device=logits.device)
    C = logits.shape[1]
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    gx = F.conv2d(logits, sobel_x, padding=1, groups=C)
    gy = F.conv2d(logits, sobel_y, padding=1, groups=C)
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    boundary = mag.sum(dim=1, keepdim=True)
    bmax = boundary.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    boundary = (boundary / bmax).clamp(0, 1)
    return boundary.squeeze(0).squeeze(0).cpu().numpy()


def gt_boundary_from_label(label_path, target_h, target_w, ignore_index=255):
    """Compute GT boundary map from label image."""
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {label_path}")
    label = cv2.resize(label, (target_w, target_h),
                       interpolation=cv2.INTER_NEAREST)
    label_t = torch.from_numpy(label).long().unsqueeze(0)
    boundary = compute_boundary_gt(label_t, ignore_index=ignore_index)
    return boundary.squeeze(0).squeeze(0).numpy()


def compute_boundary_metrics(pred, gt, threshold=0.5):
    """Compute boundary IoU and F1."""
    pred_bin = (pred > threshold).astype(np.float32)
    gt_bin = (gt > threshold).astype(np.float32)

    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection

    iou = intersection / max(union, 1e-6)
    precision = intersection / max(pred_bin.sum(), 1e-6)
    recall = intersection / max(gt_bin.sum(), 1e-6)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return dict(boundary_iou=iou, boundary_f1=f1,
                boundary_precision=precision, boundary_recall=recall)


def boundary_to_colormap(boundary, cmap='hot'):
    """Convert boundary map [0,1] to RGB colormap image."""
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    colored = cm(boundary)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def overlay_boundary_on_image(img, boundary, color=(0, 255, 0),
                              threshold=0.5, alpha=0.6):
    """Overlay boundary contour on the original image."""
    overlay = img.copy()
    mask = boundary > threshold
    overlay[mask] = (np.array(color) * alpha +
                     overlay[mask] * (1 - alpha)).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Qual 2: Boundary Visualization')
    parser.add_argument('--config', required=True, help='DAPCN model config')
    parser.add_argument('--checkpoint', required=True, help='DAPCN checkpoint')
    parser.add_argument('--baseline-config', default=None,
                        help='Baseline config (if different from --config)')
    parser.add_argument('--baseline-checkpoint', default=None,
                        help='Baseline model checkpoint')
    parser.add_argument('--img-dir', required=True, help='Image directory')
    parser.add_argument('--ann-dir', required=True, help='Annotation directory')
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--seg-map-suffix', default='.tif')
    parser.add_argument('--out-dir', default='work_dirs/qual2_boundary')
    parser.add_argument('--num-images', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build DAPCN model
    cfg = Config.fromfile(args.config)
    model = build_model(cfg, args.checkpoint, args.device)
    pipeline = build_pipeline(cfg)

    # Build baseline model (optional)
    baseline_model = None
    if args.baseline_checkpoint:
        baseline_cfg = Config.fromfile(
            args.baseline_config if args.baseline_config else args.config)
        baseline_model = build_model(baseline_cfg, args.baseline_checkpoint,
                                     args.device)

    # Collect images
    img_files = sorted([f for f in os.listdir(args.img_dir)
                        if f.endswith(args.img_suffix)])[:args.num_images]

    all_metrics = {'dapcn': [], 'baseline': []}

    for img_name in mmcv.track_iter_progress(img_files):
        stem = img_name.replace(args.img_suffix, '')
        img_path = osp.join(args.img_dir, img_name)
        ann_path = osp.join(args.ann_dir, stem + args.seg_map_suffix)

        # Original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # DAPCN prediction
        logits = inference_single(model, img_path, pipeline, args.device)
        _, _, H, W = logits.shape
        pred_boundary = pred_boundary_from_logits(logits)

        # GT boundary
        gt_boundary = gt_boundary_from_label(ann_path, H, W)

        # Metrics
        m_dapcn = compute_boundary_metrics(pred_boundary, gt_boundary)
        all_metrics['dapcn'].append(m_dapcn)

        # Visualize
        img_resized = cv2.resize(img_rgb, (W, H))
        gt_vis = boundary_to_colormap(gt_boundary)
        pred_vis = boundary_to_colormap(pred_boundary)
        gt_overlay = overlay_boundary_on_image(img_resized, gt_boundary)
        pred_overlay = overlay_boundary_on_image(img_resized, pred_boundary)

        panels = [img_resized, gt_vis, gt_overlay]
        labels = ['Image', 'GT Boundary', 'GT Overlay']

        if baseline_model is not None:
            base_logits = inference_single(baseline_model, img_path,
                                           pipeline, args.device)
            base_boundary = pred_boundary_from_logits(base_logits)
            base_vis = boundary_to_colormap(base_boundary)
            base_overlay = overlay_boundary_on_image(img_resized, base_boundary)
            m_base = compute_boundary_metrics(base_boundary, gt_boundary)
            all_metrics['baseline'].append(m_base)

            panels += [base_vis, base_overlay]
            labels += [
                f"Baseline (F1={m_base['boundary_f1']:.3f})",
                'Baseline Overlay',
            ]

        panels += [pred_vis, pred_overlay]
        labels += [
            f"+DAPCN (F1={m_dapcn['boundary_f1']:.3f})",
            '+DAPCN Overlay',
        ]

        # Compose figure
        import matplotlib.pyplot as plt
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        for ax, panel, label in zip(axes, panels, labels):
            ax.imshow(panel)
            ax.set_title(label, fontsize=10)
            ax.axis('off')
        plt.tight_layout()
        fig.savefig(osp.join(args.out_dir, f'{stem}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # Print aggregate metrics
    print('\n=== Boundary Quality Metrics (mean) ===')
    for key in ['dapcn', 'baseline']:
        if all_metrics[key]:
            avg = {k: np.mean([m[k] for m in all_metrics[key]])
                   for k in all_metrics[key][0]}
            print(f"  {key}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in avg.items()))

    # Save metrics
    import json
    metrics_path = osp.join(args.out_dir, 'boundary_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: [dict(m) for m in v]
                   for k, v in all_metrics.items() if v}, f, indent=2)
    print(f'Metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
