#!/usr/bin/env python
"""Generate competition submission PNGs from model predictions.

Usage:
    python tools/generate_submission.py <config> <checkpoint> [--out-dir OUT_DIR]

Example:
    python tools/generate_submission.py \
        configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py \
        work_dirs/openearthmap/segformer_train1000/best_mIoU.pth \
        --out-dir submission_output/
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
import warnings
import zipfile

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
warnings.filterwarnings('ignore')
import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models.builder import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate OpenEarthMap competition submission')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out-dir',
        default='submission_output',
        help='directory to save prediction PNGs')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU id to use')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta'].get('CLASSES', dataset.CLASSES)
    model.PALETTE = checkpoint['meta'].get('PALETTE', dataset.PALETTE)

    model = MMDataParallel(model, device_ids=[args.gpu_id])
    outputs = single_gpu_test(model, data_loader)

    os.makedirs(args.out_dir, exist_ok=True)

    split_file = cfg.data.test.split
    if not osp.isabs(split_file):
        split_file = osp.join(cfg.data.test.data_root, split_file)

    with open(split_file) as f:
        names = [line.strip() for line in f if line.strip()]

    assert len(names) == len(outputs), (
        f'Split file has {len(names)} entries but model produced '
        f'{len(outputs)} predictions')

    # Collect original image sizes for rescaling predictions
    ori_shapes = []
    data_loader_eval = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    for data in data_loader_eval:
        meta = data['img_metas'][0].data[0][0]
        ori_shapes.append(meta['ori_shape'][:2])  # (H, W)

    print(f'Saving {len(outputs)} predictions to {args.out_dir}/')
    prog_bar = mmcv.ProgressBar(len(outputs))
    for name, pred, ori_shape in zip(names, outputs, ori_shapes):
        basename = osp.basename(name)
        out_path = osp.join(args.out_dir, basename + '.png')
        if isinstance(pred, str):
            pred = np.load(pred)
        pred = pred.astype(np.uint8)
        # Resize prediction back to original image size if needed
        if pred.shape != tuple(ori_shape):
            pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_path, pred)
        prog_bar.update()

    zip_path = args.out_dir.rstrip('/') + '.zip'
    print(f'\nCreating submission zip: {zip_path}')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(args.out_dir)):
            if fname.endswith('.png'):
                zf.write(osp.join(args.out_dir, fname), fname)

    print(f'Done. Submission zip: {zip_path}')
    print(f'Total files: {len([f for f in os.listdir(args.out_dir) if f.endswith(".png")])}')


if __name__ == '__main__':
    main()
