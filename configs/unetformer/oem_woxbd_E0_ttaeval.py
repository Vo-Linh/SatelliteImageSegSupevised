# E0 — TTA re-evaluation of the EXISTING best_mIoU.pth (iter 44k). No retrain.
#
# Purpose: denoise the headline mIoU (currently a single lucky c1=48.1 eval)
# and measure the multi-scale + h-flip TTA ceiling before spending any
# GPU-hours on a retrain. The aug_test softmax-averaging path already exists
# in EncoderDecoder; we only need a labeled VAL `data.test` with a
# MultiScaleFlipAug pipeline.
#
# Run (when the GPU is free):
#   .venv/bin/python tools/test.py \
#       configs/unetformer/oem_woxbd_E0_ttaeval.py \
#       work_dirs/openearthmap/unetformer_woxbd_resnext101_32x16d_postfusion/best_mIoU.pth \
#       --eval mIoU
#
# (Do NOT pass --aug-test: the ratios/flip are baked into the pipeline below.
#  --aug-test would also force scale 0.5/1.75 which is out-of-distribution for
#  these 1000px tiles trained at 512 crops.)

_base_ = ['./unetformer_openearthmap_train3000_val500_40k_resnext101_32x16d_postfusion.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Multi-scale + horizontal-flip TTA. Pad uses size_divisor=32 (NOT a fixed
# 512 size) so every scale stays divisible by the backbone strides.
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        img_ratios=[0.75, 1.0, 1.25],
        flip=True,
        flip_direction=['horizontal'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Point `data.test` at the LABELED val split so dataset.evaluate(mIoU) works.
data = dict(
    samples_per_gpu=8,
    train=dict(split='train_woxbd_3000.txt'),
    val=dict(split='val_woxbd_500.txt'),
    test=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_woxbd_500.txt',
        ignore_index=255,
        pipeline=tta_pipeline),
)
