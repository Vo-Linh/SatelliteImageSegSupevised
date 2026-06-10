# TTA re-evaluation of the Swin-B best_mIoU.pth on the labeled 500-image val
# split. Multi-scale + h-flip aug is baked into data.test below (softmax-
# averaged in EncoderDecoder.aug_test over the slide-mode crops), so do NOT
# pass --aug-test on the CLI.
#
# Run (GPU must be free enough):
#   .venv-blackwell/bin/python tools/test.py \
#       configs/pyramidmamba/pyramidmamba_train3000_val500_swinb_E0_ttaeval.py \
#       work_dirs/openearthmap/pyramidmamba_dapcn_woxbd_train3000_val500_swinb/best_mIoU.pth \
#       --eval mIoU

_base_ = ['./pyramidmamba_openearthmap_train3000_val500_40k_swinb.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Multi-scale + horizontal-flip TTA. Pad uses size_divisor=32 so every scale
# stays divisible by the Swin patch/window strides.
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

# Point data.test at the LABELED val split so dataset.evaluate(mIoU) works.
data = dict(
    samples_per_gpu=4,
    test=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_woxbd_500.txt',
        test_mode=True,
        ignore_index=255,
        pipeline=tta_pipeline),
)
