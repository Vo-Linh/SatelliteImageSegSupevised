# Bareland (class_1) OVERSAMPLING — train from scratch.
#
# Rationale: two loss-level fixes (class weighting, Lovasz warm-start) failed
# to move Bareland. Root issue is EXPOSURE, not loss shape: Bareland appears in
# only 510/3000 (17%) of train images, so the model sees it ~1 batch in 6 and
# its decision boundary stays unstable (val IoU swings 24-48%).
#
# This config changes EXACTLY ONE thing vs the 72.07-TTA baseline: the train
# split is replaced by train_woxbd_3000_bareland3x.txt, which duplicates the
# 510 Bareland-containing images 3x -> Bareland presence 17% -> 38%. Everything
# else (model, unweighted CE, DAPG on, boundary, 60k schedule, optimizer) is
# inherited unchanged, so any mIoU delta is attributable to exposure alone.
#
# Train from scratch (NO warm-start — the LR restart hurt last time):
#   .venv/bin/python tools/train.py configs/unetformer/oem_woxbd_bareland_oversample3x.py
# TTA-eval the best ckpt (arch matches baseline):
#   .venv/bin/python tools/test.py configs/unetformer/oem_woxbd_bareland_oversample3x.py \
#       work_dirs/openearthmap/oem_woxbd_bareland_oversample3x/best_mIoU.pth --eval mIoU

_base_ = ['./unetformer_openearthmap_train3000_val500_40k_resnext101_32x16d_postfusion.py']

work_dir = './work_dirs/openearthmap/oem_woxbd_bareland_oversample3x'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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

data = dict(
    samples_per_gpu=8,
    train=dict(split='train_woxbd_3000_bareland3x.txt'),  # <-- the only change
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
