# Lovasz-Softmax, WEIGHT-FREE, warm-started from the existing 72-mIoU
# checkpoint. Fast fine-tune to test whether the direct mIoU surrogate moves
# Bareland (c1) where class weighting did not.
#
# Delta vs baseline = EXACTLY ONE change: add CE + 0.5*Lovasz.
#   * CE stays UNWEIGHTED (no class_weight) — the lever that failed is unused.
#   * DAPG kept ON (proto_lambda inherited = 0.1) so the full checkpoint loads
#     with zero key mismatch and Lovasz is the only new signal (clean A/B).
#   * Lovasz computed at FULL label resolution inside the mixin (key
#     'loss_lovasz'); LovaszLoss has no learnable params.
#
# Warm-start: load_from the converged checkpoint, fine-tune 16k iters at half
# LR (1.5e-5). Report at peak ckpt with the baked-in TTA data.test:
#   .venv/bin/python tools/test.py \
#       configs/unetformer/oem_woxbd_lovasz_weightfree_warmstart.py \
#       work_dirs/openearthmap/oem_woxbd_lovasz_weightfree_warmstart/best_mIoU.pth \
#       --eval mIoU

_base_ = ['./unetformer_openearthmap_train3000_val500_40k_resnext101_32x16d_postfusion.py']

work_dir = './work_dirs/openearthmap/oem_woxbd_lovasz_weightfree_warmstart'

# Warm-start segmentation weights from the converged baseline (strict=False:
# all backbone/decoder/cls_seg/DAPG keys match; nothing is dropped).
load_from = ('/home/ubuntu/SatelliteImageSegSupevised/work_dirs/openearthmap/'
             'unetformer_woxbd_resnext101_32x16d_postfusion/best_mIoU.pth')

model = dict(
    decode_head=dict(
        lovasz_lambda=0.5,
        lovasz_loss=dict(
            type='LovaszLoss',
            loss_type='multi_class',
            classes='present',
            per_image=False),
    ))

# Gentle fine-tune: half LR, short schedule. paramwise_cfg/betas/wd inherited.
optimizer = dict(lr=1.5e-5)
runner = dict(type='IterBasedRunner', max_iters=16000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True, save_best='mIoU')

# TTA eval of THIS run's checkpoint (arch matches the baseline: DAPG on).
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
