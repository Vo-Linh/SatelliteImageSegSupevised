# E1 (new baseline) — capped class-weighted CE + DAPG removed.
#
# Rationale (from loss-component review):
#   * CE was UNWEIGHTED on a severe 9-class imbalance; class_1 Bareland
#     (~1.3% of pixels, IoU 48 at best, std ~6.7 across evals) is the single
#     mIoU bottleneck and variance source. A frequency-grounded class_weight,
#     capped to [0.7, 2.5] so it does not destabilize the rare class under
#     lr=3e-5 / grad_clip=5, directly attacks it.
#   * DAPG is net-neutral (in-repo ablation: boundary_only 0.7031 vs
#     boundary+dapg 0.7027), its quality term is dead, and its intra term
#     contracts the SAME fused feature feeding the classifier toward 32
#     class-agnostic centroids. proto_lambda=0 removes it cleanly and reclaims
#     ~3.5% of gradient for CE, and de-noises later A/Bs.
#   * Boundary loss (benign, mildly helps edges) is kept.
#
# class_weight order = (c0 bg, c1 Bareland, c2 Rangeland, c3 Developed,
#   c4 Road, c5 Tree, c6 Water, c7 Agriculture, c8 Building).
# Measured train pixel freq ~ [0.8, 1.6, 25, 16, 6, 20, 3, 13.5, 14]%.
#
# Report at the PEAK checkpoint with TTA (use oem_woxbd_E0_ttaeval.py against
# this run's best_mIoU.pth). Watch loss_seg/acc_seg for the first ~8k iters:
# if acc_seg < 70% or loss_seg stays > ~2x baseline, soften c1 2.5 -> 2.0.

_base_ = ['./unetformer_openearthmap_train3000_val500_40k_resnext101_32x16d_postfusion.py']

work_dir = './work_dirs/openearthmap/oem_woxbd_E1_clsweight_nodapg'

model = dict(
    decode_head=dict(
        proto_lambda=0.0,  # DAPG off (dynamic_anchor + dapg_loss not built)
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 2.5, 0.7, 0.85, 1.5, 0.75, 1.3, 0.9, 0.9]),
    ))

# TTA eval of THIS run's checkpoint (arch matches: DAPG off). Inherited by the
# E2/E3 configs. Use after training:
#   .venv/bin/python tools/test.py <this_config> <work_dir>/best_mIoU.pth --eval mIoU
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
    test=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_woxbd_500.txt',
        ignore_index=255,
        pipeline=tta_pipeline),
)
