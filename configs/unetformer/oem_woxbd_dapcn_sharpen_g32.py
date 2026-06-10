# DAPCN method experiment — make dapg_intra / dapg_inter actually DESCEND
# while keeping max_groups = 32.
#
# Goal: this is a METHODS experiment about the DAPG loss dynamics, NOT a direct
# mIoU push. We hold the prototype count at 32 and instead turn the knobs the
# mechanism analysis says create the floors:
#
#   WHY they were stuck (baseline temp=0.1, ema_decay=0.9, num_iters=3,
#   lambda_inter=0.5, proto_lambda=0.1):
#     * SOFT assignment (softmax over 32 protos) makes every M-step centroid a
#       weighted mean of ALL features -> all 32 inherit the global-mean
#       direction -> mutually correlated -> dapg_inter floors (~0.145, avg
#       pairwise cos ~0.45). Same softness keeps each pixel's weighted-cos < 1
#       -> dapg_intra floors (~0.08).
#     * ema_decay=0.9 freezes the prototype trajectory -> losses flatline.
#     * lambda_quality drives quality_net->1 (dead, no pruning).
#     * proto_lambda=0.1 is too weak to move the CE-pinned features.
#
#   KNOBS CHANGED (each targets a specific cause):
#     temperature 0.1 -> 0.04   : sharpen assignment toward one-hot -> centroids
#                                 become hard cluster means (NOT global-mean-
#                                 shrunk) -> they separate -> inter & intra drop.
#                                 [primary lever]
#     num_iters    3  -> 5      : let the sharpened clusters actually converge.
#     ema_decay   0.9 -> 0.5    : unfreeze the trajectory so prototypes adapt.
#     lambda_inter 0.5 -> 1.0   : push pairwise separation harder.
#     lambda_quality 0.1 -> 0.0 : drop the dead, degenerate term.
#     proto_lambda 0.1 -> 0.2   : give DAPG enough gradient to move features.
#
# GUARDRAIL: proto_lambda=0.2 fights CE; this can hurt mIoU. Watch IoU.class_*
# at the first evals — if mIoU drops >1 pt vs the 71-baseline, lower
# proto_lambda back toward 0.1. Success criterion HERE is dapg_inter/intra
# trending well below their old floors (inter << 0.145, intra << 0.08), not mIoU.
#
# From scratch (warm-start LR-restart hurt last time):
#   .venv/bin/python tools/train.py configs/unetformer/oem_woxbd_dapcn_sharpen_g32.py
# Then watch: tr '\r' '\n' < <work_dir>/<ts>.log | grep -oE "dapg_loss_(intra|inter): [0-9.]+"

_base_ = ['./unetformer_openearthmap_train3000_val500_40k_resnext101_32x16d_postfusion.py']

work_dir = './work_dirs/openearthmap/oem_woxbd_dapcn_sharpen_g32'

model = dict(
    decode_head=dict(
        proto_lambda=0.2,                       # was 0.1 — stronger DAPG gradient
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=32,                      # held at 32 as requested
            temperature=0.04,                   # was 0.1 — PRIMARY: sharper assignment
            num_iters=5,                        # was 3  — converge the clusters
            ema_decay=0.5,                      # was 0.9 — unfreeze prototypes
            min_quality=0.3,
        ),
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,
            lambda_inter=1.0,                   # was 0.5 — push separation harder
            lambda_quality=0.0,                 # was 0.1 — kill the dead term
        ),
    ))

# TTA eval of this run's checkpoint (arch matches baseline: DAPG on).
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
