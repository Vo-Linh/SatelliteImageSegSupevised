# PyramidMamba (plain baseline) + ResNeXt101_32x16d on OpenEarthMap
# Train on 1500 samples, val on 2000. Plain CrossEntropy, no DAPCN.
#
# Faithful EfficientPyramidMamba decoder (arXiv:2406.10828, GeoSeg) ported as an
# mmseg decode head. Requires `mamba_ssm` + `causal_conv1d` (CUDA) and
# `transformers<4.41`. This is the pure-comparison baseline; the DAPCN variant
# lives in ../pyramidmamba_openearthmap_train1500_40k_resnext101_32x16d.py.
#
# FULL config for the data-size sweep; train500/train1000 and the wo_xBD
# (3000/500) variants inherit from it and only override the splits.

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/datasets/openearthmap_val2000.py',
    '../../_base_/schedules/schedule_40k_openearthmap.py',
]

seed = 0
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4),
    ),
    decode_head=dict(
        type='PyramidMambaHead',
        in_channels=[256, 2048],     # selected: stride-4, stride-32
        in_index=[0, 3],
        channels=128,                # cosmetic; effective width = decode_channels // 2
        num_classes=9,
        encoder_channels=(256, 2048),
        decode_channels=256,         # matches the proven ResNeXt101 UNetFormer recipe
        last_feat_size=16,           # 512 crop // 32
        d_state=16,
        d_conv=4,
        expand=2,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# samples_per_gpu=8 matches the ResNeXt101 baselines. PyramidMamba's Mamba
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(split='train_1500_fixed.txt'),
)


# Match the proven ResNeXt101 recipe (lr=6e-5, 60k) — the base 3e-5/40k
# under-trains the rare, water-confusable class 0. (deep-merges with base schedule)
optimizer = dict(lr=6e-5)
runner = dict(type='IterBasedRunner', max_iters=60000)

# Keep at most 3 rolling checkpoints (the latest is always among them);
# best_mIoU.pth is kept separately via evaluation's save_best='mIoU'.
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)

work_dir = './work_dirs/openearthmap/pyramidmamba_baseline_train1500_resnext101_32x16d'
