# PyramidMamba (EfficientPyramidMamba decoder) + ResNeXt101_32x16d on OpenEarthMap
# DAPCN variant — train on 1500 samples, val on 2000.
#
# PyramidMamba: Pyramid Pooling + Mamba SSM decoder (arXiv:2406.10828, GeoSeg).
# Requires `mamba_ssm` + `causal_conv1d` (CUDA-compiled) and `transformers<4.41`.
# Backbone gives stride-4 (256ch) + stride-32 (2048ch) features; the head selects
# them via in_index=[0, 3]. DAPCN auxiliary losses mirror the ResNeXt101
# UNetFormer-DAPCN config (da_position='after_fusion').
#
# This is the FULL config for the data-size sweep; train500/train1000 and the
# wo_xBD (3000/500) variants inherit from it and only override the splits.

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/openearthmap_val2000.py',
    '../_base_/schedules/schedule_40k_openearthmap.py',
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
        type='PyramidMambaDAPCNHead',
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
        # DAPCN auxiliary losses (operate on the fused decoder feature).
        da_position='after_fusion',
        boundary_lambda=0.15,
        proto_lambda=0.1,
        contrastive_lambda=0.1,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=32,
            temperature=0.1,
            num_iters=3,
            ema_decay=0.9,
            min_quality=0.3,
        ),
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,
            lambda_inter=0.5,
            lambda_quality=0.1,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# Optimizer/schedule mirror the proven ResNeXt101 UNetFormer recipe
# (lr=6e-5, 60k iters) — the base 3e-5/40k under-trains the rare, water-
# confusable class 0. Also boost DA-component learning rates.
optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=2.0, decay_mult=1.0),
            'prototypes': dict(lr_mult=5.0, decay_mult=0.01),
            'quality_net': dict(lr_mult=5.0, decay_mult=1.0),
        }))

runner = dict(type='IterBasedRunner', max_iters=60000)

# Keep at most 3 rolling checkpoints (the latest is always among them);
# best_mIoU.pth is kept separately via evaluation's save_best='mIoU'.
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)

# samples_per_gpu=8 matches the ResNeXt101 baselines. PyramidMamba's Mamba
# block is memory-heavy — lower to 4 if you hit OOM.
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(split='train_1500_fixed.txt'),
)

work_dir = './work_dirs/openearthmap/pyramidmamba_dapcn_train1500_resnext101_32x16d'
