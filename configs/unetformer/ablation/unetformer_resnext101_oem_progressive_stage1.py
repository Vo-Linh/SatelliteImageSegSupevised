# Ablation Config: Progressive Training - Stage 1 (Boundary Only)
# First train with boundary loss, then add DAPG in Stage 2

_base_ = [
    '../../_base_/datasets/openearthmap_val2000.py',
    '../../_base_/default_runtime.py'
]

# Model configuration - Boundary loss only (Stage 1)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)),
    decode_head=dict(
        type='UNetFormerDAPCNHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=64,
        num_classes=9,
        encoder_channels=(256, 512, 1024, 2048),
        decode_channels=256,
        window_size=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ignore_index=255,
        da_position='after_fusion',
        # Stage 1: Boundary only
        boundary_lambda=0.15,
        proto_lambda=0.0,
        contrastive_lambda=0.0,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=32,
            temperature=0.1,
            num_iters=3,
            ema_decay=0.9,
            min_quality=0.3),
        dapg_loss=dict(
            type='DAPGLoss', margin=0.3, lambda_inter=0.5,
            lambda_quality=0.1)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Stage 1: Train for 30k iterations with boundary only
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')

# Dataset configuration - OpenEarthMap 1500
data_root = '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'
dataset_type = 'OpenEarthMapDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        split='train_1500_fixed.txt',
        ignore_index=255,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=test_pipeline))

# Optimizer - standard settings
optimizer = dict(
    type='AdamW',
    lr=3e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=2.0, decay_mult=1.0),
            prototypes=dict(lr_mult=5.0, decay_mult=0.01),
            quality=dict(lr_mult=1.0, decay_mult=1.0),
            quality_net=dict(lr_mult=5.0, decay_mult=1.0))))

optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

# Standard LR schedule
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

work_dir = './work_dirs/openearthmap/ablation/unetformer_resnext101_progressive_stage1'
gpu_ids = range(0, 1)
