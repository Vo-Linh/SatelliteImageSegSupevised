log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
dataset_type = 'OpenEarthMapDataset'
data_root = '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'
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
    dict(type='RandomRotate', degree=(-180, 180), prob=0.5),
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
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/train',
        ann_dir='annotations/train',
        split='train_1500_fixed.txt',
        ignore_index=255,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomRotate', degree=(-180, 180), prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1024, 1024),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1024, 1024),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=2.0, decay_mult=1.0),
            prototypes=dict(lr_mult=1.0, decay_mult=0.01),
            quality=dict(lr_mult=1.0, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)
evaluation = dict(
    interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
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
        drop_path_rate=0.2,
        input_transform='multiple_select',
        dropout_ratio=0.2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ignore_index=255,
        da_position='after_fusion',
        boundary_lambda=0.15,
        proto_lambda=0.3,
        contrastive_lambda=0.1,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,
            temperature=1.0,
            num_iters=1,
            ema_decay=0.99),
        dapg_loss=dict(
            type='DAPGLoss', margin=0.3, lambda_inter=1.0,
            lambda_quality=0.5)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = './work_dirs/openearthmap/unetformer_train1500_resnext101_32x16d'
gpu_ids = range(0, 1)
