# DDRNet on OpenEarthMap - Train on 1000 samples
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/openearthmap_val2000.py',
    '../_base_/schedules/schedule_40k_openearthmap.py',
]

seed = 0
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DDRNet',
        in_channels=3,
        channels=64,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    decode_head=dict(
        type='DDRNetDAPCNHead',
        in_channels=512,
        channels=128,
        num_classes=9,
        in_index=-1,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        da_position='before_fusion',
        boundary_lambda=0.3,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        proto_lambda=0.1,
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,
            temperature=0.1,
            num_iters=3,
            init_method='xavier',
            min_quality=0.1,
            use_quality_gate=True,
            use_mask_predictor=False,
            ema_decay=0.0,
        ),
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,
            lambda_inter=0.5,
            lambda_quality=0.1,
        ),
        contrastive_lambda=0.1,
        contrastive_temperature=0.07,
        contrastive_sample_ratio=0.1,
        warmup_iters=500,
        num_prototypes_per_class=1,
        prototype_ema=0.999,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

work_dir = './work_dirs/openearthmap/ddrnet_train1000'
