# UNetFormer on OpenEarthMap - Train on 500 samples (STABLE CONFIG)
# This config reduces DAPCN loss weights to prevent gradient explosion
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
        type='UNetFormer',
        in_channels=3,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='UNetFormerDAPCNHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        channels=256,
        num_classes=9,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        da_position='after_fusion',
        # Reduced DAPCN weights for stability
        boundary_lambda=0.1,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        proto_lambda=0.5,
        num_prototypes_per_class=1,
        prototype_ema=0.999,
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=9,
            temperature=0.07,
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
        contrastive_lambda=0.05,
        contrastive_temperature=0.07,
        contrastive_sample_ratio=0.1,
        warmup_iters=1000,
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)

# Use lower learning rate for stability
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))

work_dir = './work_dirs/openearthmap/unetformer_train500_stable'
