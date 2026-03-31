# SegFormer (MiT-B5) on OpenEarthMap - Train on 1500 samples
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
        type='mit_b5',
        style='pytorch',
    ),
    decode_head=dict(
        type='SegFormerDAPCNHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768, conv_kernel_size=1),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ignore_index=255,
        # --- DAPC before_fusion parameters ---
        da_position='after_fusion',
        da_feature_dim=768,
        boundary_lambda=0.15,
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
        dapg_loss=dict(type='DAPGLoss', margin=0.3, lambda_inter=0.5, lambda_quality=0.1),
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

work_dir = './work_dirs/openearthmap/segformer_train1500'
