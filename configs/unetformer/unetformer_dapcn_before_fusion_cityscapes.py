# UNetFormer with DAPCN head (before fusion) on Cityscapes
# Hybrid CNN-Transformer backbone with dynamic anchor prototype-guided contrastive learning
# Applied before feature fusion in the decoder

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/cityscapes_half_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='UNetFormerDAPCNHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        channels=256,
        num_classes=19,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        # DAPCN Configuration
        # Position of dynamic anchor application: before or after feature fusion
        da_position='before_fusion',
        # Boundary detection and loss
        boundary_lambda=0.3,  # Weight for boundary auxiliary loss
        boundary_mode='sobel',  # Gradient-based boundary detection method
        boundary_loss_mode='binary',  # Binary cross-entropy for boundary prediction
        # Prototype learning
        proto_lambda=0.1,  # Weight for prototype learning loss
        num_prototypes_per_class=1,  # Number of prototype vectors per class
        prototype_ema=0.999,  # EMA decay for prototype updates (0.999 = slow update)
        # Dynamic Anchor Module Configuration
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,  # Maximum number of anchor groups
            temperature=0.1,  # Temperature for soft assignment
            num_iters=3,  # Number of optimization iterations per batch
            init_method='xavier',  # Weight initialization method
            min_quality=0.1,  # Minimum quality threshold for anchors
            use_quality_gate=True,  # Enable quality-based filtering
            use_mask_predictor=False,  # Disable per-anchor mask prediction
            ema_decay=0.0,  # EMA for anchor tracking (0.0 = no EMA)
        ),
        # DAPG Loss Configuration (Dynamic Anchor Prototype-Guided Loss)
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,  # Margin for triplet-like loss
            lambda_inter=0.5,  # Weight for inter-class separation
            lambda_quality=0.1,  # Weight for quality-aware loss
        ),
        # Contrastive Learning Configuration
        contrastive_lambda=0.1,  # Weight for contrastive loss
        contrastive_temperature=0.07,  # Temperature for contrastive similarity
        contrastive_sample_ratio=0.1,  # Ratio of hard samples to mine
        # Training Configuration
        warmup_iters=500,  # Number of iterations for loss warmup
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'norm': dict(decay_mult=0.0),
        }))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=4000, metric='mIoU')
