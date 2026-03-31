# ---------------------------------------------------------------
# PIDNet + Dynamic Anchor Module (Before Fusion)
# Dataset: Cityscapes | Supervised Training
# DA Position: Before feature fusion (operates on raw backbone features)
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/cityscapes_half_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0
norm_cfg = dict(type='BN', requires_grad=True)

# ---------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=64,
        ppm_channels=96,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    decode_head=dict(
        type='PIDNetDAPCNHead',
        in_channels=512,  # PIDNet fused_s4 channels (channels=64 → 64*8=512)
        channels=128,     # Decode head intermediate channels
        num_classes=19,   # Cityscapes has 19 semantic classes
        in_index=-1,      # Select last backbone output (fused_s4)
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        # --- Dynamic Anchor Module: BEFORE Fusion ---
        # Operates on raw backbone features before decode head processing
        # DA feature_dim auto-inferred: in_channels=512
        da_position='before_fusion',
        # --- Boundary Loss Configuration ---
        # Encourages the model to learn sharper, more defined segment boundaries
        boundary_lambda=0.3,  # Weight for boundary loss
        boundary_mode='sobel',  # Use Sobel filter for boundary detection
        boundary_loss_mode='binary',  # Binary cross-entropy for boundary prediction
        # --- Prototype Loss Configuration ---
        # Ensures prototypes are well-separated and representative
        proto_lambda=0.1,  # Weight for prototype loss
        # --- Dynamic Anchor Module Parameters ---
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,  # Maximum number of anchor groups
            temperature=0.1,  # Temperature for softmax in group assignment
            num_iters=3,  # Number of iterations for anchor optimization
            init_method='xavier',  # Initialization method for anchors
            min_quality=0.1,  # Minimum quality threshold for anchors
            use_quality_gate=True,  # Enable quality gating mechanism
            use_mask_predictor=False,  # Disable pixel-level mask prediction
            ema_decay=0.0,  # EMA decay for anchor updates (0 = no EMA)
        ),
        # --- DAPG Loss Configuration ---
        # Encourages compact prototypes and class separation
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,  # Margin for class separation
            lambda_inter=0.5,  # Weight for inter-class repulsion
            lambda_quality=0.1,  # Weight for quality regularization
        ),
        # --- Contrastive Learning Configuration ---
        # Enhances feature discrimination across classes
        contrastive_lambda=0.1,  # Weight for contrastive loss
        contrastive_temperature=0.07,  # Temperature for contrastive similarity
        contrastive_sample_ratio=0.1,  # Ratio of samples for contrastive learning
        # --- Warmup Configuration ---
        # Gradually introduces DA module during training
        warmup_iters=500,  # Number of iterations for warmup phase
        # --- Prototype Configuration ---
        num_prototypes_per_class=1,  # Number of prototypes per semantic class
        prototype_ema=0.999,  # EMA decay for prototype updates
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ---------------------------------------------------------------
# Optimizer Configuration
# ---------------------------------------------------------------
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),  # Higher LR for decoder head
            'norm': dict(decay_mult=0.0),  # No weight decay for normalization
        }))

# ---------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=4000, metric='mIoU')
