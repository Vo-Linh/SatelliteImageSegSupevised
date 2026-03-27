# ---------------------------------------------------------------
# SegFormer (MiT-B5) + Dynamic Anchor Module (After Fusion)
# Dataset: Cityscapes | Supervised Training
# ---------------------------------------------------------------
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
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch',
    ),
    decode_head=dict(
        type='SegFormerDAPCNHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=19,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dim=256,
            conv_kernel_size=1,
        ),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # --- Dynamic Anchor: AFTER fusion ---
        da_position='after_fusion',
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

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=4000, metric='mIoU')
