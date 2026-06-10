# Exp 4 Ablation: CE + L_boundary + L_dapg (no contrastive)
# UNetFormer ResNeXt101 on OpenEarthMap train1000
# Reference: unetformer_openearthmap_train3500_40k_resnext101_32x16d_postfusion.py

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
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        da_position='after_fusion',
        # --- Ablation: boundary + DAPG (no contrastive) ---
        boundary_lambda=0.15,
        proto_lambda=0.1,
        contrastive_lambda=0.0,
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

# Override optimizer — boost DA component learning rates
optimizer = dict(
    type='AdamW',
    lr=3e-5,
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

work_dir = './work_dirs/openearthmap/ablation/unetformer_resnext101_boundary_dapg'
data = dict(samples_per_gpu=8,
            workers_per_gpu=2,
            train=dict(
                split='train_1500_fixed.txt'))

runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
