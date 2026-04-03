# Exp 4 Ablation: CE + L_dapg ONLY
# UNetFormer ResNeXt101 on OpenEarthMap train1000

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
        # --- Ablation: DAPG ONLY ---
        boundary_lambda=0.0,
        proto_lambda=0.1,
        contrastive_lambda=0.0,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,
            temperature=0.5,
            num_iters=3,
        ),
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3, lambda_inter=0.5, lambda_quality=0.1,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

work_dir = './work_dirs/openearthmap/ablation/unetformer_resnext101_dapg_only'
data = dict(samples_per_gpu=8, workers_per_gpu=2)
runner = dict(type='IterBasedRunner', max_iters=60000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
