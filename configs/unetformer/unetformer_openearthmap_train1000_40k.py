# UNetFormer B0 on OpenEarthMap - Train on 1000 samples
# Paper-faithful: ResNet18 encoder + GLA decoder + DAPCN

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/openearthmap_val2000.py',
    '../_base_/schedules/schedule_40k_openearthmap.py',
]

seed = 0
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet18.fb_swsl_ig1b_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4),
    ),
    decode_head=dict(
        type='UNetFormerDAPCNHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=64,
        num_classes=9,
        encoder_channels=(64, 128, 256, 512),
        decode_channels=64,
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
        boundary_lambda=0.15,
        proto_lambda=0.1,
        contrastive_lambda=0.1,
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
            margin=0.3, lambda_inter=0.5, lambda_quality=0.1
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

work_dir = './work_dirs/openearthmap/unetformer_train1000'
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
