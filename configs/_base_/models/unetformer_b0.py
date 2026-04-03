# UNetFormer B0 base model config
# Paper-faithful: CNN encoder (ResNet18 via timm) + Transformer decoder (GlobalLocalAttention)
# B0 variant: encoder_channels=(64, 128, 256, 512), decode_channels=64

norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

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
        type='UNetFormerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=64,
        num_classes=19,
        encoder_channels=(64, 128, 256, 512),
        decode_channels=64,
        window_size=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        dropout_ratio=0.1,
        input_transform='multiple_select',
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
