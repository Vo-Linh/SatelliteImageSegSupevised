# ---------------------------------------------------------------
# DeepLabV3+ with ResNet-101 (dilated stride-8)
#
# Backbone: ResNetV1c-101 with output_stride=8 (d8 variant)
#   - Stages 2-3 use atrous convolutions (dilations 2, 4)
#   - All four stages output at strides {4, 8, 8, 8}
#   - Channel dimensions: [256, 512, 1024, 2048]
#
# Decoder: DepthwiseSeparableASPPHead (DeepLabV3+)
#   - ASPP on stage-3 features (2048-d) with rates (1, 12, 24, 36)
#   - Low-level (c1) branch from stage-0 (256 → 48)
#   - Fused output: 512-d before conv_seg
#
# Pretrained: torchvision ResNet-101 v1c (deep stem)
#
# Dimension trace:
#   Encoder: [256, 512, 1024, 2048]
#   ASPP:    2048 → 5×512 → cat 2560 → bottleneck 512
#   c1:      256 → 48
#   Fusion:  cat(512, 48) = 560 → sep_bottleneck → 512
#   conv_seg: 512 → num_classes
# ---------------------------------------------------------------

norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
