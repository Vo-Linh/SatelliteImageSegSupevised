# PyramidMamba (Swin-B backbone) + DAPCN on OpenEarthMap — train 1500, val 2000.
#
# The Swin-Transformer variant of PyramidMamba (GeoSeg's paper model). Differs
# from the ResNeXt101 configs in two ways:
#   1. The Swin backbone emits NHWC features -> head uses backbone_nhwc=True to
#      permute them to NCHW before the decoder.
#   2. timm Swin is resolution-locked to img_size, so we fix img_size=512 and run
#      SLIDE inference at 512 (test_cfg mode='slide'), matching GeoSeg's fixed-size
#      patch / sliding-window methodology while keeping the 512 training crop used
#      by every other model here.
#
# Requires mamba_ssm + causal_conv1d (CUDA) and transformers<4.41.
# Full config for the data-size sweep; the other sizes inherit from it.

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
        model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(0, 1, 2, 3),    # strides 4/8/16/32, dims [128,256,512,1024]
        img_size=512,                # Swin is resolution-locked to this size
    ),
    decode_head=dict(
        type='PyramidMambaDAPCNHead',
        in_channels=[128, 1024],     # selected: stride-4 (128), stride-32 (1024)
        in_index=[0, 3],
        channels=128,                # cosmetic; effective width = decode_channels // 2
        num_classes=9,
        encoder_channels=(128, 1024),
        decode_channels=256,         # parity with resnext set (GeoSeg Swin default: 128)
        last_feat_size=16,           # 512 // 32
        d_state=16,
        d_conv=4,
        expand=2,
        backbone_nhwc=True,          # permute Swin NHWC -> NCHW
        input_transform='multiple_select',
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        da_position='after_fusion',  # required for NHWC backbones (DA sees NCHW fused feat)
        boundary_lambda=0.15,
        proto_lambda=0.1,
        contrastive_lambda=0.1,
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
    # Slide inference at 512 — each window matches the Swin img_size.
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)),
)

# Mirror the proven ResNeXt101 recipe (lr=6e-5, 60k) + DA-LR boost.
optimizer = dict(
    type='AdamW',
    lr=6e-5,
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

runner = dict(type='IterBasedRunner', max_iters=60000)

# Keep at most 3 rolling checkpoints; best_mIoU.pth kept separately via save_best.
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(split='train_1500_fixed.txt'),
)

work_dir = './work_dirs/openearthmap/pyramidmamba_dapcn_train1500_swinb'
