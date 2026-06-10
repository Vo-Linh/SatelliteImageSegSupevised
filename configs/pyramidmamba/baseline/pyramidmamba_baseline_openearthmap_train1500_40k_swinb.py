# PyramidMamba (Swin-B backbone, plain baseline) on OpenEarthMap — train 1500, val 2000.
# Plain CrossEntropy, no DAPCN. Swin-Transformer variant of PyramidMamba.
#
# Swin emits NHWC (backbone_nhwc=True permutes to NCHW) and is resolution-locked to
# img_size, so img_size=512 + SLIDE inference at 512 (matching GeoSeg's fixed-patch
# methodology). Requires mamba_ssm + causal_conv1d (CUDA) and transformers<4.41.
# Full config for the data-size sweep; the other sizes inherit from it.

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
        model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(0, 1, 2, 3),
        img_size=512,
    ),
    decode_head=dict(
        type='PyramidMambaHead',
        in_channels=[128, 1024],
        in_index=[0, 3],
        channels=128,                # cosmetic; effective width = decode_channels // 2
        num_classes=9,
        encoder_channels=(128, 1024),
        decode_channels=256,
        last_feat_size=16,
        d_state=16,
        d_conv=4,
        expand=2,
        backbone_nhwc=True,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)),
)

# Match the proven recipe (lr=6e-5, 60k); deep-merges with the base schedule.
optimizer = dict(lr=6e-5)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)

# Swin-B (~88M) is lighter than resnext101_32x16d — lower if you hit OOM.
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(split='train_1500_fixed.txt'),
)

work_dir = './work_dirs/openearthmap/pyramidmamba_baseline_train1500_swinb'
