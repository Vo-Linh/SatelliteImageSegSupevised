# ---------------------------------------------------------------
# Satellite UDA Dataset Configuration
#
# Template for source -> target domain adaptation with satellite
# imagery. Replace data_root, img_dir, ann_dir, and dataset types
# with your actual dataset paths.
#
# Supported dataset types:
#   - 'CustomDataset' with custom classes (most flexible)
#   - Any registered dataset in mmseg/datasets/
#
# Example scenarios:
#   - Region A -> Region B (geographic shift)
#   - Sensor A -> Sensor B (spectral/resolution shift)
#   - Season A -> Season B (temporal shift)
# ---------------------------------------------------------------

# Dataset settings — customise for your satellite datasets
dataset_type = 'CustomDataset'

# Number of land-cover classes (update for your taxonomy)
num_classes = 7

# Class names (update for your taxonomy)
classes = [
    'background', 'building', 'road', 'water',
    'vegetation', 'barren', 'agriculture',
]

# Palette for visualisation (RGB per class)
palette = [
    [0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255],
    [0, 255, 0], [128, 96, 0], [0, 255, 255],
]

# Image normalisation (ImageNet RGB stats as default;
# replace with dataset-specific stats for best results)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

# --- Source domain pipeline ---
source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# --- Target domain pipeline ---
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# --- Test/validation pipeline ---
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root='data/satellite_source/',
            img_dir='images',
            ann_dir='labels',
            img_suffix='.png',
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root='data/satellite_target/',
            img_dir='images',
            ann_dir='labels',
            img_suffix='.png',
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            pipeline=target_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root='data/satellite_target/',
        img_dir='images',
        ann_dir='labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root='data/satellite_target/',
        img_dir='images',
        ann_dir='labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
