# ---------------------------------------------------------------
# Semi-Supervised Satellite Dataset Configuration
#
# Structure: labeled subset (split file) + unlabeled (full set)
# Both from the SAME domain.
#
# Before training:
#   1. Update data_root to your satellite dataset path
#   2. Create a split file (splits/labeled.txt) with one image
#      basename per line (without suffix), e.g.:
#        img_001
#        img_042
#        img_107
#   3. Update classes, palette, num_classes for your taxonomy
#   4. Optionally generate sample_class_stats.json and
#      samples_with_class.json for Rare Class Sampling
#      (see tools/convert_datasets/ for reference)
# ---------------------------------------------------------------

dataset_type = 'CustomDataset'

num_classes = 7

classes = [
    'background', 'building', 'road', 'water',
    'vegetation', 'barren', 'agriculture',
]

palette = [
    [0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255],
    [0, 255, 0], [128, 96, 0], [0, 255, 255],
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

# --- Labeled subset pipeline (with augmentation) ---
labeled_train_pipeline = [
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

# --- Unlabeled subset pipeline ---
# Labels are loaded but only used by EMA teacher for evaluation;
# during training, pseudo-labels replace them.
unlabeled_train_pipeline = [
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
        type='SSLDataset',
        labeled=dict(
            type=dataset_type,
            data_root='data/satellite/',
            img_dir='images',
            ann_dir='labels',
            img_suffix='.png',
            seg_map_suffix='.png',
            # Split file: one basename per line (no suffix)
            split='splits/labeled.txt',
            classes=classes,
            palette=palette,
            pipeline=labeled_train_pipeline),
        unlabeled=dict(
            type=dataset_type,
            data_root='data/satellite/',
            img_dir='images',
            ann_dir='labels',
            img_suffix='.png',
            seg_map_suffix='.png',
            # No split = use ALL images as unlabeled pool
            classes=classes,
            palette=palette,
            pipeline=unlabeled_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root='data/satellite/',
        img_dir='images',
        ann_dir='labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root='data/satellite/',
        img_dir='images',
        ann_dir='labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
