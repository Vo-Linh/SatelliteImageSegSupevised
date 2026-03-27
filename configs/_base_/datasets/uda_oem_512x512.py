# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/OEM_edit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
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
            type='CustomDataset',
            data_root='data/OEM_edit/',
            img_dir='Train/Images',
            ann_dir='Train/labels',
            split='label.txt',
            img_suffix='.tif',
            seg_map_suffix='.tif',
            pipeline=source_pipeline,
            test_mode=False),
        target=dict(
            type='CustomDataset',
            data_root='data/OEM_edit/',
            img_dir='Train/Images',
            ann_dir=None,
            split='unlabel.txt',
            img_suffix='.tif',
            pipeline=target_pipeline,
            test_mode=True)),
    val=dict(
        type='CustomDataset',
        data_root='data/OEM_edit/',
        img_dir='Train/Images',
        ann_dir='Train/labels',
        split='label.txt',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        pipeline=test_pipeline,
        test_mode=False),
    test=dict(
        type='CustomDataset',
        data_root='data/OEM_edit/',
        img_dir='Train/Images',
        split='unlabel.txt',
        img_suffix='.tif',
        pipeline=test_pipeline,
        test_mode=True))
