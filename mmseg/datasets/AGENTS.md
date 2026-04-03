# DATASETS KNOWLEDGE BASE

**Scope:** Dataset classes, data loading, and preprocessing pipelines

## OVERVIEW
Dataset zoo with modular preprocessing pipeline. Primary dataset is OpenEarthMap (remote sensing). All datasets inherit from `CustomDataset` and use `@DATASETS.register_module()`. OpenEarthMap uses `CustomDataset` directly (no dedicated dataset class).

## STRUCTURE
```
mmseg/datasets/
├── pipelines/       # Transform operations
├── custom.py        # Base dataset class
├── builder.py       # Dataset/dataloader builders
└── {dataset}.py     # Specific datasets
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add new dataset | `custom.py` | Inherit `CustomDataset` |
| Add transform | `pipelines/transforms.py` | Use `@PIPELINES.register_module()` |
| Load images | `pipelines/loading.py` | `LoadImageFromFile` |
| Load annotations | `pipelines/loading.py` | `LoadAnnotations` |
| Compose pipeline | `pipelines/compose.py` | `Compose` class |
| Dataset builder | `builder.py` | `build_dataset()` |
| UDA dataset | `uda_dataset.py` | Source/target pairs |
| SSL dataset | `ssl_dataset.py` | Semi-supervised |

## DATASETS (9)
| Dataset | File | Classes | Use Case |
|---------|------|---------|----------|
| OpenEarthMap | *(uses CustomDataset)* | 7 | Remote sensing/aerial imagery (PRIMARY) |
| Cityscapes | `cityscapes.py` | 19 | Urban driving (legacy, vendorized) |
| GTA | `gta.py` | 19 | Synthetic data |
| Synthia | `synthia.py` | 16 | Synthetic data |
| ACDC | `acdc.py` | 19 | Adverse conditions |
| DarkZurich | `dark_zurich.py` | 19 | Night scenes |
| UDA | `uda_dataset.py` | - | Domain adaptation |
| SSL | `ssl_dataset.py` | - | Semi-supervised |
| Custom | `custom.py` | Configurable | Base class |

## PIPELINES

### Loading
| Transform | File | Purpose |
|-----------|------|---------|
| LoadImageFromFile | `loading.py` | Read image, to_float32 |
| LoadAnnotations | `loading.py` | Read mask, reduce_zero_label |

### Augmentation
| Transform | File | Key Args |
|-----------|------|----------|
| Resize | `transforms.py` | img_scale, ratio_range |
| RandomCrop | `transforms.py` | crop_size, cat_max_ratio |
| RandomFlip | `transforms.py` | prob, direction |
| RandomRotate | `transforms.py` | prob, degree |
| PhotoMetricDistortion | `transforms.py` | brightness, contrast, saturation |

### Processing
| Transform | File | Purpose |
|-----------|------|---------|
| Normalize | `transforms.py` | mean, std, to_rgb |
| Pad | `transforms.py` | size, pad_val |
| DefaultFormatBundle | `formating.py` | Data to tensor |
| Collect | `formating.py` | Select keys for model |

## CONVENTIONS

**Dataset Definition**:
```python
@DATASETS.register_module()
class MyDataset(CustomDataset):
    CLASSES = ('class1', 'class2', ...)
    PALETTE = [[128, 64, 128], ...]  # RGB per class
    
    def __init__(self, split, **kwargs):
        super().__init__(
            data_root='data/my_dataset/',
            img_dir='images/',
            ann_dir='annotations/',
            split=split,
            **kwargs
        )
```

**Pipeline Config**:
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
```

**Normalization** (standard):
```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],      # ImageNet
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
```

## ANTI-PATTERNS
- **NEVER** use absolute paths in dataset configs
- **NEVER** forget `CLASSES` and `PALETTE` in custom datasets
- **ALWAYS** use `Collect` at end of pipeline
- **NEVER** modify transforms in-place (return new dict)

## UNIQUE STYLES

**Multi-Scale Training**:
```python
dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0))
```

**UDA Dataset** (`uda_dataset.py`):
- Returns dict with `img`, `img_metas`, `gt_semantic_seg` for source
- Returns `target_img`, `target_img_metas` for target

**SSL Dataset** (`ssl_dataset.py`):
- Labeled + unlabeled samples
- Different pipelines for each

## BUILDER FUNCTIONS
```python
from mmseg.datasets import build_dataset, build_dataloader

dataset = build_dataset(cfg.data.train)
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=2,
    workers_per_gpu=2,
    dist=True,  # Distributed
    shuffle=True
)
```

## NOTES
- **OpenEarthMap**: Uses `CustomDataset` directly via base config. 7 classes. Data root at `data/OpenEarthMap/OpenEarthMap_flat/`. Config uses absolute path (anti-pattern).
- **Cityscapes/GTA/Synthia/ACDC/DarkZurich**: Vendorized mmseg datasets — present but not used in this project's experiments
- **Test Time Aug** (`pipelines/test_time_aug.py`): Multi-scale + flip inference
- **OHEM Sampler** (`core/seg/sampler/`): Online hard example mining via `@PIXEL_SAMPLERS.register_module()`
