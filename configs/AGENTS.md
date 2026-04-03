# CONFIGS KNOWLEDGE BASE

**Scope:** Experiment configurations, schedules, and model definitions

## OVERVIEW
MMSegmentation-style Python configs with inheritance. 13 OpenEarthMap experiment configs across 3 backbone families with varying train set sizes (500/1000/1500).

## STRUCTURE
```
configs/
├── _base_/              # Base configs (inherited)
│   ├── datasets/        # Dataset configs (openearthmap_val2000.py)
│   ├── models/          # Model architecture configs
│   ├── schedules/       # Optimizer + LR schedules
│   ├── default_runtime.py
│   ├── uda/             # Domain adaptation
│   └── ssl/             # Semi-supervised
└── {model}/             # Model-specific experiments
    └── {model}_openearthmap_train{N}_40k.py
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Base dataset | `_base_/datasets/openearthmap_val2000.py` | OpenEarthMap config |
| Base model | `_base_/models/segformer.py` | Model architecture |
| Base schedule | `_base_/schedules/schedule_40k_openearthmap.py` | 40k iter schedule |
| Runtime config | `_base_/default_runtime.py` | Logging, workflow |
| DA config | `_base_/uda/dapcn.py` | Dynamic anchor settings |
| DDRNet exp | `ddrnet/` | 3 OpenEarthMap configs |
| SegFormer exp | `segformer/` | 3 OpenEarthMap configs |
| UNetFormer exp | `unetformer/` | 6 OpenEarthMap configs |

## EXPERIMENTS

### OpenEarthMap (13 configs — varying train set sizes)

| Backbone | Train 500 | Train 1000 | Train 1500 | Other |
|----------|-----------|------------|------------|-------|
| DDRNet | `ddrnet/ddrnet_openearthmap_train500_40k.py` | `ddrnet/ddrnet_openearthmap_train1000_40k.py` | `ddrnet/ddrnet_openearthmap_train1500_40k.py` | — |
| SegFormer | `segformer/segformer_mit-b5_openearthmap_train500_40k.py` | `segformer/segformer_mit-b5_openearthmap_train1000_40k.py` | `segformer/segformer_mit-b5_openearthmap_train1500_40k.py` | — |
| UNetFormer | `unetformer/unetformer_openearthmap_train500_40k.py` | `unetformer/unetformer_openearthmap_train1000_40k.py` | `unetformer/unetformer_openearthmap_train1500_40k.py` | `train500_40k_stable.py`, `train500_40k_quick_eval.py`, `train1000_40k_resnext101_32x16d.py` |

### Base Configs

| Category | Files | Notes |
|----------|-------|-------|
| Datasets | `_base_/datasets/openearthmap_val2000.py` | Uses absolute `data_root` (anti-pattern) |
| Models | `_base_/models/segformer.py`, `_base_/models/ddrnet.py`, `_base_/models/unetformer.py` | 17 total model templates |
| Schedules | `_base_/schedules/schedule_40k_openearthmap.py`, `_base_/schedules/adamw.py`, `_base_/schedules/poly10warm.py`, `_base_/schedules/poly10.py` | OpenEarthMap uses 40k iterations |
| Runtime | `_base_/default_runtime.py` | TextLoggerHook, interval 50 |

## CONVENTIONS

**Config Inheritance**:
```python
_base_ = [
    '../_base_/models/segformer.py',
    '../_base_/datasets/openearthmap_val2000.py', 
    '../_base_/schedules/schedule_40k_openearthmap.py',
    '../_base_/default_runtime.py'
]

# Override specific keys in child config
data = dict(
    train=dict(split='splits/train_1000.txt'),
)
```

**Relative Paths**: Always use `../` relative to config location:
```python
_base_ = ['../_base_/models/ddrnet.py']  # From ddrnet/ddrnet_*.py
```

## KEY CONFIG SECTIONS

### Model Config
```python
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(...),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        channels=256,
        num_classes=7,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # or 'slide'
)
```

### Data Config
```python
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/train',
        ann_dir='labels/train',
        split='splits/train_1000.txt',
        pipeline=train_pipeline
    ),
    val=dict(...),
    test=dict(...)
)
```

### Schedule Config
```python
# Optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

# LR schedule
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# Runner
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
```

## ANTI-PATTERNS
- **NEVER** use absolute paths in configs
- **NEVER** forget `by_epoch=False` for iter-based training
- **ALWAYS** use `../../` prefix when referencing base configs from deep paths
- **NEVER** modify `_base_` configs directly; override in child configs
- **WARNING**: `_base_/datasets/openearthmap_val2000.py` has absolute `data_root` = `/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/`
- **WARNING**: Several configs use `pretrained=` directly — use `init_cfg` instead

## UNIQUE STYLES

**DA Config Keys**:
```python
decode_head=dict(
    da_position='before_fusion',  # or 'after_fusion'
    dynamic_anchor=dict(
        num_prototypes=10,
        feat_channels=256,
        # ...
    ),
    dapg_loss=dict(
        temperature=0.1,
        loss_weight=0.1
    )
)
```

**Config Generation**:
- `run_experiments.py` creates configs in `configs/generated/`
- Unique names: `{timestamp}_{cfg_name}_{uuid5}`
- Auto-generates work directories

## COMMANDS
```bash
# Activate virtual environment first
source .venv/bin/activate

# Train with config
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py

# Override config values
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py \
    --cfg-options optimizer.lr=0.0001

# Resume training
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py \
    --resume-from work_dirs/latest.pth

# Auto-resume
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py \
    --auto-resume
```

## NOTES
- **Config system**: Uses MMCV Config (Python files)
- **Inheritance**: Child configs override parent values
- **Work dir**: Auto-set to `work_dirs/{config_name}/` unless specified
- **Log interval**: 50 iterations (from `default_runtime.py`)
- **Checkpoint interval**: 40000 iterations (model-specific)
- **Eval interval**: 4000 iterations (model-specific)
