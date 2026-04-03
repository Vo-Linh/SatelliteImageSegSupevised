# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-03
**Commit:** ec2f2dd
**Branch:** feature/openearthmap-experiments

## OVERVIEW
Satellite Image Segmentation with Dynamic Anchor Prototype Cross-Attention Network (DAPCN). Based on vendorized MMSegmentation v0.16.0. Implements prototype learning with quality gating for improved boundary detection. Focuses on remote sensing with OpenEarthMap dataset across 3 backbone families.

## STRUCTURE
```
./
├── mmseg/           # Core framework (apis, core, datasets, models, ops) — vendorized v0.16.0
├── configs/         # Experiment configs (_base_ templates + model-specific)
│   ├── _base_/      # Inherited base configs (models, datasets, schedules)
│   └── {model}/     # Per-model experiments (OpenEarthMap)
├── tools/           # Training/testing scripts (MMSeg standard)
├── run_experiments.py      # Batch experiment runner (BROKEN: missing experiments.py)
└── test_contrastive_loss.py # Standalone loss validation script
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new backbone | `mmseg/models/backbones/` | Use `@BACKBONES.register_module()` |
| Add new decode head | `mmseg/models/decode_heads/` | Inherit `BaseDecodeHead` |
| Add new loss | `mmseg/models/losses/` | Use `@LOSSES.register_module()` |
| Add new dataset | `mmseg/datasets/` | Inherit `CustomDataset` |
| Add transform | `mmseg/datasets/pipelines/` | Use `@PIPELINES.register_module()` |
| Add DA component | `mmseg/models/utils/` | See `mmseg/models/utils/AGENTS.md` |
| Training | `tools/train.py` | Single-GPU entry point |
| Distributed training | `tools/dist_train.sh` | DDP launcher (torch.distributed.launch) |
| Testing | `tools/test.py` | Evaluation script |
| Config templates | `configs/_base_/` | Inherit via `_base_` list |

## ENTRY POINTS
```bash
# Single GPU training
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py

# Multi-GPU training
bash tools/dist_train.sh <config> <num_gpus>

# Test/Evaluate
python tools/test.py <config> <checkpoint> --eval mIoU
```

## CONVENTIONS

**Registration Pattern**: All components use decorators:
```python
@BACKBONES.register_module()
@HEADS.register_module()
@LOSSES.register_module()
@DATASETS.register_module()
@PIPELINES.register_module()
@PIXEL_SAMPLERS.register_module()
@SEGMENTORS.register_module()
```

**Config Inheritance**: Use `_base_` list to inherit:
```python
_base_ = [
    '../_base_/models/segformer.py',
    '../_base_/datasets/openearthmap_val2000.py',
    '../_base_/schedules/schedule_40k_openearthmap.py',
    '../_base_/default_runtime.py'
]
```

**Model Pipeline**: Backbones → Necks (optional) → DecodeHeads → Segmentors (EncoderDecoder) → Output

**Checkpoint Format**: Saves meta dict with `mmseg_version`, `CLASSES`, `PALETTE`

## ANTI-PATTERNS
- **NEVER** mix `init_cfg` and `pretrained` parameters (mutual exclusion — assertion error)
- **NEVER** use `pretrained` in new code (deprecated, use `init_cfg` instead)
- **ALWAYS** use `by_epoch=False` for iter-based training configs
- **AVOID** absolute paths in configs; use relative paths with `../../`
- **NEVER** modify `_base_` configs directly; override in child configs
- **WARNING**: `configs/_base_/datasets/openearthmap_val2000.py` uses absolute `data_root` — violates convention

## UNIQUE STYLES

**Dynamic Anchor Components** (see `mmseg/models/utils/AGENTS.md`):
- `mmseg/models/utils/dynamic_anchor.py` — Core DA module (EM refinement, quality gating)
- `mmseg/models/utils/prototype_memory.py` — Cross-batch class-conditioned prototype bank
- `mmseg/models/utils/dapcn_utils.py` — Boundary extraction and GT computation
- `mmseg/models/decode_heads/dapcn_head_mixin.py` — Mixin wiring DA losses into any head
- `mmseg/models/losses/dapg_loss.py` — Dynamic anchor prototype grouping loss
- `mmseg/models/losses/affinity_boundary_loss.py` — Affinity-based boundary detection loss

**Experiment Naming**: Auto-generated with timestamp + UUID:
```
work_dirs/{machine}-{exp_name}/{yymmdd_HHMM}_{config_name}_{uuid5}
```

**Config Generation**: `configs/generated/` — target for JSON configs from `run_experiments.py` (currently empty — script broken)

**TIMMBackbone**: UNetFormer configs refactored to use `timm_backbone.py` wrapper instead of custom backbone

## COMMANDS
```bash
# Activate virtual environment first
source .venv/bin/activate

# Install dependencies
pip install -e .

# Single GPU training
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py

# Multi-GPU (4 GPUs)
bash tools/dist_train.sh configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py 4

# Test/Evaluate
python tools/test.py <config> <checkpoint> --eval mIoU
```

## NOTES
- **Broken import**: `run_experiments.py` imports `experiments.py` which doesn't exist — batch config generation non-functional
- **No test suite**: No pytest/tox/CI. Only `tools/test.py` (model eval) and `test_contrastive_loss.py` (standalone demo)
- **No CI/CD**: No GitHub Actions, Jenkins, or linting configs
- **Logging**: TextLoggerHook to `work_dirs/`; Tensorboard commented out in `default_runtime.py`
- **MMCV version**: Requires 1.3.7–1.7.2 (enforced in `mmseg/__init__.py`)
- **Work dirs**: Auto-created at `work_dirs/{exp_name}/{unique_name}/`
- **configs/generated/**: Empty — would be populated by broken `run_experiments.py`
- **Pretrained anti-pattern**: Several configs use `pretrained=` directly instead of `init_cfg`
- **TODO**: `encoder_decoder.py:174` has `# TODO refactor` marker

## DEPENDENCIES
```
torch>=1.8
mmcv-full>=1.3.7
timm
```

## FRAMEWORK VERSION
MMSegmentation v0.16.0
