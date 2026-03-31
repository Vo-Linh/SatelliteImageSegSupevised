# OpenEarthMap Integration into MMSegmentation

## TL;DR

> **Quick Summary**: Integrate OpenEarthMap 9-class dataset support into MMSegmentation with base dataset config, 40k schedule, and training configs for SegFormer-B5, UNetFormer, and DDRNet (500/1000/1500 samples each).
>
> **Deliverables**:
> - Base dataset config: `configs/_base_/datasets/openearthmap_val2000.py`
> - Schedule config: `configs/_base_/schedules/schedule_40k_openearthmap.py`
> - Training configs: 9 total — 3x SegFormer-B5, 3x UNetFormer, 3x DDRNet (train500, train1000, train1500)
>
> **Estimated Effort**: Medium (~11 files, well-patterned)
> **Parallel Execution**: YES - Wave 1: base files; Wave 2: model configs
> **Critical Path**: Create base configs (dataset + schedule) → Create model configs

---

## Context

### Original Request
User provided `docs/OPENEARTHMAP_TRAINING_GUIDE.md` and wants OpenEarthMap dataset integrated into the MMSegmentation codebase at `/home/ubuntu/mmsegmentation`. Requested configs for SegFormer, UNetFormer, and DDRNet.

### Research Findings
- **Best reference**: `mmseg/datasets/cityscapes.py` — custom dataset class pattern with CLASSES and PALETTE
- **Dataset class base**: `mmseg/datasets/custom.py` — CustomDataset base class
- **Existing OEM reference**: `configs/_base_/datasets/uda_oem_512x512.py` — existing OpenEarthMap UDA variant
- **Model configs**: Existing `configs/segformer/`, `configs/ddrnet/`, `configs/unetformer/` show clear inheritance patterns
- **Schedule pattern**: `configs/_base_/schedules/poly10warm.py` — standard poly LR with warmup
- **DDRNet pattern**: `configs/ddrnet/ddrnet_dapcn_before_fusion_cityscapes.py`
- **UNetFormer pattern**: `configs/unetformer/unetformer_dapcn_before_fusion_cityscapes.py`
- **SegFormer-B5**: Uses `mit_b5` backbone with pretrained weights

### Metis Review
**Identified Gaps (addressed)**:
- Scope expanded: SegFormer + UNetFormer + DDRNet (3 models × 3 sample sizes = 9 configs)
- Data path: Hardcoded from guide (`/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/`)
- Batch size: Default 4 (can be overridden per GPU)
- Image format: `.tif` (confirmed from guide)

---

## Work Objectives

### Core Objective
Add OpenEarthMap dataset support to MMSegmentation enabling training with 500/1000/1500 samples and validation on fixed 2000 samples across 3 model architectures.

### Concrete Deliverables
1. `mmseg/datasets/openearthmap.py` — Custom OpenEarthMapDataset class with 9 classes and palette
2. `mmseg/datasets/__init__.py` — Update to export OpenEarthMapDataset
3. `configs/_base_/datasets/openearthmap_val2000.py` — Base dataset config using OpenEarthMapDataset
4. `configs/_base_/schedules/schedule_40k_openearthmap.py` — 40k iteration schedule with poly LR decay
5. `configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py` — SegFormer-B5 on 500 samples
6. `configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py` — SegFormer-B5 on 1000 samples
7. `configs/segformer/segformer_mit-b5_openearthmap_train1500_40k.py` — SegFormer-B5 on 1500 samples
8. `configs/unetformer/unetformer_openearthmap_train500_40k.py` — UNetFormer on 500 samples
9. `configs/unetformer/unetformer_openearthmap_train1000_40k.py` — UNetFormer on 1000 samples
10. `configs/unetformer/unetformer_openearthmap_train1500_40k.py` — UNetFormer on 1500 samples
11. `configs/ddrnet/ddrnet_openearthmap_train500_40k.py` — DDRNet on 500 samples
12. `configs/ddrnet/ddrnet_openearthmap_train1000_40k.py` — DDRNet on 1000 samples
13. `configs/ddrnet/ddrnet_openearthmap_train1500_40k.py` — DDRNet on 1500 samples

### Definition of Done
- [ ] `python -c "from mmcv import Config; Config.fromfile('configs/_base_/datasets/openearthmap_val2000.py')"` — loads without error
- [ ] `python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py` — starts training
- [ ] All 3 configs inherit correctly and can build datasets

### Must Have
- 9 classes with exact palette from training guide
- Train pipelines with data augmentation (resize, crop, flip, photometric)
- Val/test pipelines without augmentation
- 40k iterations, poly LR decay, SGD optimizer
- Annotation file support for train splits

### Must NOT Have
- Modifications to existing dataset configs
- Non-standard dependencies or preprocessing
- Test-time augmentation in base config

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (MMSeg uses config-driven evaluation, no pytest)
- **Automated tests**: None (use tools/test.py for model evaluation)
- **QA Policy**: Agent-executed verification of config loading and dataset instantiation

### QA Policy
Every task includes agent-executed QA scenarios — no human intervention required.

---

## Execution Strategy

### Parallel Execution Waves

> Two waves for dependency management. Wave 1 creates base configs (needed by all model configs). Wave 2 creates all 9 model configs in parallel.

```
Wave 1 (Foundation — must complete before Wave 2):
├── Task 1: Create custom dataset class mmseg/datasets/openearthmap.py
├── Task 2: Update mmseg/datasets/__init__.py to export OpenEarthMapDataset
└── Task 3: Create base dataset config

Wave 2 (Depends on Wave 1):
├── Task 4: Create schedule config

Wave 3 (Model configs — depends on Wave 1):
├── Task 5: Create SegFormer train500 config
├── Task 6: Create SegFormer train1000 config
├── Task 7: Create SegFormer train1500 config
├── Task 8: Create UNetFormer train500 config
├── Task 9: Create UNetFormer train1000 config
├── Task 10: Create UNetFormer train1500 config
├── Task 11: Create DDRNet train500 config
├── Task 12: Create DDRNet train1000 config
└── Task 13: Create DDRNet train1500 config
```

### Agent Dispatch Summary
- **Wave 1**: **3** — T1-T3 → `quick` (dataset class + config)
- **Wave 2**: **1** — T4 → `quick` (schedule)
- **Wave 3**: **9** — T5-T13 → `quick` (model configs, parallel)

---

## TODOs

- [x] 1. Create custom dataset class `mmseg/datasets/openearthmap.py`

  **What to do**:
  - Create `mmseg/datasets/openearthmap.py` with OpenEarthMapDataset class
  - Inherit from CustomDataset
  - Define CLASSES tuple with 9 class names: 'class_0' through 'class_8'
  - Define PALETTE list with RGB values from training guide: [[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128], [0,128,128], [128,128,128], [64,0,0]]
  - Set img_suffix='.tif', seg_map_suffix='.tif' in __init__
  - Call super().__init__ with appropriate parameters

  **Must NOT do**:
  - Do NOT add custom processing beyond CLASSES/PALETTE definition
  - Do NOT modify existing dataset files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: Simple dataset class following existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: All other tasks (all configs depend on dataset class)
  - **Blocked By**: None

  **References**:
  - `mmseg/datasets/cityscapes.py` — Custom dataset class pattern with CLASSES and PALETTE
  - `mmseg/datasets/custom.py` — CustomDataset base class

  **Acceptance Criteria**:
  - [ ] File created: mmseg/datasets/openearthmap.py
  - [ ] Class name: OpenEarthMapDataset
  - [ ] CLASSES defined with 9 classes
  - [ ] PALETTE defined with 9 RGB triplets
  - [ ] img_suffix='.tif', seg_map_suffix='.tif'
  - [ ] Imports: from .builder import DATASETS, from .custom import CustomDataset

  **QA Scenarios**:

  ```
  Scenario: Dataset class imports without error
    Tool: Bash
    Preconditions: File exists
    Steps:
      1. Run: python -c "from mmseg.datasets import OpenEarthMapDataset; print('OpenEarthMapDataset imported successfully')"
    Expected Result: OpenEarthMapDataset imported successfully
    Evidence: .sisyphus/evidence/task-1-dataset-class-import.txt

  Scenario: Dataset class has correct CLASSES and PALETTE
    Tool: Bash
    Preconditions: Class imports
    Steps:
      1. Run: python -c "from mmseg.datasets import OpenEarthMapDataset; print('Classes:', OpenEarthMapDataset.CLASSES); print('Palette length:', len(OpenEarthMapDataset.PALETTE))"
    Expected Result: Classes tuple with 9 elements, Palette list with 9 elements
    Evidence: .sisyphus/evidence/task-1-dataset-class-attributes.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 2. Update `mmseg/datasets/__init__.py` to export OpenEarthMapDataset

  **What to do**:
  - Read `mmseg/datasets/__init__.py`
  - Add import: `from .openearthmap import OpenEarthMapDataset`
  - Add `OpenEarthMapDataset` to `__all__` list

  **Must NOT do**:
  - Do NOT modify other imports or exports
  - Do NOT remove existing entries from __all__

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: Simple file edit to add import and export

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: All other tasks
  - **Blocked By**: Task 1

  **References**:
  - `mmseg/datasets/__init__.py` — Where to add import and export

  **Acceptance Criteria**:
  - [ ] Import added: `from .openearthmap import OpenEarthMapDataset`
  - [ ] Export added: `OpenEarthMapDataset` in `__all__`

  **QA Scenarios**:

  ```
  Scenario: Dataset can be imported from mmseg.datasets
    Tool: Bash
    Steps:
      1. Run: python -c "from mmseg.datasets import OpenEarthMapDataset; print('Successfully imported from mmseg.datasets')"
    Expected Result: No import error
    Evidence: .sisyphus/evidence/task-2-init-import.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 3. Create base dataset config `configs/_base_/datasets/openearthmap_val2000.py`

  **What to do**:
  - Create `configs/_base_/datasets/openearthmap_val2000.py`
  - Use dataset_type = 'OpenEarthMapDataset' (the custom class)
  - Define train_pipeline with augmentation (RandomResize, RandomCrop, RandomFlip, PhotoMetricDistortion)
  - Define test_pipeline without augmentation (just LoadImage, Resize, LoadAnnotations, PackSegInputs)
  - Define train_dataloader with batch_size=4, InfiniteSampler
  - Define val_dataloader with batch_size=1, DefaultSampler, ann_file pointing to val_2000_fixed.txt
  - Use img_suffix='.tif', seg_map_suffix='.tif'
  - Set ignore_index=255
  - Data root: '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'

  **Must NOT do**:
  - Do NOT add non-standard augmentation
  - Do NOT use absolute paths outside data_root

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: Simple file creation from existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 5-13 (all model configs depend on dataset config)
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `configs/_base_/datasets/cityscapes_half_512x512.py` — Standard dataset pipeline pattern
  - `configs/_base_/datasets/ssl_satellite_512x512.py` — Custom dataset class usage pattern

  **Acceptance Criteria**:
  - [ ] File created: configs/_base_/datasets/openearthmap_val2000.py
  - [ ] dataset_type = 'OpenEarthMapDataset'
  - [ ] data_root = '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'
  - [ ] train_pipeline includes: LoadImageFromFile, LoadAnnotations, RandomResize, RandomCrop, RandomFlip, PhotoMetricDistortion, PackSegInputs
  - [ ] test_pipeline includes: LoadImageFromFile, Resize, LoadAnnotations, PackSegInputs
  - [ ] train_dataloader uses batch_size=4, InfiniteSampler
  - [ ] val_dataloader uses batch_size=1, ann_file for val_2000_fixed.txt

  **QA Scenarios**:

  ```
  Scenario: Config loads without syntax error
    Tool: Bash
    Preconditions: File exists at correct path
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/_base_/datasets/openearthmap_val2000.py'); print('Config loaded successfully')"
    Expected Result: Config loaded successfully
    Evidence: .sisyphus/evidence/task-3-config-load.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 4. Create schedule config `configs/_base_/schedules/schedule_40k_openearthmap.py`

  **What to do**:
  - Create `configs/_base_/schedules/schedule_40k_openearthmap.py`
  - Define optimizer: SGD with lr=0.01, momentum=0.9, weight_decay=0.0005
  - Define optim_wrapper with gradient clipping (max_norm=1.0, norm_type=2)
  - Define param_scheduler: PolyLR with power=0.9, eta_min=0, begin=0, end=80000
  - Define train_cfg: IterBasedTrainLoop with max_iters=80000, val_interval=4000
  - Define val_cfg: ValLoop
  - Define test_cfg: TestLoop
  - Define default_hooks: IterTimerHook, LoggerHook (interval=50), ParamSchedulerHook, CheckpointHook (interval=4000, save_best='mIoU', rule='greater', max_keep_ckpts=2), DistSamplerSeedHook, SegVisualizationHook

  **Must NOT do**:
  - Do NOT use by_epoch=True (must be False for iter-based)
  - Do NOT use Adam or AdamW (guide specifies SGD)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: Simple file creation from existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `configs/_base_/schedules/poly10warm.py` — Standard poly LR schedule pattern
  - `configs/_base_/schedules/adamw.py` — Optimizer config reference
  - Training guide specifies: SGD, lr=0.01, momentum=0.9, weight_decay=0.0005, poly power=0.9, 80k iterations

  **Acceptance Criteria**:
  - [ ] File created: configs/_base_/schedules/schedule_40k_openearthmap.py
  - [ ] Optimizer: SGD with lr=0.01, momentum=0.9, weight_decay=0.0005
  - [ ] Clip grad: max_norm=1.0, norm_type=2
  - [ ] PolyLR: power=0.9, eta_min=0, end=80000
  - [ ] max_iters=80000, val_interval=4000
  - [ ] CheckpointHook: interval=4000, save_best='mIoU', rule='greater', max_keep_ckpts=2

  **QA Scenarios**:

  ```
  Scenario: Schedule config loads without error
    Tool: Bash
    Preconditions: File exists
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/_base_/schedules/schedule_40k_openearthmap.py'); print('Schedule loaded:', cfg.max_iters)"
    Expected Result: Prints max_iters=80000
    Evidence: .sisyphus/evidence/task-4-schedule-load.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 5. Create SegFormer-B5 train500 config

  **What to do**:
  - Create `configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py`
  - Use _base_ inheritance: segformer_mit-b5.py model, openearthmap_val2000.py dataset, default_runtime.py, schedule_40k_openearthmap.py schedule
  - Override model data_preprocessor: SegDataPreProcessor with size=(512, 512), ImageNet mean/std, bgr_to_rgb=True, pad_val=0, seg_pad_val=255
  - Override decode_head: num_classes=9, CrossEntropyLoss
  - Override train_dataloader.dataset.ann_file: train_500_fixed.txt path
  - Set work_dir: ./work_dirs/openearthmap/segformer_train500

  **Must NOT do**:
  - Do NOT use init_cfg and pretrained together (mutual exclusion)
  - Do NOT use pretrained parameter (deprecated, use init_cfg)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: File creation from existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6-13)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2, 3, 4 (needs dataset class, init, dataset config, schedule)

  **References**:
  - `configs/segformer/segformer_dapcn_before_fusion_cityscapes.py` — SegFormer config structure with _base_ inheritance
  - `configs/_base_/models/segformer_b5.py` — SegFormer-B5 model config
  - `configs/_base_/default_runtime.py` — Runtime config

  **Acceptance Criteria**:
  - [ ] File created: configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py
  - [ ] _base_ list includes: ../_base_/default_runtime.py, ../_base_/datasets/openearthmap_val2000.py, ../_base_/schedules/schedule_40k_openearthmap.py
  - [ ] model.backbone type='mit_b5'
  - [ ] model.decode_head.num_classes=9
  - [ ] train_dataloader.dataset.ann_file points to train_500_fixed.txt
  - [ ] work_dir set to ./work_dirs/openearthmap/segformer_train500

  **QA Scenarios**:

  ```
  Scenario: Config loads and shows correct train ann_file
    Tool: Bash
    Preconditions: File exists, Tasks 1-4 complete
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py'); print('Train ann_file:', cfg.data.train.dataset.ann_file); print('num_classes:', cfg.model.decode_head.num_classes)"
    Expected Result: Shows path to train_500_fixed.txt and num_classes=9
    Evidence: .sisyphus/evidence/task-5-train500-load.txt

  Scenario: Config can build model without error
    Tool: Bash
    Preconditions: Config loads
    Steps:
      1. Run: python -c "from mmcv import Config; from mmseg.models import build_model; cfg = Config.fromfile('configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py'); model = build_model(cfg.model); print('Model built successfully')"
    Expected Result: Model built successfully
    Evidence: .sisyphus/evidence/task-5-train500-model.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 6. Create SegFormer-B5 train1000 config

  **What to do**:
  - Create `configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py`
  - Same structure as train500 config, but:
    - work_dir: ./work_dirs/openearthmap/segformer_train1000
    - ann_file: train_1000_fixed.txt path

  **Must NOT do**:
  - Same as Task 5

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: File creation from existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 7-13)
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:
  - Same as Task 5
  - Training guide: train_1000_fixed.txt for 1000 samples

  **Acceptance Criteria**:
  - [ ] File created: configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py
  - [ ] _base_ list identical to train500
  - [ ] ann_file points to train_1000_fixed.txt
  - [ ] work_dir: ./work_dirs/openearthmap/segformer_train1000

  **QA Scenarios**:

  ```
  Scenario: Config loads with train1000 ann_file
    Tool: Bash
    Preconditions: File exists, Tasks 1-4 complete
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py'); print('Train ann_file:', cfg.data.train.dataset.ann_file)"
    Expected Result: Shows path to train_1000_fixed.txt
    Expected Result: Shows path to train_1000_fixed.txt
    Evidence: .sisyphus/evidence/task-4-train1000-load.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 7. Create SegFormer-B5 train1500 config

  **What to do**:
  - Create `configs/segformer/segformer_mit-b5_openearthmap_train1500_40k.py`
  - Same structure as train500 config, but:
    - work_dir: ./work_dirs/openearthmap/segformer_train1500
    - ann_file: train_1500_fixed.txt path

  **Must NOT do**:
  - Same as Task 3

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: File creation from existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2 (needs dataset and schedule configs)

  **References**:
  - Same as Task 3
  - Training guide: train_1500_fixed.txt for 1500 samples

  **Acceptance Criteria**:
  - [ ] File created: configs/segformer/segformer_mit-b5_openearthmap_train1500_40k.py
  - [ ] _base_ list identical to train500
  - [ ] ann_file points to train_1500_fixed.txt
  - [ ] work_dir: ./work_dirs/openearthmap/segformer_train1500

  **QA Scenarios**:

  ```
  Scenario: Config loads with train1500 ann_file
    Tool: Bash
    Preconditions: File exists, Tasks 1 and 2 complete
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/segformer/segformer_mit-b5_openearthmap_train1500_40k.py'); print('Train ann_file:', cfg.data.train.dataset.ann_file)"
    Expected Result: Shows path to train_1500_fixed.txt
    Evidence: .sisyphus/evidence/task-5-train1500-load.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 8. Create UNetFormer train500 config

  **What to do**:
  - Create `configs/unetformer/unetformer_openearthmap_train500_40k.py`
  - Follow pattern from `configs/unetformer/unetformer_dapcn_before_fusion_cityscapes.py`
  - _base_ = ['../_base_/default_runtime.py', '../_base_/datasets/openearthmap_val2000.py', '../_base_/schedules/schedule_40k_openearthmap.py']
  - Model: backbone=dict(type='UNetFormer'), decode_head=dict(type='UNetFormerDAPCNHead', num_classes=9)
  - work_dir: ./work_dirs/openearthmap/unetformer_train500
  - Override train_dataloader.dataset.ann_file to point to train_500_fixed.txt

  **Must NOT do**:
  - Do NOT include DAPCN parameters beyond what's in the reference (keep same structure)
  - Do NOT use init_cfg and pretrained together

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-11)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `configs/unetformer/unetformer_dapcn_before_fusion_cityscapes.py` — UNetFormer structure with DAPCN
  - `configs/_base_/datasets/ssl_satellite_512x512.py` — Custom classes/palette pattern

  **Acceptance Criteria**:
  - [ ] File created: configs/unetformer/unetformer_openearthmap_train500_40k.py
  - [ ] backbone type='UNetFormer', decode_head type='UNetFormerDAPCNHead'
  - [ ] num_classes=9
  - [ ] ann_file points to train_500_fixed.txt

  **QA Scenarios**:

  ```
  Scenario: Config loads with train500 ann_file
    Tool: Bash
    Preconditions: File exists, Tasks 1 and 2 complete
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/unetformer/unetformer_openearthmap_train500_40k.py'); print('Train ann_file:', cfg.data.train.dataset.ann_file)"
    Expected Result: Shows path to train_500_fixed.txt
    Evidence: .sisyphus/evidence/task-6-unetformer-train500-load.txt
  ```

  **Commit**: NO (batch with other files)

- [x] 9. Create UNetFormer train1000 config

  **What to do**:
  - Create `configs/unetformer/unetformer_openearthmap_train1000_40k.py`
  - Same as train500 but:
    - work_dir: ./work_dirs/openearthmap/unetformer_train1000
    - ann_file: train_1000_fixed.txt

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-6, 8-11)
  - **Blocked By**: Tasks 1, 2

  **Acceptance Criteria**:
  - [ ] File created: configs/unetformer/unetformer_openearthmap_train1000_40k.py
  - [ ] ann_file points to train_1000_fixed.txt

  **QA Scenarios**:

  ```
  Scenario: Config loads with train1000 ann_file
    Tool: Bash
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/unetformer/unetformer_openearthmap_train1000_40k.py'); print('Train ann_file:', cfg.data.train.dataset.ann_file)"
    Expected Result: Shows path to train_1000_fixed.txt
    Evidence: .sisyphus/evidence/task-7-unetformer-train1000-load.txt
  ```

  **Commit**: NO

- [x] 10. Create UNetFormer train1500 config

  **What to do**:
  - Create `configs/unetformer/unetformer_openearthmap_train1500_40k.py`
  - Same as train500 but:
    - work_dir: ./work_dirs/openearthmap/unetformer_train1500
    - ann_file: train_1500_fixed.txt

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-7, 9-11)
  - **Blocked By**: Tasks 1, 2

  **Acceptance Criteria**:
  - [ ] File created: configs/unetformer/unetformer_openearthmap_train1500_40k.py
  - [ ] ann_file points to train_1500_fixed.txt

  **Commit**: NO

- [x] 11. Create DDRNet train500 config

  **What to do**:
  - Create `configs/ddrnet/ddrnet_openearthmap_train500_40k.py`
  - Follow pattern from `configs/ddrnet/ddrnet_dapcn_before_fusion_cityscapes.py`
  - _base_ = ['../_base_/default_runtime.py', '../_base_/datasets/openearthmap_val2000.py', '../_base_/schedules/schedule_40k_openearthmap.py']
  - Model: backbone=dict(type='DDRNet'), decode_head=dict(type='DDRNetDAPCNHead', num_classes=9)
  - work_dir: ./work_dirs/openearthmap/ddrnet_train500
  - Override train_dataloader.dataset.ann_file to point to train_500_fixed.txt

  **Must NOT do**:
  - Do NOT include DAPCN parameters beyond what's in the reference (keep same structure)
  - Do NOT use init_cfg and pretrained together

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-8, 10-11)
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `configs/ddrnet/ddrnet_dapcn_before_fusion_cityscapes.py` — DDRNet structure with DAPCN

  **Acceptance Criteria**:
  - [ ] File created: configs/ddrnet/ddrnet_openearthmap_train500_40k.py
  - [ ] backbone type='DDRNet', decode_head type='DDRNetDAPCNHead'
  - [ ] num_classes=9
  - [ ] ann_file points to train_500_fixed.txt

  **QA Scenarios**:

  ```
  Scenario: Config loads with train500 ann_file
    Tool: Bash
    Steps:
      1. Run: python -c "from mmcv import Config; cfg = Config.fromfile('configs/ddrnet/ddrnet_openearthmap_train500_40k.py'); print('Train ann_file:', cfg.data.train.dataset.ann_file)"
    Expected Result: Shows path to train_500_fixed.txt
    Evidence: .sisyphus/evidence/task-9-ddrnet-train500-load.txt
  ```

  **Commit**: NO

- [x] 12. Create DDRNet train1000 config

  **What to do**:
  - Create `configs/ddrnet/ddrnet_openearthmap_train1000_40k.py`
  - Same as train500 but:
    - work_dir: ./work_dirs/openearthmap/ddrnet_train1000
    - ann_file: train_1000_fixed.txt

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-9, 11)
  - **Blocked By**: Tasks 1, 2

  **Acceptance Criteria**:
  - [ ] File created: configs/ddrnet/ddrnet_openearthmap_train1000_40k.py
  - [ ] ann_file points to train_1000_fixed.txt

  **Commit**: NO

- [x] 13. Create DDRNet train1500 config

  **What to do**:
  - Create `configs/ddrnet/ddrnet_openearthmap_train1500_40k.py`
  - Same as train500 but:
    - work_dir: ./work_dirs/openearthmap/ddrnet_train1500
    - ann_file: train_1500_fixed.txt

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-10)
  - **Blocked By**: Tasks 1, 2

  **Acceptance Criteria**:
  - [ ] File created: configs/ddrnet/ddrnet_openearthmap_train1500_40k.py
  - [ ] ann_file points to train_1500_fixed.txt

  **Commit**: NO

---

## Final Verification Wave

- [ ] F1. **Config Loading Verification** — `quick`
  Verify all 11 config files load without error.
  Output: `All configs [11/11] load successfully`

- [ ] F2. **Inheritance Chain Verification** — `quick`
  Verify each training config's _base_ chain resolves correctly.
  Output: `All [9/9] configs inherit from base correctly`

- [ ] F3. **Dataset Instantiation Check** — `quick`
  Attempt to build train dataset from each config (data may not exist, but config structure should be valid).
  Output: `Dataset build [N/9] with expected failures due to missing data`

- [ ] F4. **File Existence Check** — `quick`
  Verify all expected files exist at correct paths.
  Output: `Files [11/11] exist`

---

## Commit Strategy

- **Batch commit after all tasks complete**:
  - Message: `feat: add OpenEarthMap dataset support with SegFormer-B5, UNetFormer, and DDRNet configs`
  - Files:
    - configs/_base_/datasets/openearthmap_val2000.py
    - configs/_base_/schedules/schedule_40k_openearthmap.py
    - configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py
    - configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py
    - configs/segformer/segformer_mit-b5_openearthmap_train1500_40k.py
    - configs/unetformer/unetformer_openearthmap_train500_40k.py
    - configs/unetformer/unetformer_openearthmap_train1000_40k.py
    - configs/unetformer/unetformer_openearthmap_train1500_40k.py
    - configs/ddrnet/ddrnet_openearthmap_train500_40k.py
    - configs/ddrnet/ddrnet_openearthmap_train1000_40k.py
    - configs/ddrnet/ddrnet_openearthmap_train1500_40k.py

---

## Success Criteria

### Verification Commands
```bash
# All configs should load
python -c "from mmcv import Config; [Config.fromfile(f) for f in [
  'configs/_base_/datasets/openearthmap_val2000.py',
  'configs/_base_/schedules/schedule_40k_openearthmap.py',
  'configs/segformer/segformer_mit-b5_openearthmap_train500_40k.py',
  'configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py',
  'configs/segformer/segformer_mit-b5_openearthmap_train1500_40k.py',
  'configs/unetformer/unetformer_openearthmap_train500_40k.py',
  'configs/unetformer/unetformer_openearthmap_train1000_40k.py',
  'configs/unetformer/unetformer_openearthmap_train1500_40k.py',
  'configs/ddrnet/ddrnet_openearthmap_train500_40k.py',
  'configs/ddrnet/ddrnet_openearthmap_train1000_40k.py',
  'configs/ddrnet/ddrnet_openearthmap_train1500_40k.py'
]]; print('All 11 configs load successfully')"
```

### Final Checklist
- [ ] All 11 files created at correct paths
- [ ] All configs load without syntax/import errors
- [ ] 9 classes defined with correct palette
- [ ] 40k iteration schedule configured correctly
- [ ] Train configs point to correct annotation files (500/1000/1500)
- [ ] UNetFormer configs use UNetFormerDAPCNHead
- [ ] DDRNet configs use DDRNetDAPCNHead
- [ ] Training can be started with `python tools/train.py <config>`

