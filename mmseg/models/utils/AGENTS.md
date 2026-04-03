# MODELS/UTILS KNOWLEDGE BASE

**Scope:** DAPCN core components — dynamic anchor module, prototype memory, boundary utilities

## OVERVIEW
Core Dynamic Anchor Prototype Cross-Attention Network (DAPCN) implementation. Contains the learnable prototype module, cross-batch memory bank, boundary extraction utilities, and helper functions used by DA-enabled decode heads and losses.

## STRUCTURE
```
mmseg/models/utils/
├── dynamic_anchor.py     # DynamicAnchorModule — core DA with EM refinement
├── prototype_memory.py   # PrototypeMemory — cross-batch class-conditioned bank
├── dapcn_utils.py        # Boundary extraction + GT computation helpers
├── make_divisible.py     # Channel rounding utility
├── res_layer.py          # ResNet-style layer builder
└── shape_convert.py      # NCHW ↔ NHWC conversion helpers
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add DA variant | `dynamic_anchor.py` | Modify `DynamicAnchorModule` or subclass |
| Tune prototype bank | `prototype_memory.py` | EMA decay, init strategy, per-class count |
| Add boundary operator | `dapcn_utils.py` | `extract_boundary_map` supports sobel/diff/laplacian |
| Channel math | `make_divisible.py` | `make_divisible(v, divisor, min_value)` |
| Build ResBlocks | `res_layer.py` | `ResLayer` with configurable blocks |

## KEY COMPONENTS

### DynamicAnchorModule (`dynamic_anchor.py`)
- **Registration**: `@MODELS.register_module(force=True)`
- **Purpose**: Learnable K prototypes with differentiable EM refinement and quality gating
- **Key methods**:
  - `__init__(feature_dim, max_groups, min_quality, num_iters, temperature, init_method, use_quality_gate, use_mask_predictor, ema_decay)`
  - `forward(features)` → returns `(assign_valid, proto_valid, quality_valid)`
  - `_init_prototypes()`: Xavier/kaiming/normal initialization
  - `_update_ema(proto_refined)`: EMA smoothing of prototypes
- **Config key**: `dynamic_anchor=dict(type='DynamicAnchorModule', feature_dim=..., ...)`
- **Integration**: Instantiated by decode heads via `DAPCNHeadMixin.init_dapcn()`

### PrototypeMemory (`prototype_memory.py`)
- **Registration**: Not a registered module — used directly as a Python class
- **Purpose**: Cross-batch memory bank of class-conditioned prototypes with EMA updates
- **Key methods**:
  - `__init__(num_classes, feature_dim, num_prototypes_per_class, ema, init_strategy)`
  - `update(features, labels, mask)`: EMA update of stored prototypes
  - `forward()` → `(num_classes * K, D)` tensor of all prototypes
  - `get_all_normalised()`: L2-normalized prototypes
  - `_fps_init(feats, k)`: Farthest point sampling for initial seeding
- **Standalone function**: `prototype_contrastive_loss(feat, labels, memory, temperature)` — InfoNCE contrastive loss
- **Init strategies**: 'zeros', 'random', 'fps' (farthest point sampling)

### DAPCN Utils (`dapcn_utils.py`)
- **Registration**: None — pure utility functions
- **Functions**:
  - `extract_boundary_map(logits, mode='sobel')`: Extract boundary prediction from logits. Modes: 'sobel', 'diff', 'laplacian'
  - `compute_boundary_gt(seg_label, ignore_index=255)`: Generate binary boundary ground-truth from segmentation labels using morphological erosion

## CONVENTIONS
- `DynamicAnchorModule` uses `force=True` in registration to allow override
- `da_position` config key ('before_fusion' | 'after_fusion') determines where DA features are computed in the decode head pipeline
- Prototype memory is NOT a registered module — it's instantiated directly by `DAPCNHeadMixin`
- All boundary utilities operate on tensors in (B, C, H, W) or (B, H, W) format

## ANTI-PATTERNS
- **NEVER** set `feature_dim` to mismatched value — DA output dim must match head's expected input
- **NEVER** call `PrototypeMemory.update()` before first forward pass — bank must be initialized
- **AVOID** setting `ema_decay=0.0` in DynamicAnchorModule unless you want no prototype persistence
- **ALWAYS** ensure `num_prototypes_per_class` matches between PrototypeMemory and DynamicAnchorModule configs

## DATA FLOW
```
Input features (from backbone/neck)
  → DynamicAnchorModule.forward()
    → EM refinement (num_iters iterations)
    → Quality gating (optional)
    → EMA update (optional)
  → DAPGLoss (prototype grouping)
  → PrototypeMemory.update() (EMA bank update)
  → prototype_contrastive_loss (InfoNCE against memory bank)
  → extract_boundary_map / compute_boundary_gt (boundary losses)
```
