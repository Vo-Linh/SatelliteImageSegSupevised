# MODELS KNOWLEDGE BASE

**Scope:** Neural network architectures, backbones, heads, losses, and utilities

## OVERVIEW
Model zoo with 9+ backbones, 8+ decode heads, and custom Dynamic Anchor (DAPCN) components. All components use `@register_module()` decorator pattern. UNetFormer backbone recently refactored to use TIMMBackbone wrapper.

## STRUCTURE
```
mmseg/models/
├── backbones/       # Feature extractors
├── decode_heads/    # Segmentation heads
├── losses/          # Loss functions
├── necks/           # Feature fusion
├── segmentors/      # End-to-end models
└── utils/           # Shared utilities
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add ResNet variant | `backbones/resnet.py` | BasicBlock, Bottleneck classes |
| Add transformer | `backbones/mix_transformer.py` | SegFormer-style |
| Add DDRNet | `backbones/ddrnet.py` | Real-time segmentation |
| Add PIDNet | `backbones/pidnet.py` | PIDNet backbone |
| Add custom head | `decode_heads/decode_head.py` | Base class |
| Add DA head | `decode_heads/dapcn_head_mixin.py` | DA mixin pattern |
| Add loss | `losses/cross_entropy_loss.py` | Base loss pattern |
| Add DA loss | `losses/dapg_loss.py` | Dynamic anchor prototype |
| Builder functions | `builder.py` | `build_backbone()`, `build_head()`, etc. |

## KEY COMPONENTS

### Backbones (9+)
| Backbone | File | Use Case | Notes |
|----------|------|----------|-------|
| ResNet | `resnet.py` (713 lines) | General purpose | BasicBlock, Bottleneck, DCN/plugin support |
| ResNeXt | `resnext.py` | Grouped convolutions | |
| ResNeSt | `resnest.py` | Split-attention | |
| DDRNet | `ddrnet.py` (597 lines) | Real-time | Dual-branch, BilateralFusion, DAPPM |
| PIDNet | `pidnet.py` (645 lines) | Boundary detection | Three-branch (P/I/D), PagFM fusion |
| MixTransformer | `mix_transformer.py` (579 lines) | Vision transformer | SegFormer-style, mit_b0–b5 |
| UNetFormer | `timm_backbone.py` | UNet + transformer | Refactored from custom backbone to TIMMBackbone wrapper |
| Segmenter | `segmenter_backbone.py` | ViT-based | |
| KNet | `knet_backbone.py` | Kernel-based | |

### Decode Heads (8+)
| Head | File | Notes |
|------|------|-------|
| Base | `decode_head.py` | Abstract base class |
| DDRNet | `ddrnet_head.py` | With DA integration |
| PIDNet | `pidnet_head.py` | Boundary head |
| SegFormer | `segformer_head.py` | MLP decoder |
| SegFormer+DA | `segformer_dapcn_head.py` | DA-enhanced via DAPCNHeadMixin |
| KNet | `knet_head.py` | Kernel head |
| Segmenter | `segmenter_head.py` | Mask transformer |
| UNetFormer | `unetformer_head.py` | GLA decoder (paper-faithful rewrite) |

### Losses (6)
| Loss | File | Purpose |
|------|------|---------|
| CrossEntropy | `cross_entropy_loss.py` | Standard CE |
| DAPG | `dapg_loss.py` | Dynamic anchor prototype grouping (`@LOSSES.register_module(force=True)`) |
| AffinityBoundary | `affinity_boundary_loss.py` | Affinity-based boundary detection |
| Accuracy | `accuracy.py` | Pixel accuracy metric |
| Utils | `utils.py` | Weighted loss wrapper |

## CONVENTIONS

**Module Registration**:
```python
from mmseg.models.builder import BACKBONES

@BACKBONES.register_module()
class MyBackbone(BaseModule):
    def __init__(self, ...):
        super().__init__(init_cfg)
```

**Init Config Pattern** (preferred over `pretrained`):
```python
init_cfg=dict(
    type='Pretrained',
    checkpoint='path/to/checkpoint.pth'
)
```

**Forward Method Signature**:
```python
def forward(self, x):
    # x: (B, C, H, W)
    # Returns list of feature maps for multi-scale
    return [feat1, feat2, feat3, feat4]
```

## ANTI-PATTERNS
- **NEVER** use `pretrained` param directly (deprecated)
- **NEVER** set both `init_cfg` and `pretrained` (assertion error)
- **ALWAYS** call `super().__init__(init_cfg)` in backbone `__init__`
- **NEVER** return single tensor from backbone; always return list

## UNIQUE STYLES

**Dynamic Anchor Module** (`utils/dynamic_anchor.py`) — see `utils/AGENTS.md` for full details:
- `DynamicAnchorModule`: EM-refined prototypes with quality gating, `@MODELS.register_module(force=True)`
- Configurable `da_position`: 'before_fusion' or 'after_fusion'

**Prototype Memory** (`utils/prototype_memory.py`) — see `utils/AGENTS.md`:
- `PrototypeMemory`: Class-conditioned prototype bank with EMA updates and FPS initialization
- `prototype_contrastive_loss`: InfoNCE-style contrastive loss function

**DAPCN Utils** (`utils/dapcn_utils.py`):
- `extract_boundary_map`: Boundary extraction (sobel, diff, laplacian modes)
- `compute_boundary_gt`: Boundary ground-truth from segmentation labels

**DAPCN Head Mixin** (`decode_heads/dapcn_head_mixin.py`):
- Glue mixin that wires boundary, DAPG, and contrastive losses into any decode head
- Not a registered module — used via Python mixin inheritance

**TIMMBackbone Wrapper** (`backbones/timm_backbone.py`):
- Wraps any `timm` model as an mmseg backbone with `@BACKBONES.register_module()`
- Used by UNetFormer configs (replaced custom backbone in recent refactor)

## BUILDER FUNCTIONS
```python
from mmseg.models import build_backbone, build_head, build_loss

backbone = build_backbone(cfg.backbone)
head = build_head(cfg.decode_head)
loss = build_loss(cfg.loss_decode)
```

## NOTES
- **178 symbols** across 66 model files
- **7506 total lines** in backbones + heads + losses
- All backbones support `frozen_stages` for fine-tuning
- Most support `norm_eval=True` for BN freeze
- Gradient checkpointing available in transformer backbones
