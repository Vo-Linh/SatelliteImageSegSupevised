# UNetFormer DAPCN Architecture Workflow

**Config**: `configs/unetformer/unetformer_dapcn_after_fusion_cityscapes.py`

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           UNetFormer with DAPCN                                 │
│                    Dynamic Anchor Prototype-Guided Network                      │
└─────────────────────────────────────────────────────────────────────────────────┘

    Input Image (3×512×512)
            │
            ▼
┌───────────────────────────────┐
│     UNetFormer Backbone       │
│   (Hybrid CNN-Transformer)    │
│                               │
│   4-Stage Hierarchical        │
│   Feature Extraction          │
└───────────────────────────────┘
            │
            ▼
    [64, 128, 256, 512] channels
     at scales H/4, H/8, H/16, H/32
            │
            ▼
┌───────────────────────────────┐
│    Feature Pyramid Decoder    │
│                               │
│   Multi-scale fusion with     │
│   U-Net style skip connections│
└───────────────────────────────┘
            │
            ▼
    Fused Feature (256, H/8, W/8)
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌──────────┐  ┌──────────────┐
│   DAPCN  │  │ Segmentation │
│  Module  │  │    Head      │
│(after    │  │              │
│ fusion)  │  │  Conv +      │
└──────────┘  │ Upsample     │
    │         └──────────────┘
    │                 │
    │                 ▼
    │         Segmentation Logits
    │         (19 classes, H×W)
    │                 │
    │                 ▼
    │    ┌────────────────────────┐
    │    │     Multi-Task Loss    │
    │    │                        │
    │    │  L = Lce + 0.3×Lbound  │
    │    │      + 0.1×Ldapg       │
    │    │      + 0.1×Lcontrast   │
    │    └────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   Dynamic Anchor Module      │
│                              │
│  - 64 prototype groups       │
│  - EM refinement (3 iters)   │
│  - Quality gating (≥0.1)     │
│  - Persistent nn.Parameter   │
└──────────────────────────────┘
```

---

## Detailed Data Flow

### 1. Input Processing

```
Input: (B, 3, 512, 512)
         │
         ▼
┌──────────────────────┐
│ Patch Embedding      │
│ Conv4x4 + LayerNorm  │
└──────────────────────┘
         │
         ▼
(B, H/4 × W/4, 64)
```

### 2. UNetFormer Backbone (Encoder Path)

```
Stage 0: 2 Transformer Blocks
├── Window Attention (local, window_size=7)
├── Global Token Attention (global context)
└── Output: (B, H/4 × W/4, 64)
         │
         ▼
Stage 1: 2 Blocks + PatchMerging (2x downsample)
└── Output: (B, H/8 × W/8, 128)
         │
         ▼
Stage 2: 6 Blocks + PatchMerging (2x downsample)
└── Output: (B, H/16 × W/16, 256)
         │
         ▼
Stage 3: 2 Blocks + PatchMerging (2x downsample)
└── Output: (B, H/32 × W/32, 512)
         │
         ▼
Decoder Path (U-Net style with skip connections)
```

### 3. Feature Pyramid Decoder

```
Multi-scale features from backbone:
┌─────────────────────────────────────────────┐
│  Scale 0: (B, 64,  H/8,  W/8)  →  Proj to 256  │
│  Scale 1: (B, 128, H/16, W/16) →  Proj to 256  │
│  Scale 2: (B, 256, H/32, W/32) →  Proj to 256  │
│  Scale 3: (B, 512, H/32, W/32) →  Proj to 256  │
└─────────────────────────────────────────────┘
         │
         ▼
Progressive Fusion (bottom-up):
┌─────────────────────────────────────────────┐
│  1. Start with deepest feature (Scale 3)    │
│  2. Upsample ×2                             │
│  3. Concat with next scale                  │
│  4. Apply decoder conv                      │
│  5. Repeat for all scales                   │
└─────────────────────────────────────────────┘
         │
         ▼
Final Conv → (B, 256, H/8, W/8)
```

### 4. DAPCN Module (After Fusion Position)

```
Fused Feature (B, 256, H/8, W/8)
         │
         ▼
┌──────────────────────────────────────────┐
│     DynamicAnchorModule                  │
│                                          │
│  Parameters:                             │
│  - max_groups: 64                        │
│  - temperature: 0.1                      │
│  - num_iters: 3 (EM iterations)          │
│  - min_quality: 0.1                      │
│  - use_quality_gate: True                │
└──────────────────────────────────────────┘
         │
         ▼
EM Refinement Loop (3 iterations):
┌──────────────────────────────────────────┐
│                                          │
│  E-step:                                 │
│    sim = feats @ proto.T / temperature   │
│    assign = softmax(sim)                 │
│                                          │
│  M-step:                                 │
│    proto = assign.T @ feats / sizes      │
│    proto = normalize(proto)              │
│                                          │
└──────────────────────────────────────────┘
         │
         ▼
Quality Gating:
┌──────────────────────────────────────────┐
│  - Quality network scores each prototype │
│  - Filter: quality ≥ 0.1                 │
│  - Ensure at least 1 prototype survives  │
└──────────────────────────────────────────┘
         │
         ▼
Outputs:
  - assignments: (N, K')  [N = B×H/8×W/8, K' ≤ 64]
  - prototypes:  (K', 256)
  - quality:     (K',)
```

### 5. Segmentation Head

```
Fused Feature (B, 256, H/8, W/8)
         │
         ▼
┌──────────────────────────┐
│ Classification Layer     │
│ Conv + Upsample ×8       │
└──────────────────────────┘
         │
         ▼
Segmentation Logits (B, 19, H, W)
         │
         ▼
Softmax → Class Probabilities
```

---

## Loss Computation

### Total Loss Function

```
L_total = L_ce + λ_boundary × L_boundary + λ_proto × L_dapg + λ_contrastive × L_contrastive

where:
  L_ce:           CrossEntropy (λ = 1.0)
  L_boundary:     Boundary detection (λ = 0.3)
  L_dapg:         Dynamic Anchor Prototype-Guided (λ = 0.1)
  L_contrastive:  InfoNCE contrastive (λ = 0.1)
```

### Individual Losses

```
1. CrossEntropy Loss (L_ce)
   ┌─────────────────────────────────────┐
   │ Standard pixel-wise classification  │
   │ L_ce = -Σ y_true × log(y_pred)      │
   └─────────────────────────────────────┘

2. Boundary Loss (L_boundary)
   ┌─────────────────────────────────────┐
   │ Sobel edge detection on predictions │
   │ Binary cross-entropy on edges       │
   │ mode = 'sobel'                      │
   └─────────────────────────────────────┘

3. DAPG Loss (L_dapg)
   ┌─────────────────────────────────────┐
   │ L_intra:  Intra-class compactness   │
   │           (1/N) Σ A_ik × (1 - cos(F_i, Q_k)) │
   │                                     │
   │ L_inter:  Inter-class separation    │
   │           max(0, cos(Q_g, Q_h) - margin) │
   │                                     │
   │ L_quality: Quality regularization   │
   │           -mean(log(quality))       │
   │                                     │
   │ L_dapg = L_intra + 0.5×L_inter + 0.1×L_quality │
   └─────────────────────────────────────┘

4. Contrastive Loss (L_contrastive)
   ┌─────────────────────────────────────┐
   │ InfoNCE on prototype memory bank    │
   │ temperature = 0.07                  │
   │ Enabled after warmup (500 iters)    │
   └─────────────────────────────────────┘
```

---

## Key Configuration Parameters

```python
# From unetformer_dapcn_after_fusion_cityscapes.py

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='UNetFormer',
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        window_size=7,
    ),
    decode_head=dict(
        type='UNetFormerDAPCNHead',
        channels=256,
        num_classes=19,
        
        # DAPCN Configuration
        da_position='after_fusion',  # Apply DAPCN after feature fusion
        boundary_lambda=0.3,
        proto_lambda=0.1,
        contrastive_lambda=0.1,
        
        # Dynamic Anchor Module
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,
            temperature=0.1,
            num_iters=3,
            min_quality=0.1,
            use_quality_gate=True,
        ),
        
        # DAPG Loss
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,
            lambda_inter=0.5,
            lambda_quality=0.1,
        ),
        
        warmup_iters=500,  # Before contrastive loss activates
    ),
)
```

---

## Data Dimension Summary

| Stage | Tensor Shape | Channels | Resolution |
|-------|-------------|----------|------------|
| Input | (B, 3, 512, 512) | 3 | 512×512 |
| Patch Embed | (B, 128×128, 64) | 64 | 128×128 |
| Stage 0 | (B, 128×128, 64) | 64 | 128×128 |
| Stage 1 | (B, 64×64, 128) | 128 | 64×64 |
| Stage 2 | (B, 32×32, 256) | 256 | 32×32 |
| Stage 3 | (B, 16×16, 512) | 512 | 16×16 |
| Fused | (B, 256, 64, 64) | 256 | 64×64 |
| Prototypes | (K', 256) | 256 | K' anchors |
| Output | (B, 19, 512, 512) | 19 | 512×512 |

---

## Training Pipeline

```
For each iteration:
  1. Load batch (images, labels)
  
  2. Forward pass:
     ├─ Backbone → multi-scale features
     ├─ Decoder → fused features
     ├─ DAPCN → prototypes, assignments, quality
     └─ Head → segmentation logits
  
  3. Compute losses:
     ├─ L_ce from logits vs labels
     ├─ L_boundary from edge detection
     ├─ L_dapg from prototypes
     └─ L_contrastive (after warmup)
  
  4. Backward pass:
     └─ Gradients flow to:
        ├─ Network weights
        ├─ Prototype parameters
        └─ Quality network
  
  5. Optimizer step (AdamW)
     └─ Update all parameters
```

---

## File Locations

```
mmseg/
├── models/
│   ├── backbones/
│   │   └── unetformer.py              # UNetFormer backbone
│   ├── decode_heads/
│   │   ├── unetformer_head.py         # UNetFormerDAPCNHead
│   │   └── dapcn_head_mixin.py        # DAPCN mixin
│   ├── losses/
│   │   ├── dapg_loss.py               # DAPG loss
│   │   └── affinity_boundary_loss.py  # Boundary loss
│   └── utils/
│       ├── dynamic_anchor.py          # DynamicAnchorModule
│       └── prototype_memory.py        # Prototype memory bank
│
configs/
└── unetformer/
    └── unetformer_dapcn_after_fusion_cityscapes.py  # This config
```

---

## To Generate Visual Diagrams

Run the Python script (requires graphviz):

```bash
# Install dependencies
pip install graphviz

# Generate diagrams
python draw_workflow.py

# Output files:
#   workflow_diagrams/unetformer_dapcn_detailed.png
#   workflow_diagrams/unetformer_dapcn_simplified.png
#   workflow_diagrams/unetformer_dapcn_training.png
```

Or view the diagrams online by pasting the DOT code from draw_workflow.py into:
- https://dreampuf.github.io/GraphvizOnline/
- https://edotor.net/
