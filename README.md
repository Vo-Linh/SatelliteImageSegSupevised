# DAPCN — Dynamic Anchor Prototype Cross-Attention Network

Satellite image segmentation with prototype learning and quality gating for improved boundary detection on remote sensing imagery. Built on [MMSegmentation](https://github.com/open-mmlab/mmseg) v0.16.0.

## Features

- **Dynamic Anchor Module** — Learnable prototypes with differentiable EM refinement and quality gating
- **Prototype Memory Bank** — Cross-batch class-conditioned prototypes with EMA updates
- **Boundary-Aware Losses** — Sobel/Laplacian/diff-based boundary detection with affinity loss
- **Plug-and-Play** — DAPCNHeadMixin wires DA losses into any decode head via Python mixin
- **Zero Inference Overhead** — All auxiliary losses are training-only; identical inference speed to baseline

## Supported Backbones

| Backbone | Configs |
|----------|---------|
| DDRNet | `configs/ddrnet/` |
| SegFormer (MiT-B5) | `configs/segformer/` |
| UNetFormer (TIMM) | `configs/unetformer/` |

## Dataset

[OpenEarthMap](https://open-earth-map.github.io/) — 7-class remote sensing land cover mapping from overhead satellite imagery.

Train splits: 500, 1000, 1500 samples. Eval: 2000 samples.

## Quick Start

```bash
# Setup
source .venv/bin/activate
pip install -e .

# Train
python tools/train.py configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py

# Multi-GPU
bash tools/dist_train.sh configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py 4

# Evaluate
python tools/test.py <config> <checkpoint> --eval mIoU
```

## Project Structure

```
├── configs/                # Experiment configs
│   ├── _base_/            # Inherited base configs
│   ├── ddrnet/             # DDRNet experiments
│   ├── segformer/         # SegFormer experiments
│   └── unetformer/        # UNetFormer experiments
├── mmseg/                 # Core framework (vendorized MMSeg v0.16.0)
│   ├── models/
│   │   ├── backbones/     # Feature extractors
│   │   ├── decode_heads/   # Segmentation heads + DAPCN mixin
│   │   ├── losses/        # DAPGLoss, AffinityBoundaryLoss, CrossEntropy
│   │   └── utils/         # DynamicAnchorModule, PrototypeMemory, dapcn_utils
│   ├── datasets/          # Dataset classes + pipelines
│   └── core/              # Evaluation hooks, samplers
├── tools/                 # Training/testing scripts
│   ├── train.py
│   ├── test.py
│   └── dist_train.sh
└── docs/
    └── experiment_design.md  # Full experiment plan
```

## Loss Components

DAPCN adds three auxiliary losses to the base cross-entropy:

```
L = L_ce + λ_boundary * L_boundary + λ_proto * L_dapg + λ_contrastive * L_contrastive
```

| Loss | Default Weight | Description |
|------|:---:|-------------|
| Boundary | 0.3 | Sobel/Laplacian boundary detection |
| DAPG | 0.1 | Dynamic anchor prototype grouping |
| Contrastive | 0.1 | InfoNCE against prototype memory bank |

## Extending DAPCN

To add DAPCN to a new decode head:

```python
from mmseg.models.decode_heads.dapcn_head_mixin import DAPCNHeadMixin

class MyDAPCNHead(DAPCNHeadMixin, BaseDecodeHead):
    def __init__(self, da_position='before_fusion', dynamic_anchor=None, **kwargs):
        super().__init__(init_cfg=kwargs.get('init_cfg'))
        self.init_dapcn(da_position=da_position, dynamic_anchor=dynamic_anchor, ...)
```

## Dependencies

- Python 3.10+
- PyTorch >= 1.8
- [mmcv-full](https://github.com/open-mmlab/mmcv) >= 1.3.7, <= 1.7.2
- [timm](https://github.com/rwightman/pytorch-image-models)

## Acknowledgements

- [MMSegmentation](https://github.com/open-mmlab/mmseg) — base segmentation framework
- [OpenEarthMap](https://open-earth-map.github.io/) — remote sensing dataset
