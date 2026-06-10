# PyramidMamba on OpenEarthMap

Integration of **PyramidMamba** (Pyramid Pooling + Mamba SSM decoder) from
[GeoSeg](https://github.com/WangLibo1995/GeoSeg) — *"PyramidMamba: Rethinking
Pyramid Feature Fusion with Selective Space State Model for Semantic Segmentation
of Remote Sensing Imagery"* (arXiv:2406.10828) — into this mmsegmentation codebase,
in the same style as UNetFormer / SegFormer.

The GeoSeg `EfficientPyramidMamba` decoder is ported as an mmseg decode head:

- `mmseg/models/decode_heads/pyramidmamba_modules.py` — `MambaLayer` (PPM + Mamba),
  `ConvFFN`, `Block`, `PyramidMambaDecoder`.
- `mmseg/models/decode_heads/pyramidmamba_head.py` — `PyramidMambaHead` (plain) and
  `PyramidMambaDAPCNHead` (with the repo's DAPCN auxiliary losses via `DAPCNHeadMixin`).

The model runs on the stock `EncoderDecoder` + `TIMMBackbone`. The backbone
(`resnext101_32x16d`, `out_indices=(1,2,3,4)` → channels `[256, 512, 1024, 2048]`)
provides four feature maps; the head selects the **stride-4** (256ch) and
**stride-32** (2048ch) maps via `in_index=[0, 3]`, fuses them with a Mamba-PPM
block, and classifies at full resolution.

## Dependencies (required)

PyramidMamba needs the CUDA-compiled Mamba SSM kernels (original **Mamba v1** API,
`from mamba_ssm import Mamba`). Install matching `cu118 / torch2.0 / cp310` wheels
into the project `.venv`:

```bash
.venv/bin/pip install --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
.venv/bin/pip install "transformers>=4.38,<4.41" ninja
```

**Caveats**

- Pin **`mamba-ssm` 1.2.x** (not 2.x — its API differs) and **`transformers<4.41`**:
  `mamba_ssm/__init__.py` imports `GreedySearchDecoderOnlyOutput` /
  `SampleDecoderOnlyOutput` from `transformers.generation`, which were removed in
  transformers ≥ 4.41.
- `transformers 4.40` requires `huggingface_hub<1.0`, so this **downgrades
  huggingface_hub** (1.8.0 → 0.36.2). `timm` pretrained loading still works.
- `mamba_ssm` is imported **lazily** (only when a PyramidMamba head is instantiated),
  so the rest of `mmseg` still imports without these packages.

Verify: `.venv/bin/python -c "from mamba_ssm import Mamba; print('ok')"`.

## Configs

| Config | Head | Train split | Val split |
|---|---|---|---|
| `pyramidmamba_openearthmap_train500_40k_resnext101_32x16d.py`  | DAPCN | `train_500_fixed`  | `val_2000_fixed` |
| `pyramidmamba_openearthmap_train1000_40k_resnext101_32x16d.py` | DAPCN | `train_1000_fixed` | `val_2000_fixed` |
| `pyramidmamba_openearthmap_train1500_40k_resnext101_32x16d.py` | DAPCN | `train_1500_fixed` | `val_2000_fixed` |
| `pyramidmamba_openearthmap_train3000_val500_40k_resnext101_32x16d.py` | DAPCN | `train_woxbd_3000` | `val_woxbd_500` |
| `baseline/pyramidmamba_baseline_openearthmap_train{500,1000,1500}_40k_resnext101_32x16d.py` | plain | as above | `val_2000_fixed` |
| `baseline/pyramidmamba_baseline_openearthmap_train3000_val500_40k_resnext101_32x16d.py` | plain | `train_woxbd_3000` | `val_woxbd_500` |

`train1500` is the full config; the other sizes inherit from it and only override the
split + `work_dir`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/pyramidmamba/pyramidmamba_openearthmap_train1500_40k_resnext101_32x16d.py --seed 0
```

**Memory:** the Mamba block is memory-heavy — lower `samples_per_gpu` to 4 (or 2) if you hit OOM.

## Backbone variants

Two backbones are provided (GeoSeg's two PyramidMamba variants):

- **ResNeXt101-32x16d** (EfficientPyramidMamba decoder) — `*_resnext101_32x16d.py`,
  `mode='whole'` inference. NCHW features.
- **Swin-B** (`swin_base_patch4_window12_384`) — `*_swinb.py`. The paper's main model.
  Swin emits **NHWC** features, so the head sets `backbone_nhwc=True` to permute them
  to NCHW. timm Swin is **resolution-locked to `img_size`**, so these configs fix
  `img_size=512` and use **slide inference** (`test_cfg=dict(mode='slide',
  crop_size=(512,512), stride=(512,512))`) — matching GeoSeg's fixed-patch /
  sliding-window methodology while keeping the 512 training crop used by every other
  model here. (Alternative: `img_size=1024` + train/eval at 1024, matching the paper's
  exact setup — a config-only change.)

Each variant has the full DAPCN + plain-baseline × {500,1000,1500,woxbd} family.

## Bug fixed: class 0 / "unknown" collapse

An early run scored **IoU.class_0 = 0** — PyramidMamba collapsed the uniform-black
"unknown"/no-data class into class 6 (water), while learning classes 1–8 fine.

Root cause (fixed in `pyramidmamba_modules.py`): the decoder's
`self.apply(self._init_weights)` (as in GeoSeg's original) recursed into the
**Mamba** block and overwrote its `nn.Linear` layers — zeroing `dt_proj.bias`
(properly ≈ −4.6, inverse-softplus). That blows up the SSM timestep
`dt = softplus(...)` (≈0.69 instead of ~0.01), saturating the selective scan and
crippling the **global-context path**. The model then relied on local features,
which can't distinguish solid-black no-data from dark water — so the rare class 0
never won the argmax. `PyramidMambaDecoder._init_decoder_weights()` now skips the
Mamba subtree so its SSM init is preserved.

The configs also use `decode_channels=256`, `lr=6e-5`, `max_iters=60000` (matching
the proven ResNeXt101 UNetFormer recipe). These were not the cause; revert to
`128`/`3e-5`/`40k` if you prefer the lighter/faithful setup.
