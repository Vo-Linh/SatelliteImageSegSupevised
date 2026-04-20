# DAPCN Experiment Design

## Overview

DAPCN (Dynamic Anchor Prototype Clustering Network) is a plug-and-play module that augments any semantic segmentation decode head with two auxiliary loss components:

1. **DynamicAnchorModule + DAPGLoss** — Dataset-level learnable prototypes with differentiable EM refinement
2. **Boundary Loss** — Sobel/Laplacian/diff-based boundary detection with BCE or affinity loss

**Total Loss:**
```
L = L_ce + boundary_lambda * L_boundary + proto_lambda * L_dapg
```

**Default hyperparameters:**
- boundary_lambda = 0.15, proto_lambda = 0.1
- DynamicAnchor: K=64, tau=0.1, EM iters=3, init=xavier, quality_gate=True
- DAPGLoss: margin=0.3, lambda_inter=0.5, lambda_quality=0.1

---

## Dataset

**OpenEarthMap (OEM):** Satellite imagery semantic segmentation
- 9 classes
- Training splits: 500, 1000, 1500 samples
- Validation: 2000 samples
- Image format: .tif, crop size 512x512
- Training: 40K iterations, AdamW (lr=6e-5), poly schedule with warmup

---

## Exp 1: Main Results — DAPCN as Plug-and-Play Module

**Goal:** Demonstrate that DAPCN consistently improves diverse segmentation architectures on satellite imagery.

**Setup:** Each model trained on OEM train1000 split, evaluated on val2000.

| Model | Backbone | Type | Config |
|-------|----------|------|--------|
| UNetFormer | ResNet18 | CNN+Transformer | `unetformer/unetformer_openearthmap_train1000_40k.py` |
| UNetFormer | ResNeXt101-32x16d | CNN+Transformer | `unetformer/unetformer_openearthmap_train1000_40k_resnext101_32x16d.py` |
| DDRNet | DDRNet-23 | Real-time CNN | `ddrnet/ddrnet_openearthmap_train1000_40k.py` |
| SegFormer | MiT-B5 | Transformer | `segformer/segformer_mit-b5_openearthmap_train1000_40k.py` |
| PIDNet | PIDNet-S | Real-time CNN | `pidnet/pidnet_openearthmap_train1000_40k.py` |
| KNet | ResNet-50 | Kernel-based | `knet/knet_openearthmap_train1000_40k.py` |
| SegMenter | ViT-S | Transformer | `segmenter/segmenter_openearthmap_train1000_40k.py` |

**Results Table:**

| Model | Backbone | Baseline mIoU | +DAPCN mIoU | Delta |
|-------|----------|:---:|:---:|:---:|
| UNetFormer | ResNet18 | | | |
| UNetFormer | ResNeXt101 | | | |
| DDRNet | DDRNet-23 | | | |
| SegFormer | MiT-B5 | | | |
| PIDNet | PIDNet-S | | | |
| KNet | ResNet-50 | | | |
| SegMenter | ViT-S | | | |

**Metrics:** mIoU, mF1, per-class IoU

**Expected Outcomes:**
- DAPCN should deliver consistent mIoU improvements (+1.0–3.0%) across all 7 architectures, validating the plug-and-play claim.
- Larger gains expected on lightweight/real-time models (DDRNet, PIDNet) where the baseline decoder has limited representational capacity — DAPCN's prototype-based regularization compensates for this.
- Transformer-based models (SegFormer, SegMenter) may show smaller but still positive gains, since self-attention already captures global context; DAPCN's boundary loss should still help refine edges.
- UNetFormer-ResNeXt101 (strongest backbone) should have the highest absolute mIoU, but the smallest relative DAPCN improvement — stronger features leave less room for auxiliary supervision.
- Per-class IoU improvements should be most pronounced on classes with complex boundaries (building, road) and under-represented classes (bareland, water) where prototype clustering provides better feature structure.

---

## Exp 2: DA Position Ablation — Before vs After Fusion

**Goal:** Compare where DAPCN operates in the feature pipeline.

- `before_fusion`: DA receives `inputs[-1]` (raw backbone feature, C=in_channels[-1])
- `after_fusion`: DA receives fused decoder feature (C=channels)

**Setup:** OEM train1000, 3 representative architectures.

**Configs needed (OEM variants to create by toggling `da_position` in each model's train1000 config):**

| Model | before_fusion | after_fusion |
|-------|:---:|:---:|
| UNetFormer-R18 | `unetformer_openearthmap_train1000_40k.py` (da_position=before) | create variant with da_position=after |
| SegFormer-B5 | create variant with da_position=before | `segformer_mit-b5_openearthmap_train1000_40k.py` (da_position=after) |
| DDRNet | `ddrnet_openearthmap_train1000_40k.py` (da_position=before) | create variant with da_position=after |

**Results Table:**

| Model | before_fusion mIoU | after_fusion mIoU |
|-------|:---:|:---:|
| UNetFormer-R18 | | |
| SegFormer-B5 | | |
| DDRNet | | |

**Expected Outcomes:**
- `after_fusion` should generally outperform `before_fusion` for architectures with strong multi-scale decoders (UNetFormer, SegFormer), because the DA module operates on richer, semantically fused features where prototype clustering is more meaningful.
- `before_fusion` may be competitive or even better for single-scale decoders (DDRNet) where the raw backbone feature at the deepest level already carries strong semantic information and the decoder is lightweight.
- The gap between positions should be small (0.3–0.8% mIoU), indicating DAPCN is robust to placement — an important practical property for a plug-and-play module.
- Computational cost differs: `before_fusion` processes higher-dimensional features (e.g., 512 or 2048 channels), while `after_fusion` operates on the decoder's reduced channels (e.g., 64 or 256), making `after_fusion` cheaper for large backbones.

---

## Exp 3: Data Efficiency — Performance vs Training Set Size

**Goal:** Show DAPCN is especially beneficial in low-data satellite regimes.

**Setup:** UNetFormer-R18 on OEM with varying training splits.

| Train Samples | Baseline mIoU | +DAPCN mIoU | Relative Delta% |
|:---:|:---:|:---:|:---:|
| 500 | | | |
| 1000 | | | |
| 1500 | | | |

**Existing configs (all +DAPCN):**
- `unetformer/unetformer_openearthmap_train500_40k.py`
- `unetformer/unetformer_openearthmap_train1000_40k.py`
- `unetformer/unetformer_openearthmap_train1500_40k.py`

**Extended across all models:**

| Model | train500 | train1000 | train1500 |
|-------|----------|-----------|-----------|
| UNetFormer R18 | existing | existing | existing |
| DDRNet | existing | existing | existing |
| SegFormer MiT-B5 | existing | existing | existing |
| PIDNet | new | new | new |
| KNet (R50) | new | new | new |
| SegMenter (ViT) | new | new | new |

**Expected Outcomes:**
- DAPCN's relative improvement (Delta%) should **increase as training data decreases** — this is the key hypothesis. With fewer samples, the model is more prone to overfitting and poor class boundary delineation; DAPCN's prototype-based regularization and boundary loss act as strong inductive biases that compensate.
- At train500, we expect +2.0–4.0% mIoU improvement; at train1500, the gap narrows to +0.8–2.0% as the baseline has enough data to learn reasonable features on its own.
- This "data efficiency" story is particularly compelling for satellite imagery, where labeled data is expensive and geographically limited. It positions DAPCN as especially valuable for real-world remote sensing applications with scarce annotations.
- The trend should hold across all model families, but lightweight models (DDRNet, PIDNet) should show the steepest gains at low data since they have less capacity to learn from limited samples alone.

---

## Exp 4: Loss Component Ablation

**Goal:** Isolate the contribution of each DAPCN loss component.

**Setup:** UNetFormer-ResNeXt101 on OEM train1000. Disable components by setting lambda=0. Existing configs: `configs/unetformer/ablation/unetformer_resnext101_oem_{boundary_only,dapg_only,boundary_dapg}.py`.

| CE | +L_boundary | +L_dapg | mIoU | Config |
|:---:|:---:|:---:|:---:|:---|
| Y | | | | Baseline (no DAPCN) |
| Y | Y | | | `unetformer_resnext101_oem_boundary_only.py` (boundary_lambda=0.15, proto_lambda=0) |
| Y | | Y | | `unetformer_resnext101_oem_dapg_only.py` (boundary_lambda=0, proto_lambda=0.1) |
| Y | Y | Y | | `unetformer_resnext101_oem_boundary_dapg.py` (boundary_lambda=0.15, proto_lambda=0.1) — Full DAPCN |

**Expected Outcomes:**
- **L_boundary alone** should provide a moderate boost (+0.5–1.5% mIoU), mainly on boundary-sensitive classes (building edges, thin roads). Boundary supervision is a well-understood technique; the gain confirms it helps but is not the full story.
- **L_dapg alone** should be the **single strongest component** (+1.0–2.0% mIoU). The DynamicAnchorModule's learnable prototypes with EM refinement provide a structural prior on the feature space — encouraging compact, well-separated clusters even without explicit class labels at the prototype level.
- **L_boundary + L_dapg (Full DAPCN)** should achieve the best result, demonstrating that the two losses are complementary: boundary loss sharpens spatial boundaries while DAPG structures the feature space globally.
- This ablation validates that DAPCN's improvement is not driven by a single trick but by the complementary interplay of geometric (boundary) and structural (prototype) objectives.

---

## Exp 5: DynamicAnchorModule Hyperparameter Sensitivity

**Goal:** Study sensitivity to key DA hyperparameters.

**Setup:** UNetFormer-ResNeXt101 on OEM train1000. Vary one parameter at a time, others at default.

### 5a. Number of Prototypes (max_groups K)

| K | mIoU |
|:---:|:---:|
| 16 | |
| 32 | |
| **64** (default) | |
| 128 | |

### 5b. EM Iterations

| num_iters | mIoU |
|:---:|:---:|
| 0 (no refinement) | |
| 1 | |
| **3** (default) | |
| 5 | |

### 5c. Temperature

| tau | mIoU |
|:---:|:---:|
| 0.05 | |
| **0.1** (default) | |
| 0.5 | |
| 1.0 | |

### 5d. Initialization Method

| init_method | mIoU |
|:---:|:---:|
| **xavier** (default) | |
| kaiming | |
| normal | |

**Expected Outcomes:**

**5a. Number of Prototypes (K):**
- Performance should follow an inverted-U curve. K=16 is too few to capture the diversity of satellite imagery sub-patterns (shadow, texture, boundary types). K=128 may dilute each prototype's specialization and slow convergence.
- K=64 (default) should be near-optimal or tied with K=32. The quality gate dynamically prunes low-quality prototypes, so the effective K adapts — making the method robust to over-estimation.
- Key insight: even K=16 should improve over baseline, showing the mechanism is beneficial regardless of exact K.

**5b. EM Iterations:**
- num_iters=0 (no refinement, using persistent prototypes directly) should still improve over baseline, proving the learnable prototypes themselves are valuable — the EM refinement is a bonus, not a requirement.
- Performance should increase from 0→1→3, then plateau or slightly degrade at 5 (diminishing returns + potential overfitting to batch statistics).
- num_iters=3 (default) should be the sweet spot — enough refinement to adapt prototypes to each batch without losing the dataset-level prior.

**5c. Temperature:**
- tau=0.05 makes assignments very sharp (near one-hot) — prototypes specialize aggressively, which can be unstable with limited data.
- tau=0.1 (default) should be optimal — soft enough for gradient flow but sharp enough for meaningful prototype specialization.
- tau=0.5 and tau=1.0 make assignments increasingly uniform, weakening the clustering effect. Expect monotonic performance decline for tau > 0.1.
- This parameter has the highest sensitivity — worth careful tuning for new datasets.

**5d. Initialization Method:**
- Xavier (default) should perform best for normalized-feature regimes since it respects the feature magnitude distribution.
- Kaiming may be slightly worse due to its ReLU-oriented scaling assumption that doesn't match the cosine-similarity-based EM.
- Normal should be competitive since 1/sqrt(D) scaling is reasonable.
- The gap should be small (<0.5% mIoU) because the EM refinement corrects poor initialization within a few training iterations — demonstrating the robustness of the persistent prototype design.

---

## Exp 6: Boundary Loss Mode Ablation

**Goal:** Compare boundary extraction operators and loss formulations.

**Setup:** UNetFormer-ResNeXt101 on OEM train1000.

| boundary_mode | boundary_loss_mode | mIoU | Boundary F1 |
|:---:|:---:|:---:|:---:|
| **sobel** (default) | **binary** (default) | | |
| laplacian | binary | | |
| diff | binary | | |
| sobel | affinity | | |
| sobel | hybrid | | |

**Expected Outcomes:**
- **Sobel + binary** (default) should be strongest or near-best. Sobel is a well-established edge detector that captures both horizontal and vertical gradients; binary BCE directly supervises boundary prediction — simple and effective.
- **Laplacian + binary** should be slightly worse. Laplacian is a second-order operator that captures edges isotropically but is more sensitive to noise — a problem in satellite imagery with varying illumination and sensor noise.
- **Diff + binary** is the simplest (first-order neighbor difference). It should be competitive with Sobel on clean satellite images but less robust to noise.
- **Sobel + affinity** uses relational pixel-pair supervision instead of per-pixel BCE. It should improve Boundary F1 specifically because it models whether adjacent pixels belong to the same class — more geometrically meaningful than binary boundary prediction. However, mIoU gain may be similar to binary mode.
- **Sobel + hybrid** (weighted combination) should achieve the best Boundary F1 by combining the strengths of both: binary for coarse boundary detection, affinity for fine-grained boundary refinement. The mIoU improvement over binary-only may be marginal (+0.1–0.3%) since boundary quality beyond a threshold doesn't strongly affect region-level IoU.
- Overall, the boundary mode choice has moderate impact. Sobel is the safest default; affinity and hybrid modes are worth exploring if boundary quality is a key evaluation criterion.

---

## Exp 7: DA Position — Full Model Coverage on OpenEarthMap

**Goal:** Extend the DA-position study (Exp 2) across all 6 model families on the main OEM dataset, to establish whether `before_fusion` vs `after_fusion` interacts with architecture choice at scale.

**Setup:** OEM train1000, 40K iterations. For each model, create/run `before_fusion` and `after_fusion` variants of the existing `*_openearthmap_train1000_40k.py` config by setting `da_position` accordingly.

| Model | Baseline mIoU | +DAPCN (before) mIoU | +DAPCN (after) mIoU |
|-------|:---:|:---:|:---:|
| UNetFormer-R18 | | | |
| SegFormer-B5 | | | |
| DDRNet | | | |
| PIDNet | | | |
| KNet (R50) | | | |
| SegMenter (ViT) | | | |

**Expected Outcomes:**
- DAPCN should deliver positive gains at both positions across all 6 models, confirming the mechanism is architecture-agnostic within the OEM domain.
- `after_fusion` should generally win for architectures with strong multi-scale decoders (UNetFormer, SegFormer), where fused features carry richer semantics for prototype clustering.
- `before_fusion` may be competitive or preferable for single-scale / lightweight decoders (DDRNet, PIDNet), where the raw backbone feature already carries strong semantic information.
- Gaps between positions should be small (0.3–0.8% mIoU), demonstrating that DAPCN is robust to placement — an important practical property for a plug-and-play module.
- Per-class IoU gains should concentrate on boundary-heavy classes (building, road) and under-represented classes (bareland, water), regardless of position choice.

---

## Exp 8: Computational Overhead Analysis

**Goal:** Quantify the parameter/FLOPs/inference overhead of DAPCN.

**Key insight:** DAPCN auxiliary losses (boundary, DAPG) are **training-only**. At inference, only the DynamicAnchorModule's learnable prototypes add parameters (K * C floats), but they are not used in the forward pass.

| Model | Params (M) | +DAPCN Params (M) | Overhead% | FLOPs (G) | +DAPCN FLOPs (G) | Inference ms |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| UNetFormer-R18 | | | | | | |
| DDRNet | | | | | | |
| SegFormer-B5 | | | | | | |
| PIDNet | | | | | | |
| KNet (R50) | | | | | | |
| SegMenter (ViT) | | | | | | |

**Expected Outcomes:**
- **Parameter overhead** should be minimal (<1% for most models). The DynamicAnchorModule adds K*C parameters (e.g., 64*256 = 16K floats = 0.06MB) plus a small quality_net MLP. The PrototypeMemory is a buffer (no gradient), not counted as learnable params.
- **FLOPs overhead at inference: zero.** DAPCN's auxiliary losses (boundary, DAPG) are only computed during training. At inference, the model uses the standard forward path — `decode_head.forward()` does not invoke the DynamicAnchorModule. This is the strongest efficiency argument.
- **Training FLOPs overhead** is moderate (~5–15%), primarily from the EM refinement (3 iterations of matrix multiply over N*K) and the boundary Sobel convolution. Larger overheads on SegMenter/SegFormer due to higher feature dimensions.
- **Inference latency** should be identical between baseline and +DAPCN, since the DA module and boundary loss are not part of the inference graph. Any measured difference should be within noise margin (<1ms).
- This is a key selling point: DAPCN provides "free" improvements at inference time — the cost is only during training. For deployment-sensitive applications (real-time satellite monitoring, edge devices), this is highly attractive.

---

## Qualitative Experiments

### Qual 1: Segmentation Visualization
- Side-by-side: Image | GT | Baseline | +DAPCN
- Focus on: boundary sharpness, small objects (thin roads), confusing classes (vegetation vs barren)
- Select from OEM val2000

**Expected Outcomes:**
- +DAPCN predictions should show visibly **sharper, more accurate boundaries** around buildings, roads, and water bodies compared to baseline. The boundary loss directly supervises edge quality.
- Baseline predictions often show "bleeding" between adjacent classes (e.g., tree/agriculture, building/road); +DAPCN should reduce this due to the inter-group separation enforced by L_dapg.
- Small, isolated objects (e.g., narrow roads, small water ponds) should be better preserved by +DAPCN — the prototype clustering provides structural priors that prevent these regions from being absorbed by dominant surrounding classes.

### Qual 2: Boundary Quality Visualization
- GT boundary | Baseline predicted boundary | +DAPCN predicted boundary
- Overlay on original satellite image
- Compute boundary IoU as quantitative support
- **Tool:** `tools/visualize_boundary.py`

**Expected Outcomes:**
- +DAPCN boundary maps should be **thinner and more precise** than baseline — less noisy, fewer false-positive boundary pixels in homogeneous regions (open fields, water).
- Baseline boundaries tend to be thick and diffuse because the model is only trained with per-pixel CE loss; +DAPCN's explicit boundary supervision produces crisp, single-pixel-wide boundaries closer to GT.
- Boundary overlay on the image should show +DAPCN contours tightly hugging building edges and road borders, while baseline contours may be offset or fragmented.
- Quantitative boundary IoU improvement of +3–8% expected, with boundary F1 improvement of +5–10%.

### Qual 3: Prototype Evolution (t-SNE/UMAP)
- Visualize DA prototypes at training stages: iter 0, 10K, 20K, 40K
- Color-code by quality score (show quality gating effect)
- Show how class-agnostic prototypes relate to semantic classes
- **Tool:** `tools/visualize_prototypes.py`

**Expected Outcomes:**
- **Early training (iter 4K):** Prototypes should be randomly scattered with low, uniform quality scores. The EM refinement is just beginning to discover structure in the feature space.
- **Mid training (iter 16K–20K):** Prototypes should form visible clusters in the t-SNE embedding. Quality scores should diverge — some prototypes specialize (high quality), others remain generic (low quality, candidates for gating).
- **Late training (iter 40K):** Prototypes should form tight, well-separated clusters. The quality histogram should be bimodal: a group of high-quality (>0.5) specialized prototypes and a tail of low-quality (<0.1) prototypes that are effectively pruned by the quality gate.
- The inter-prototype similarity matrix should show increasing block-diagonal structure over training — groups of prototypes that are similar to each other (capturing the same semantic concept) but dissimilar to other groups.
- Even though prototypes are class-agnostic, they should implicitly correspond to semantic sub-categories: "building boundary," "road texture," "vegetation interior," etc. The class correlation analysis in Qual 5 will confirm this.

### Qual 4: Feature Space Analysis (t-SNE)
- Pixel embeddings: baseline vs +DAPCN
- Demonstrate more compact intra-class clusters and larger inter-class separation
- **Tool:** `tools/visualize_feature_space.py`

**Expected Outcomes:**
- **Baseline features:** t-SNE should show overlapping, diffuse class clusters with unclear boundaries. Especially classes like rangeland/agriculture and bareland/developed space should heavily overlap since CE loss alone doesn't enforce feature-space structure.
- **+DAPCN features:** Clusters should be visibly **more compact and better separated**. The L_dapg loss (intra-group compactness + inter-group separation) directly shapes the feature space, and this should be clearly visible.
- The separation ratio metric (inter-distance / intra-distance) should improve by 20–50% with DAPCN. This is the single most important quantitative evidence for DAPCN's effect on feature quality.
- Rare classes (water, bareland) that are typically scattered in baseline feature space should form coherent clusters with DAPCN, explaining the per-class IoU improvements seen in Exp 1.

### Qual 5: Assignment Map Visualization
- Soft assignment maps (N, K') — which pixels attend to which prototypes
- Show different prototypes capture different semantic structures
- **Tool:** `tools/visualize_assignments.py`

**Expected Outcomes:**
- Different prototypes should activate on **semantically meaningful** regions: some prototypes activate strongly on building interiors, others on road edges, others on vegetation texture, etc. This demonstrates that the class-agnostic EM discovers semantically coherent sub-categories.
- Boundary-adjacent pixels should be assigned to specialized "boundary prototypes" that are distinct from "interior prototypes" — showing the DA module learns a boundary/interior distinction without explicit supervision (complementary to L_boundary).
- The prototype-to-class correlation heatmap should show that each prototype predominantly covers 1–2 classes, not spread uniformly — confirming meaningful specialization despite being class-agnostic.
- Hard assignment maps should produce a "super-segmentation" that is finer than the semantic segmentation: multiple prototypes per class capture sub-patterns (e.g., shadow vs sunlit building, sparse vs dense vegetation). This provides interpretability into what the model has learned.
- The quality gate should have pruned 20–40% of prototypes (those with quality < 0.1), and the surviving prototypes should show clear semantic specialization.

### Qual 6: Failure Case Analysis
- Cases where DAPCN does not help or hurts
- Analyze root causes: very small objects, heavy occlusion, ambiguous classes

**Expected Outcomes:**
- DAPCN may **not help** on images dominated by a single class (e.g., large open farmland) where there are few boundaries and little structural diversity for prototypes to discover.
- DAPCN may **slightly hurt** on very small objects (< 5x5 pixels) where the boundary loss's Sobel operator cannot resolve the object, and prototype assignment noise at small scales introduces confusion.
- Classes with inherent label ambiguity (e.g., "developed space" overlapping with "road" or "bareland") may not benefit much — DAPCN can structure features but cannot resolve fundamental label noise.
- These failure cases should be rare (<5% of images) and the degradation mild (<0.5% per-image IoU), confirming DAPCN's robustness as a general-purpose plug-in.

---

## Experiment Priority

1. **Exp 1** + **Qual 1** — Core plug-and-play claim
2. **Exp 4** — Loss component contribution
3. **Exp 3** — Data efficiency (strong for satellite domain)
4. **Exp 2** — DA position architectural insight
5. **Exp 5** — Hyperparameter robustness
6. **Exp 8** — Overhead justification
7. **Qual 3 + Qual 4** — Interpretability and visualization
8. **Exp 6** — Boundary mode details
9. **Exp 7** — Cross-dataset generalization

---

## Config File Index

### OpenEarthMap +DAPCN Configs

| Model | train500 | train1000 | train1500 |
|-------|----------|-----------|-----------|
| UNetFormer R18 | `unetformer_openearthmap_train500_40k.py` | `unetformer_openearthmap_train1000_40k.py` | `unetformer_openearthmap_train1500_40k.py` |
| UNetFormer ResNeXt101 | — | `unetformer_openearthmap_train1000_40k_resnext101_32x16d.py` | — |
| DDRNet | `ddrnet_openearthmap_train500_40k.py` | `ddrnet_openearthmap_train1000_40k.py` | `ddrnet_openearthmap_train1500_40k.py` |
| SegFormer MiT-B5 | `segformer_mit-b5_openearthmap_train500_40k.py` | `segformer_mit-b5_openearthmap_train1000_40k.py` | `segformer_mit-b5_openearthmap_train1500_40k.py` |
| PIDNet | `pidnet_openearthmap_train500_40k.py` | `pidnet_openearthmap_train1000_40k.py` | `pidnet_openearthmap_train1500_40k.py` |
| KNet (R50) | `knet_openearthmap_train500_40k.py` | `knet_openearthmap_train1000_40k.py` | `knet_openearthmap_train1500_40k.py` |
| SegMenter (ViT) | `segmenter_openearthmap_train500_40k.py` | `segmenter_openearthmap_train1000_40k.py` | `segmenter_openearthmap_train1500_40k.py` |

### OpenEarthMap Ablation Configs (UNetFormer-ResNeXt101)

| Ablation | Config |
|----------|--------|
| Boundary only | `configs/unetformer/ablation/unetformer_resnext101_oem_boundary_only.py` |
| DAPG only | `configs/unetformer/ablation/unetformer_resnext101_oem_dapg_only.py` |
| Boundary + DAPG (Full) | `configs/unetformer/ablation/unetformer_resnext101_oem_boundary_dapg.py` |

### DA Position Variants (Exp 7)

To be created per model by duplicating the train1000 config and setting `da_position='before_fusion'` or `'after_fusion'`. No separate config files ship with the repo today.
