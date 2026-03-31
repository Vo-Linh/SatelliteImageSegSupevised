# ---------------------------------------------------------------
# SegFormerDAPCNHead: SegFormer decode head with DAPCN support
# Integrates boundary-aware loss, dynamic anchor prototypes,
# DAPG loss, and persistent prototype memory bank with
# contrastive regularisation into SegFormer head.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin
from .segformer_head import MLP


@HEADS.register_module()
class SegFormerDAPCNHead(DAPCNHeadMixin, BaseDecodeHead):
    """SegFormer decode head augmented with DAPCN auxiliary losses.

    Combines SegFormer's simple and efficient multi-scale feature fusion
    with DAPCN auxiliary loss components:
        1. Boundary loss — Sobel/Laplacian/diff binary BCE, or
           affinity-based relational loss, or hybrid.
        2. DynamicAnchorModule + DAPGLoss — dataset-level learnable
           prototypes with per-batch EM refinement.
        3. PrototypeMemory + InfoNCE — persistent EMA class-conditioned
           prototypes accumulated across iterations.

    Loss budget (total = CE + auxiliary):
        L = L_ce
            + boundary_lambda   * L_boundary
            + proto_lambda      * L_dapg
            + contrastive_lambda * L_contrastive

    SegFormer Args:
        in_channels (int|Sequence[int]): Input channels from backbone.
        channels (int): Intermediate channels in decoder.
        num_classes (int): Number of segmentation classes.
        decoder_params (dict): Parameters for SegFormer decoder:
            - embed_dim (int): Embedding dimension for MLPs.
            - conv_kernel_size (int): Kernel size for fusion convolution.
        in_index (Sequence[int]): Indices of backbone features to use.
        input_transform (str): Must be 'multiple_select' for SegFormer.
        **kwargs: Additional BaseDecodeHead arguments.

    DAPCN Args:
        boundary_lambda (float): Weight for boundary loss. Default: 0.3.
        proto_lambda (float): Weight for DAPG prototype loss. Default: 0.1.
        contrastive_lambda (float): Weight for memory-bank contrastive
            loss. Default: 0.1.
        boundary_mode (str): Gradient operator for boundary extraction
            ('sobel', 'laplacian', 'diff'). Default: 'sobel'.
        boundary_loss_mode (str): 'binary', 'affinity', or 'hybrid'.
            Default: 'binary'.
        hybrid_binary_weight (float): Binary weight in hybrid mode.
            Default: 0.5.
        contrastive_temperature (float): InfoNCE temperature. Default: 0.07.
        contrastive_sample_ratio (float): Fraction of valid pixels to
            sample for contrastive loss. Default: 0.1.
        warmup_iters (int): Iterations before enabling contrastive loss.
            Default: 500.
        num_prototypes_per_class (int): Number of prototypes per class
            in memory bank. Default: 1.
        prototype_ema (float): EMA momentum for prototype updates.
            Default: 0.999.
        dynamic_anchor (dict | None): Config for DynamicAnchorModule.
        dapg_loss (dict | None): Config for DAPGLoss.
        affinity_loss (dict | None): Config for AffinityBoundaryLoss.
    """

    def __init__(self,
                 # DAPCN-specific parameters
                 da_position='before_fusion',
                 da_feature_dim=None,
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 contrastive_lambda=0.1,
                 boundary_mode='sobel',
                 boundary_loss_mode='binary',
                 hybrid_binary_weight=0.5,
                 contrastive_temperature=0.07,
                 contrastive_sample_ratio=0.1,
                 warmup_iters=500,
                 num_prototypes_per_class=1,
                 prototype_ema=0.999,
                 prototype_init_strategy='zeros',
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 **kwargs):
        """Initialize SegFormerDAPCNHead.

        Args:
            boundary_lambda (float): Weight for boundary loss.
            proto_lambda (float): Weight for DAPG prototype loss.
            contrastive_lambda (float): Weight for memory-bank contrastive loss.
            boundary_mode (str): Gradient operator for boundary extraction.
            boundary_loss_mode (str): Binary, affinity, or hybrid mode.
            hybrid_binary_weight (float): Binary weight in hybrid mode.
            contrastive_temperature (float): InfoNCE temperature.
            contrastive_sample_ratio (float): Fraction of pixels to sample.
            warmup_iters (int): Iterations before enabling contrastive loss.
            num_prototypes_per_class (int): Number of prototypes per class.
            prototype_ema (float): EMA momentum for prototype updates.
            prototype_init_strategy (str): Prototype initialization strategy.
            dynamic_anchor (dict | None): Config for DynamicAnchorModule.
            dapg_loss (dict | None): Config for DAPGLoss.
            affinity_loss (dict | None): Config for AffinityBoundaryLoss.
            **kwargs: Additional BaseDecodeHead arguments.
        """
        # Initialize BaseDecodeHead with multiple_select transform
        super(SegFormerDAPCNHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # ---- SegFormer-specific setup ----------------------------------------
        decoder_params = kwargs.get('decoder_params', {})
        embedding_dim = decoder_params.get('embed_dim', 256)
        conv_kernel_size = decoder_params.get('conv_kernel_size', 1)

        # Build linear projection layers for each input
        self.linear_c = {}
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        # Fusion layer
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs.get('norm_cfg', None))

        # Prediction layer
        self.linear_pred = build_conv_layer(
            self.conv_cfg, embedding_dim, self.num_classes, kernel_size=1)

        # ---- Initialize DAPCN components ------------------------------------
        # For SegFormer:
        #   before_fusion: DA operates on inputs[-1] with C=in_channels[-1]=512
        #   after_fusion:  DA operates on fused feature with C=embed_dim=256
        self.init_dapcn(
            da_position=da_position,
            da_feature_dim=da_feature_dim,
            boundary_lambda=boundary_lambda,
            proto_lambda=proto_lambda,
            contrastive_lambda=contrastive_lambda,
            boundary_mode=boundary_mode,
            boundary_loss_mode=boundary_loss_mode,
            hybrid_binary_weight=hybrid_binary_weight,
            contrastive_temperature=contrastive_temperature,
            contrastive_sample_ratio=contrastive_sample_ratio,
            warmup_iters=warmup_iters,
            num_prototypes_per_class=num_prototypes_per_class,
            prototype_ema=prototype_ema,
            prototype_init_strategy=prototype_init_strategy,
            dynamic_anchor=dynamic_anchor,
            dapg_loss=dapg_loss,
            affinity_loss=affinity_loss,
        )

    def forward(self, inputs):
        """Forward pass for SegFormer decoder.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.

        Returns:
            Tensor: Segmentation logits (B, C, H, W).
        """
        x = inputs
        n, _, h, w = x[-1].shape

        _c = {}
        for i in self.in_index:
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)

        return x

    def forward_train(self, inputs, img_metas, gt_semantic_seg,
                      train_cfg, seg_weight=None):
        """Forward + loss computation with DAPCN auxiliary objectives.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.
            img_metas (list[dict]): Image metadata.
            gt_semantic_seg (Tensor): Ground truth semantic segmentation map.
            train_cfg (dict): Training configuration.
            seg_weight (Tensor | None): Per-pixel segmentation weight mask.

        Returns:
            dict: Loss dictionary with CE loss and DAPCN auxiliary losses.
        """
        # ---- Standard segmentation forward --------------------------------
        seg_logits = self.forward(inputs)                    # (B, C, H', W')
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        # ---- Get fused feature before classification -----------------------
        fused_feature = self._get_fused_feature(inputs)

        # ---- Compute DAPCN auxiliary losses --------------------------------
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)

        # ---- Merge losses --------------------------------------------------
        losses.update(dapcn_losses)

        return losses

    def _get_fused_feature(self, inputs):
        """Re-derive the fused decoder feature (before cls_seg).

        This mirrors the forward path but stops before linear_pred,
        returning the (B, channels, H, W) representation that the
        memory bank should store.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.

        Returns:
            Tensor: Fused decoder feature (B, D, H, W).
        """
        x = inputs
        n, _, h, w = x[-1].shape

        # DEBUG: Check inputs
        for idx, inp in enumerate(x):
            if torch.isnan(inp).any():
                raise ValueError(f"_get_fused_feature: inputs[{idx}] contains NaN")

        _c = {}
        for i in self.in_index:
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)
            # DEBUG: Check after MLP
            if torch.isnan(_c[i]).any():
                raise ValueError(f"_get_fused_feature: _c[{i}] after MLP/resize contains NaN")

        cat_features = torch.cat(list(_c.values()), dim=1)
        if torch.isnan(cat_features).any():
            raise ValueError(f"_get_fused_feature: concatenated features contain NaN")

        fused = self.linear_fuse(cat_features)
        if torch.isnan(fused).any():
            raise ValueError(f"_get_fused_feature: fused features after linear_fuse contain NaN")

        if self.dropout is not None:
            fused = self.dropout(fused)

        return fused
