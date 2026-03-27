# ---------------------------------------------------------------
# KNetHead: Kernel-based decode head for KNet architecture
# Uses learnable kernel parameters and cross-attention mechanisms
# for feature-guided segmentation prediction.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin


@HEADS.register_module()
class KNetHead(BaseDecodeHead):
    """Kernel-based decode head for KNet architecture.

    KNetHead uses learnable kernel parameters and cross-attention
    mechanisms to enable feature-guided segmentation prediction.
    It efficiently captures spatial and channel-wise interactions
    for improved segmentation accuracy.

    Args:
        in_channels (int|Sequence[int]): Input channels from backbone.
        channels (int): Intermediate channels in decoder.
        num_classes (int): Number of segmentation classes.
        num_kernels (int): Number of learnable kernel parameters.
            Default: 256.
        kernel_dim (int): Dimension of each kernel. Default: 256.
        interact_cfg (dict): Cross-attention/interaction config. Default: None.
        **kwargs: Additional BaseDecodeHead arguments.
    """

    def __init__(self,
                 num_kernels=256,
                 kernel_dim=256,
                 interact_cfg=None,
                 **kwargs):
        """Initialize KNetHead.

        Args:
            num_kernels (int): Number of learnable kernel parameters.
            kernel_dim (int): Dimension of each kernel.
            interact_cfg (dict): Configuration for kernel-feature interaction.
            **kwargs: Additional BaseDecodeHead arguments.
        """
        super(KNetHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

        # ---- Build projection layers for multi-scale features ---------------
        self.proj_layers = nn.ModuleList()
        if isinstance(self.in_channels, (list, tuple)):
            for in_c in self.in_channels:
                self.proj_layers.append(
                    build_conv_layer(self.conv_cfg, in_c, kernel_dim, kernel_size=1))
        else:
            self.proj_layers.append(
                build_conv_layer(self.conv_cfg, self.in_channels, kernel_dim, kernel_size=1))

        # ---- Learnable kernel parameters ------------------------------------
        # Shape: (num_kernels, kernel_dim)
        self.kernels = nn.Parameter(
            torch.randn(num_kernels, kernel_dim) * 0.02)

        # ---- Kernel-feature interaction module ------------------------------
        self.interact_cfg = interact_cfg or {}
        self.build_interact_module()

        # ---- Fusion and classification ----------------------------------------
        self.fuse_conv = ConvModule(
            in_channels=num_kernels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=kwargs.get('norm_cfg', None),
            act_cfg=kwargs.get('act_cfg', dict(type='ReLU')))

        self.cls_seg = build_conv_layer(
            self.conv_cfg, self.channels, self.num_classes, kernel_size=1)

    def build_interact_module(self):
        """Build cross-attention or interaction module.

        This module computes interactions between kernels and
        projected features to produce kernel activation maps.
        """
        interact_type = self.interact_cfg.get('type', 'dot_product')

        if interact_type == 'dot_product':
            # Simple dot-product attention
            self.interact_fn = self._dot_product_interact
        elif interact_type == 'scaled_dot_product':
            # Scaled dot-product with temperature
            self.temperature = self.interact_cfg.get('temperature', 1.0)
            self.interact_fn = self._scaled_dot_product_interact
        elif interact_type == 'bilinear':
            # Bilinear interaction with learned weights
            self.bilinear = nn.Bilinear(
                self.kernel_dim, self.kernel_dim, self.num_kernels)
            self.interact_fn = self._bilinear_interact
        else:
            # Default: dot-product
            self.interact_fn = self._dot_product_interact

    def _dot_product_interact(self, kernels, feat_proj):
        """Dot-product interaction between kernels and features.

        Args:
            kernels (Tensor): Kernel parameters (K, D).
            feat_proj (Tensor): Projected features (B, D, H, W).

        Returns:
            Tensor: Kernel activation map (B, K, H, W).
        """
        # feat_proj: (B, D, H, W) -> (B, H, W, D)
        B, D, H, W = feat_proj.shape
        feat_flat = feat_proj.permute(0, 2, 3, 1).reshape(B * H * W, D)

        # kernels: (K, D) -> dot product -> (B*H*W, K)
        interact = torch.matmul(feat_flat, kernels.t())

        # Reshape back: (B, H, W, K) -> (B, K, H, W)
        interact = interact.reshape(B, H, W, self.num_kernels).permute(
            0, 3, 1, 2).contiguous()

        return interact

    def _scaled_dot_product_interact(self, kernels, feat_proj):
        """Scaled dot-product interaction with temperature.

        Args:
            kernels (Tensor): Kernel parameters (K, D).
            feat_proj (Tensor): Projected features (B, D, H, W).

        Returns:
            Tensor: Kernel activation map (B, K, H, W).
        """
        B, D, H, W = feat_proj.shape
        feat_flat = feat_proj.permute(0, 2, 3, 1).reshape(B * H * W, D)

        # Scaled dot-product with temperature scaling
        interact = torch.matmul(feat_flat, kernels.t()) / (
            self.temperature * (D ** 0.5))

        interact = interact.reshape(B, H, W, self.num_kernels).permute(
            0, 3, 1, 2).contiguous()

        return interact

    def _bilinear_interact(self, kernels, feat_proj):
        """Bilinear interaction (more expressive but slower).

        Args:
            kernels (Tensor): Kernel parameters (K, D).
            feat_proj (Tensor): Projected features (B, D, H, W).

        Returns:
            Tensor: Kernel activation map (B, K, H, W).
        """
        B, D, H, W = feat_proj.shape
        feat_flat = feat_proj.permute(0, 2, 3, 1).reshape(B * H * W, D)

        # Bilinear interaction: output shape (B*H*W, K)
        interact = torch.stack([
            self.bilinear(feat_flat, kernels[k:k+1].expand_as(feat_flat))
            for k in range(self.num_kernels)
        ], dim=1)

        interact = interact.reshape(B, H, W, self.num_kernels).permute(
            0, 3, 1, 2).contiguous()

        return interact

    def forward(self, inputs):
        """Forward pass for KNetHead.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.

        Returns:
            Tensor: Segmentation logits (B, C, H, W).
        """
        x = inputs
        ref_size = x[0].size()[2:]

        # ---- Project and aggregate multi-scale features ---------------------
        proj_feats = []
        for i, feat in enumerate(x):
            proj = self.proj_layers[min(i, len(self.proj_layers) - 1)](feat)

            # Resize to reference size if needed
            if proj.size()[2:] != ref_size:
                proj = resize(
                    proj, size=ref_size, mode='bilinear', align_corners=False)

            proj_feats.append(proj)

        # Simple fusion: average or concatenate and reduce
        if len(proj_feats) == 1:
            fused_feat = proj_feats[0]
        else:
            # Average multi-scale features
            fused_feat = torch.stack(proj_feats, dim=0).mean(dim=0)

        # ---- Kernel-feature interaction ------------------------------------
        kernel_activation = self.interact_fn(self.kernels, fused_feat)

        # ---- Fusion and classification ----------------------------------------
        x = self.fuse_conv(kernel_activation)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.cls_seg(x)

        return x


@HEADS.register_module()
class KNetDAPCNHead(DAPCNHeadMixin, KNetHead):
    """KNet decode head augmented with DAPCN auxiliary losses.

    Combines KNetHead's kernel-based feature interaction with
    DAPCN auxiliary loss components:
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

    KNet Args:
        in_channels (int|Sequence[int]): Input channels from backbone.
        channels (int): Intermediate channels in decoder.
        num_classes (int): Number of segmentation classes.
        num_kernels (int): Number of learnable kernel parameters.
            Default: 256.
        kernel_dim (int): Dimension of each kernel. Default: 256.
        interact_cfg (dict): Cross-attention/interaction config.
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
                 # KNet-specific parameters
                 num_kernels=256,
                 kernel_dim=256,
                 interact_cfg=None,
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
        """Initialize KNetDAPCNHead.

        Args:
            num_kernels (int): Number of learnable kernel parameters.
            kernel_dim (int): Dimension of each kernel.
            interact_cfg (dict): Configuration for kernel-feature interaction.
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
        # Initialize KNetHead with kernel-specific parameters
        super(KNetDAPCNHead, self).__init__(
            num_kernels=num_kernels,
            kernel_dim=kernel_dim,
            interact_cfg=interact_cfg,
            **kwargs)

        # ---- Initialize DAPCN components ------------------------------------
        # For KNet:
        #   before_fusion: DA operates on inputs[-1] with C=in_channels[-1]=2048
        #   after_fusion:  DA operates on fused feature with C=channels=256
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

        This mirrors the forward path but stops before cls_seg,
        returning the (B, channels, H, W) representation that the
        memory bank should store.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.

        Returns:
            Tensor: Fused decoder feature (B, D, H, W).
        """
        x = inputs
        ref_size = x[0].size()[2:]

        # ---- Project and aggregate multi-scale features --------------------
        proj_feats = []
        for i, feat in enumerate(x):
            proj = self.proj_layers[min(i, len(self.proj_layers) - 1)](feat)

            # Resize to reference size if needed
            if proj.size()[2:] != ref_size:
                proj = resize(
                    proj, size=ref_size, mode='bilinear', align_corners=False)

            proj_feats.append(proj)

        # Simple fusion: average multi-scale features
        if len(proj_feats) == 1:
            fused_feat = proj_feats[0]
        else:
            fused_feat = torch.stack(proj_feats, dim=0).mean(dim=0)

        # ---- Kernel-feature interaction ------------------------------------
        kernel_activation = self.interact_fn(self.kernels, fused_feat)

        # ---- Fusion (before classification) --------------------------------
        fused = self.fuse_conv(kernel_activation)

        if self.dropout is not None:
            fused = self.dropout(fused)

        return fused
