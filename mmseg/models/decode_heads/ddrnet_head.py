# ---------------------------------------------------------------
# DDRNetHead and DDRNetDAPCNHead
# Lightweight decode heads for DDRNet with optional DAPCN support
# All sub-layers use official mmcv builders (ConvModule,
# build_conv_layer, build_norm_layer) for consistency with
# the MMSegmentation framework.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer

from mmseg.models.builder import HEADS
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin


@HEADS.register_module()
class DDRNetHead(BaseDecodeHead):
    """Simple lightweight decode head for DDRNet.

    Takes the fused feature from DDRNet backbone and applies:
    - ConvModule(in_channels, channels, 3, norm+act) → cls_seg

    All convolution and normalisation layers are constructed through
    mmcv builders so that ``conv_cfg`` / ``norm_cfg`` from the config
    file are respected.

    Args:
        in_channels (int): Input channels (fused feature dimension).
        channels (int): Hidden channels for intermediate processing.
        num_classes (int): Number of semantic classes.
        dropout_ratio (float): Dropout ratio. Default: 0.1.
        conv_cfg (dict | None): Config for conv layers. Default: None.
        norm_cfg (dict): Config for norm layers. Default: dict(type='BN').
        act_cfg (dict): Config for activation layers.
            Default: dict(type='ReLU')
        in_index (int): Index of input feature. Default: -1.
        loss_decode (dict): Config of decode loss.
        ignore_index (int): Index to ignore in loss. Default: 255.
        align_corners (bool): align_corners for F.interpolate. Default: False.
        init_cfg (dict | list[dict]): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            in_index=in_index,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            align_corners=align_corners,
            init_cfg=init_cfg)

        # Lightweight processing: Conv3x3 + Norm + Act via ConvModule
        self.bottleneck = ConvModule(
            self.in_channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.

        Returns:
            Tensor: Output logits with shape (B, num_classes, H, W).
        """
        x = self._transform_inputs(inputs)
        x = self.bottleneck(x)
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.conv_seg(x)
        return output


@HEADS.register_module()
class DDRNetDAPCNHead(DAPCNHeadMixin, DDRNetHead):
    """DDRNetHead augmented with DAPCN losses.

    Combines lightweight DDRNet decoding with boundary-aware,
    prototype-based, and contrastive learning objectives.

    Args (on top of DDRNetHead):
        da_position (str): Position of dynamic anchor.
            'before_fusion' or 'after_fusion'. Default: 'before_fusion'.
        da_feature_dim (int | None): Override DA feature dim. Default: None.
        boundary_lambda (float): Weight for boundary loss. Default: 0.3.
        proto_lambda (float): Weight for DAPG loss. Default: 0.1.
        contrastive_lambda (float): Weight for contrastive loss. Default: 0.1.
        boundary_mode (str): Boundary extraction mode. Default: 'sobel'.
        boundary_loss_mode (str): Boundary loss mode. Default: 'binary'.
        hybrid_binary_weight (float): Binary weight in hybrid mode. Default: 0.5.
        contrastive_temperature (float): InfoNCE temperature. Default: 0.07.
        contrastive_sample_ratio (float): Sampling ratio. Default: 0.1.
        warmup_iters (int): Warmup iters for contrastive loss. Default: 500.
        num_prototypes_per_class (int): Prototypes per class. Default: 1.
        prototype_ema (float): EMA momentum. Default: 0.999.
        dynamic_anchor (dict | None): Config for DynamicAnchorModule.
        dapg_loss (dict | None): Config for DAPGLoss.
        affinity_loss (dict | None): Config for AffinityBoundaryLoss.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 # --- DAPCN parameters ---
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
                 # --- Base head parameters ---
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        # Initialize parent class (DDRNetHead)
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            in_index=in_index,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            align_corners=align_corners,
            init_cfg=init_cfg)

        # Initialize DAPCN mixin
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
            affinity_loss=affinity_loss)

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward + loss computation with DAPCN auxiliary objectives.

        Args:
            inputs (list[Tensor]): List of multi-level backbone features.
            img_metas (list[dict]): List of image metadata.
            gt_semantic_seg (Tensor): Ground-truth segmentation labels.
            train_cfg (dict): Training configuration.
            seg_weight (Tensor | None): Per-pixel segmentation weights.

        Returns:
            dict[str, Tensor]: Dictionary of losses.
        """
        # ---- Standard segmentation forward --------------------------------
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        # ---- Get fused feature for DAPCN ----------------------------------
        # Re-derive the intermediate feature after bottleneck ConvModule
        # (before classification). This has shape (B, channels, H, W).
        x = self._transform_inputs(inputs)
        fused_feature = self.bottleneck(x)

        # ---- DAPCN auxiliary losses ----------------------------------------
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)

        return losses
