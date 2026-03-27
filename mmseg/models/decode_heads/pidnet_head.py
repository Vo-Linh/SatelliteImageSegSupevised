# ---------------------------------------------------------------
# PIDNetHead and PIDNetDAPCNHead
# Decode heads for PIDNet with boundary-aware design and optional DAPCN support
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

from mmseg.models.builder import HEADS
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin


class PagFM(BaseModule):
    """Pixel-Attention-Guided Fusion Module.

    A lightweight fusion module that applies channel attention before
    combining two feature maps.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        conv_cfg (dict | None): Config for conv layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. Default: dict(type='BN').
        act_cfg (dict): Config for activation layers. Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        # Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = build_conv_layer(
            conv_cfg, in_channels, in_channels // 16, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = build_conv_layer(
            conv_cfg, in_channels // 16, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Fusion convolution
        self.fuse_conv = ConvModule(
            in_channels * 2,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x1, x2):
        """Fuse two feature maps with pixel attention guidance.

        Args:
            x1: First feature map (typically detail/high-res)
            x2: Second feature map (typically boundary/context)

        Returns:
            Fused feature map
        """
        # Channel attention on x2
        att = self.gap(x2)
        att = self.fc_1(att)
        att = self.relu(att)
        att = self.fc_2(att)
        att = self.sigmoid(att)
        x2_att = x2 * att
        out = torch.cat([x1, x2_att], dim=1)
        out = self.fuse_conv(out)
        return out


@HEADS.register_module()
class PIDNetHead(BaseDecodeHead):
    """Decode head for PIDNet with boundary-aware design.

    Takes fused features from PIDNet's three branches (P, I, D) and
    produces both segmentation logits and boundary predictions.

    Args:
        in_channels (int): Input channels (fused feature dimension from PIDNet).
        channels (int): Hidden channels for intermediate processing.
        num_classes (int): Number of semantic classes.
        boundary_channels (int): Channels for boundary head. Default: 1.
        dropout_ratio (float): Dropout ratio. Default: 0.1.
        conv_cfg (dict | None): Config for conv layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. Default: None.
        act_cfg (dict): Config for activation layers.
            Default: dict(type='ReLU')
        in_index (int): Index of input feature. Default: -1.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ignore_index (int): Index to ignore in loss. Default: 255.
        align_corners (bool): align_corners for F.interpolate. Default: False.
        init_cfg (dict | list[dict]): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 boundary_channels=1,
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

        self.boundary_channels = boundary_channels

        # Fusion module for combining features
        self.fuse = PagFM(
            self.in_channels,
            channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Boundary head (optional auxiliary output)
        self.boundary_conv = ConvModule(
            channels,
            channels // 2,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.boundary_pred = build_conv_layer(
            conv_cfg, channels // 2, boundary_channels, kernel_size=1)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (Tensor): Input feature with shape (B, C, H, W).

        Returns:
            Tensor: Output logits with shape (B, num_classes, H, W).
        """
        x = self._transform_inputs(inputs)

        # Fuse with attention guidance (x self-fused)
        fused = self.fuse(x, x)

        # Classification head
        if self.dropout is not None:
            fused = self.dropout(fused)
        output = self.conv_seg(fused)

        return output

    def forward_dummy(self, inputs):
        """Dummy forward pass for model size estimation.

        Returns both segmentation and boundary outputs.
        """
        x = self._transform_inputs(inputs)
        fused = self.fuse(x, x)

        if self.dropout is not None:
            fused = self.dropout(fused)

        seg_logits = self.conv_seg(fused)
        boundary_logits = self.boundary_conv(fused)
        boundary_logits = self.boundary_pred(boundary_logits)

        return seg_logits, boundary_logits


@HEADS.register_module()
class PIDNetDAPCNHead(DAPCNHeadMixin, PIDNetHead):
    """PIDNetHead augmented with DAPCN losses.

    Combines boundary-aware PIDNet decoding with DAPCN's
    prototype-based and contrastive learning objectives.

    Args (on top of PIDNetHead):
        da_position (str): Position of dynamic anchor. Default: 'before_cls'.
        boundary_lambda (float): Weight for boundary loss. Default: 0.3.
        proto_lambda (float): Weight for DAPG loss. Default: 0.1.
        contrastive_lambda (float): Weight for contrastive loss. Default: 0.1.
        boundary_mode (str): Boundary extraction mode ('sobel', 'laplacian', 'diff').
            Default: 'sobel'.
        boundary_loss_mode (str): Boundary loss mode ('binary', 'affinity', 'hybrid').
            Default: 'binary'.
        hybrid_binary_weight (float): Binary weight in hybrid mode. Default: 0.5.
        contrastive_temperature (float): InfoNCE temperature. Default: 0.07.
        contrastive_sample_ratio (float): Sampling ratio for contrastive loss.
            Default: 0.1.
        warmup_iters (int): Warmup iterations for contrastive loss. Default: 500.
        num_prototypes_per_class (int): Number of prototypes per class. Default: 1.
        prototype_ema (float): EMA momentum for prototypes. Default: 0.999.
        prototype_init_strategy (str): Prototype initialization strategy.
            Default: 'zeros'.
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
                 boundary_channels=1,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
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
        # Initialize parent class (PIDNetHead)
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            boundary_channels=boundary_channels,
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

        # ---- Get fused feature for DAPCN
        # For PIDNet, we re-derive the fused feature after fusion module
        x = self._transform_inputs(inputs)
        fused_feature = self.fuse(x, x)

        # ---- DAPCN auxiliary losses ----------------------------------------
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)

        return losses
