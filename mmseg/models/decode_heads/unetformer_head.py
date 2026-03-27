# ---------------------------------------------------------------
# UNetFormer Decode Head: Feature Pyramid with Skip Connections
# Combines transformer encoder features with U-Net style decoder
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin


@HEADS.register_module()
class UNetFormerHead(BaseDecodeHead):
    """UNetFormer decode head with Feature Pyramid and Skip Connections.

    This head takes multi-scale features from the encoder and progressively
    upsamples them using U-Net style skip connections, similar to the decoder
    in UNet++ or Feature Pyramid Networks.

    Key components:
    1. Feature projection: Project each scale to unified channel dimension
    2. Progressive upsampling: Upsample and fuse features at each level
    3. Skip connections: Concatenate with corresponding encoder features
    4. Refinement blocks: ConvModule at each decoder level

    Args:
        in_channels (int|list[int]): Number of input channels.
            If int: single input, use in_index to select.
            If list: multiple inputs from different backbone levels.
        channels (int): Number of intermediate channels. Default: 256.
        num_classes (int): Number of semantic classes.
        in_index (int|list[int]): Input feature indices. Default: -1.
        input_transform (str|None): Input transformation type.
            'multiple_select' for multi-scale input. Default: None.
        dropout_ratio (float): Dropout ratio. Default: 0.1.
        conv_cfg (dict|None): Config for convolution. Default: None.
        norm_cfg (dict|None): Config for normalization. Default: dict(type='BN').
        act_cfg (dict): Config for activation. Default: dict(type='ReLU').
        align_corners (bool): Align corners in interpolation. Default: False.
        init_cfg (dict): Initialization config.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 in_index=-1,
                 input_transform=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(UNetFormerHead, self).__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            init_cfg=init_cfg)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # Handle multi-scale input
        if isinstance(in_channels, (list, tuple)):
            self.num_scales = len(in_channels)
            self.in_channels_list = in_channels
        else:
            self.num_scales = 1
            self.in_channels_list = [in_channels]

        # Feature projection layers: project each scale to channels dim
        self.proj_layers = nn.ModuleList()
        for in_ch in self.in_channels_list:
            proj = ConvModule(
                in_ch,
                channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.proj_layers.append(proj)

        # Decoder: Progressively fuse and upsample features
        # For multi-scale: start from highest-level feature and progressively
        # upsample while fusing with lower-level features
        self.decoder_layers = nn.ModuleList()
        if self.num_scales > 1:
            # For each fusion level
            for i in range(self.num_scales - 1):
                # Input: fused feature from previous level + current level feature
                # Both have 'channels' channels after projection
                decoder = nn.Sequential(
                    ConvModule(
                        in_channels=channels * 2,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
                )
                self.decoder_layers.append(decoder)
        else:
            # Single scale: just refine
            decoder = nn.Sequential(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )
            self.decoder_layers.append(decoder)

        # Final refinement block
        self.final_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (Tensor|list[Tensor]): Input features.
                If single tensor: (B, C, H, W)
                If list: [(B, C1, H1, W1), (B, C2, H2, W2), ...]

        Returns:
            Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        # Handle input transformation
        if isinstance(inputs, list):
            features = [inputs[i] for i in self.in_index] if isinstance(
                self.in_index, (list, tuple)) else [inputs[self.in_index]]
        else:
            features = [inputs]

        # Project all features to same channel dimension
        projected = []
        for i, feat in enumerate(features):
            if i < len(self.proj_layers):
                proj_feat = self.proj_layers[i](feat)
            else:
                proj_feat = self.proj_layers[-1](feat)
            projected.append(proj_feat)

        # Progressive fusion with upsampling
        if len(projected) > 1:
            # Start from smallest (highest level)
            fused = projected[-1]

            # Progressively upsample and fuse with larger features
            for i in range(len(projected) - 2, -1, -1):
                # Upsample fused feature to match next scale
                target_shape = projected[i].shape[2:]
                fused = F.interpolate(
                    fused,
                    size=target_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)

                # Concatenate and decode
                fused = torch.cat([fused, projected[i]], dim=1)
                fused = self.decoder_layers[len(projected) - 2 - i](fused)
        else:
            # Single scale: just refine
            fused = self.decoder_layers[0](projected[0])

        # Final refinement
        fused = self.final_conv(fused)

        # Classify
        seg_logits = self.cls_seg(fused)

        return seg_logits

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image meta info.
            gt_semantic_seg (Tensor): Ground truth segmentation map.
            train_cfg (dict): Training config dict.
            seg_weight (Tensor, optional): Segmentation weight. Default: None.

        Returns:
            dict: Dictionary of losses.
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        return losses


class UNetFormerHead_(UNetFormerHead):
    """Alias for backward compatibility."""
    pass


@HEADS.register_module()
class UNetFormerDAPCNHead(DAPCNHeadMixin, UNetFormerHead):
    """UNetFormer head with DAPCN support.

    Combines Feature Pyramid decoder with Dynamic Attention-based
    Prototype Contrastive Network auxiliary losses.

    Args:
        in_channels (int|list[int]): Number of input channels.
        channels (int): Number of intermediate channels. Default: 256.
        num_classes (int): Number of semantic classes.
        in_index (int|list[int]): Input feature indices. Default: -1.
        input_transform (str|None): Input transformation type. Default: None.
        dropout_ratio (float): Dropout ratio. Default: 0.1.
        conv_cfg (dict|None): Config for convolution. Default: None.
        norm_cfg (dict|None): Config for normalization. Default: dict(type='BN').
        act_cfg (dict): Config for activation. Default: dict(type='ReLU').
        align_corners (bool): Align corners in interpolation. Default: False.
        # DAPCN-specific args
        da_position (str): Dynamic Anchor position. Default: 'after_fusion'.
        boundary_lambda (float): Boundary loss weight. Default: 0.3.
        proto_lambda (float): Prototype loss weight. Default: 0.1.
        contrastive_lambda (float): Contrastive loss weight. Default: 0.1.
        boundary_mode (str): Boundary extraction mode. Default: 'sobel'.
        boundary_loss_mode (str): Boundary loss type. Default: 'binary'.
        dynamic_anchor (dict): DynamicAnchorModule config. Default: None.
        dapg_loss (dict): DAPGLoss config. Default: None.
        affinity_loss (dict): AffinityBoundaryLoss config. Default: None.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 in_index=-1,
                 input_transform=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 # DAPCN args
                 da_position='after_fusion',
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 contrastive_lambda=0.1,
                 boundary_mode='sobel',
                 boundary_loss_mode='binary',
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        # Initialize UNetFormerHead
        UNetFormerHead.__init__(
            self,
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            init_cfg=init_cfg)

        # Initialize DAPCN components
        # For UNetFormer:
        #   before_fusion: DA operates on inputs[-1] with C=in_channels[-1]=512
        #   after_fusion:  DA operates on fused feature with C=channels=256
        # Let the mixin auto-infer the correct feature_dim.
        self.init_dapcn(
            da_position=da_position,
            boundary_lambda=boundary_lambda,
            proto_lambda=proto_lambda,
            contrastive_lambda=contrastive_lambda,
            boundary_mode=boundary_mode,
            boundary_loss_mode=boundary_loss_mode,
            dynamic_anchor=dynamic_anchor,
            dapg_loss=dapg_loss,
            affinity_loss=affinity_loss)

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training with DAPCN losses.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image meta info.
            gt_semantic_seg (Tensor): Ground truth segmentation map.
            train_cfg (dict): Training config dict.
            seg_weight (Tensor, optional): Segmentation weight. Default: None.

        Returns:
            dict: Dictionary of segmentation and DAPCN losses.
        """
        # Standard segmentation forward
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        # Get fused feature for DAPCN
        # Re-derive the fused feature by running through projection and decoder
        if isinstance(inputs, list):
            features = [inputs[i] for i in self.in_index] if isinstance(
                self.in_index, (list, tuple)) else [inputs[self.in_index]]
        else:
            features = [inputs]

        # Project all features to same channel dimension
        projected = []
        for i, feat in enumerate(features):
            if i < len(self.proj_layers):
                proj_feat = self.proj_layers[i](feat)
            else:
                proj_feat = self.proj_layers[-1](feat)
            projected.append(proj_feat)

        # Progressive fusion (same as forward)
        if len(projected) > 1:
            fused = projected[-1]
            for i in range(len(projected) - 2, -1, -1):
                target_shape = projected[i].shape[2:]
                fused = F.interpolate(
                    fused,
                    size=target_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)
                fused = torch.cat([fused, projected[i]], dim=1)
                fused = self.decoder_layers[len(projected) - 2 - i](fused)
        else:
            fused = self.decoder_layers[0](projected[0])

        # Fused feature before final classification
        fused_feature = self.final_conv(fused)

        # Compute DAPCN losses
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)

        return losses


class UNetFormerDAPCNHead_(UNetFormerDAPCNHead):
    """Alias for backward compatibility."""
    pass
