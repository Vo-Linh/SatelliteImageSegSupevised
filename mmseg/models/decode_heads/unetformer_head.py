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
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
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
            loss_decode=loss_decode,
            ignore_index=ignore_index,
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

    def _decode_features(self, inputs):
        """Run projection + progressive fusion + final conv.

        Shared by forward() and forward_train() to avoid duplicate computation.

        Args:
            inputs (Tensor|list[Tensor]): Input features.

        Returns:
            Tensor: Fused feature of shape (B, channels, H, W).
        """
        # Handle input transformation
        if isinstance(inputs, (list, tuple)):
            features = [inputs[i] for i in self.in_index] if isinstance(
                self.in_index, (list, tuple)) else [inputs[self.in_index]]
        else:
            features = [inputs]

        for i, feat in enumerate(features):
            if torch.isnan(feat).any():
                raise ValueError(
                    f"UNetFormerHead: backbone feature[{i}] contains NaN "
                    f"(shape={feat.shape}, range=[{feat.min():.4f}, {feat.max():.4f}])"
                )

        # Project all features to same channel dimension
        projected = []
        for i, feat in enumerate(features):
            if i < len(self.proj_layers):
                proj_feat = self.proj_layers[i](feat)
            else:
                proj_feat = self.proj_layers[-1](feat)
            if torch.isnan(proj_feat).any():
                raise ValueError(
                    f"UNetFormerHead: proj_feat[{i}] contains NaN"
                )
            projected.append(proj_feat)

        # Progressive fusion with upsampling
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
                if torch.isnan(fused).any():
                    raise ValueError(
                        f"UNetFormerHead: decoder_layer[{len(projected) - 2 - i}] output NaN"
                    )
        else:
            fused = self.decoder_layers[0](projected[0])
            if torch.isnan(fused).any():
                raise ValueError(
                    "UNetFormerHead: single-scale decoder_layer output NaN"
                )

        fused = self.final_conv(fused)
        if torch.isnan(fused).any():
            raise ValueError("UNetFormerHead: final_conv output NaN")

        return fused

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (Tensor|list[Tensor]): Input features.

        Returns:
            Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        fused = self._decode_features(inputs)
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
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 # DAPCN args
                 da_position='after_fusion',
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 contrastive_lambda=0.1,
                 contrastive_temperature=0.07,
                 contrastive_sample_ratio=0.1,
                 boundary_mode='sobel',
                 boundary_loss_mode='binary',
                 num_prototypes_per_class=1,
                 prototype_ema=0.999,
                 warmup_iters=500,
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 ignore_index=255,
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
            loss_decode=loss_decode,
            ignore_index=ignore_index,
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
            contrastive_temperature=contrastive_temperature,
            contrastive_sample_ratio=contrastive_sample_ratio,
            boundary_mode=boundary_mode,
            boundary_loss_mode=boundary_loss_mode,
            num_prototypes_per_class=num_prototypes_per_class,
            prototype_ema=prototype_ema,
            warmup_iters=warmup_iters,
            dynamic_anchor=dynamic_anchor,
            dapg_loss=dapg_loss,
            affinity_loss=affinity_loss)

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        # Single decoder pass — get fused feature + logits together
        fused_feature = self._decode_features(inputs)

        if torch.isnan(fused_feature).any():
            raise ValueError(
                f"UNetFormerDAPCNHead: fused_feature contains NaN "
                f"(shape={fused_feature.shape})"
            )

        seg_logits = self.cls_seg(fused_feature)

        if torch.isnan(seg_logits).any():
            raise ValueError(
                f"UNetFormerDAPCNHead: seg_logits contains NaN "
                f"(shape={seg_logits.shape})"
            )

        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        # DAPCN losses use the same fused_feature (no recomputation)
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)

        return losses


class UNetFormerDAPCNHead_(UNetFormerDAPCNHead):
    """Alias for backward compatibility."""
    pass
