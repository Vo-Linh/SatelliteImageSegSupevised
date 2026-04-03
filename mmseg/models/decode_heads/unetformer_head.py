import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin
from .unetformer_modules import (
    ConvBNReLU, ConvBN, Conv, SeparableConvBNReLU, SeparableConvBN,
    Mlp, GlobalLocalAttention, Block, WF, FeatureRefinementHead,
)


@HEADS.register_module()
class UNetFormerHead(BaseDecodeHead):
    """UNetFormer decode head with GlobalLocalAttention decoder.
    
    Paper-faithful implementation: CNN encoder + Transformer decoder.
    
    Args:
        encoder_channels (tuple): Channel dims from backbone (64, 128, 256, 512 for B0).
        decode_channels (int): Decoder hidden channels (64 for B0).
        window_size (int): Window attention size. Default: 8.
        num_heads (int): Number of attention heads. Default: 8.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        drop_path_rate (float): DropPath rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 window_size=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop_path_rate=0.1,
                 in_index=[0, 1, 2, 3],
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 ignore_index=255,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super().__init__(
            in_channels=in_channels,
            channels=decode_channels,
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
            init_cfg=init_cfg,
        )
        self.encoder_channels = encoder_channels
        self.decode_channels = decode_channels
        self.align_corners = align_corners

        # Decoder components (paper architecture)
        # pre_conv4: project deepest features to decode_channels
        self.pre_conv4 = ConvBN(encoder_channels[3], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        # WF3 + Block3: weighted fusion with res3 + transformer
        self.wf3 = WF(in_channels=encoder_channels[2], decode_channels=decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        # WF2 + Block2: weighted fusion with res2 + transformer
        self.wf2 = WF(in_channels=encoder_channels[1], decode_channels=decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        # FeatureRefinementHead: final refinement with res1
        self.frh = FeatureRefinementHead(in_channels=encoder_channels[0], decode_channels=decode_channels)

    def forward(self, inputs):
        """Forward pass.
        
        Args:
            inputs (list[Tensor]): 4 feature maps from backbone [res1, res2, res3, res4].
        
        Returns:
            Tensor: Segmentation logits (B, num_classes, H/4, W/4).
        """
        # inputs = [res1(H/4), res2(H/8), res3(H/16), res4(H/32)]
        res1, res2, res3, res4 = inputs[0], inputs[1], inputs[2], inputs[3]

        # Deepest to shallowest processing
        x = self.pre_conv4(res4)      # (B, decode_channels, H/32, W/32)
        x = self.b4(x)                 # transformer at H/32
        x = self.wf3(x, res3)          # upsample + weighted fusion with res3 -> H/16
        x = self.b3(x)                 # transformer at H/16
        x = self.wf2(x, res2)          # upsample + weighted fusion with res2 -> H/8
        x = self.b2(x)                 # transformer at H/8
        x = self.frh(x, res1)          # feature refinement with res1 -> H/4

        # Classification (from BaseDecodeHead)
        output = self.cls_seg(x)
        return output

    def _get_fused_feature(self, inputs):
        """Get fused decoder feature BEFORE classification.
        
        Used by DAPCNHeadMixin for prototype learning.
        
        Args:
            inputs (list[Tensor]): 4 feature maps from backbone.
        
        Returns:
            Tensor: Fused feature (B, decode_channels, H/4, W/4).
        """
        res1, res2, res3, res4 = inputs[0], inputs[1], inputs[2], inputs[3]
        x = self.pre_conv4(res4)
        x = self.b4(x)
        x = self.wf3(x, res3)
        x = self.b3(x)
        x = self.wf2(x, res2)
        x = self.b2(x)
        x = self.frh(x, res1)
        return x


class UNetFormerHead_(UNetFormerHead):
    """Alias for backward compatibility."""
    pass


@HEADS.register_module()
class UNetFormerDAPCNHead(DAPCNHeadMixin, UNetFormerHead):
    """UNetFormer head with DAPCN support.
    
    Adds Dynamic Anchor Prototype Contrastive Network auxiliary losses.
    
    Args:
        da_position (str): 'before_fusion' or 'after_fusion'. Default: 'before_fusion'.
        da_feature_dim (int | None): DA feature dim. Auto-inferred if None.
        boundary_lambda (float): Boundary loss weight. Default: 0.3.
        proto_lambda (float): Prototype loss weight. Default: 0.1.
        contrastive_lambda (float): Contrastive loss weight. Default: 0.1.
        boundary_mode (str): 'sobel', 'laplacian', or 'diff'. Default: 'sobel'.
        boundary_loss_mode (str): 'binary', 'affinity', or 'hybrid'. Default: 'binary'.
        dynamic_anchor (dict | None): DynamicAnchorModule config.
        dapg_loss (dict | None): DAPGLoss config.
        affinity_loss (dict | None): AffinityBoundaryLoss config.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 window_size=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop_path_rate=0.1,
                 in_index=[0, 1, 2, 3],
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 # DAPCN params
                 da_position='before_fusion',
                 da_feature_dim=None,
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 contrastive_lambda=0.1,
                 contrastive_temperature=0.07,
                 contrastive_sample_ratio=0.1,
                 boundary_mode='sobel',
                 boundary_loss_mode='binary',
                 hybrid_binary_weight=0.5,
                 num_prototypes_per_class=1,
                 prototype_ema=0.999,
                 warmup_iters=500,
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 ignore_index=255,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))):
        # Initialize UNetFormerHead
        UNetFormerHead.__init__(
            self,
            in_channels=in_channels,
            channels=decode_channels,
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            init_cfg=init_cfg,
        )
        # Initialize DAPCN components
        self.init_dapcn(
            da_position=da_position,
            da_feature_dim=da_feature_dim,
            boundary_lambda=boundary_lambda,
            proto_lambda=proto_lambda,
            contrastive_lambda=contrastive_lambda,
            contrastive_temperature=contrastive_temperature,
            contrastive_sample_ratio=contrastive_sample_ratio,
            boundary_mode=boundary_mode,
            boundary_loss_mode=boundary_loss_mode,
            hybrid_binary_weight=hybrid_binary_weight,
            num_prototypes_per_class=num_prototypes_per_class,
            prototype_ema=prototype_ema,
            warmup_iters=warmup_iters,
            dynamic_anchor=dynamic_anchor,
            dapg_loss=dapg_loss,
            affinity_loss=affinity_loss,
        )

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, seg_weight=None):
        """Forward + loss with DAPCN auxiliary losses."""
        # Get segmentation logits
        seg_logits = self.forward(inputs)
        # Standard CE loss
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        # Get fused feature for DAPCN
        fused_feature = self._get_fused_feature(inputs)
        # DAPCN auxiliary losses
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)
        return losses


class UNetFormerDAPCNHead_(UNetFormerDAPCNHead):
    """Alias for backward compatibility."""
    pass