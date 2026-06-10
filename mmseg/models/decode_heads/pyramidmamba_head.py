# ---------------------------------------------------------------
# PyramidMamba decode head for mmsegmentation.
#
# Wraps the ported GeoSeg PyramidMamba decoder (see
# pyramidmamba_modules.py) as an mmseg BaseDecodeHead so it plugs into the
# stock EncoderDecoder + TIMMBackbone pipeline, exactly like UNetFormerHead.
#
# The backbone (e.g. ResNeXt101_32x16d via TIMMBackbone, out_indices=(1,2,3,4))
# emits four feature maps; this head selects the stride-4 (shallow) and
# stride-32 (deep) features via in_index, runs the Pyramid-Pooling + Mamba
# decoder, and classifies with cls_seg.
#
# PyramidMambaDAPCNHead adds the repo's DAPCN auxiliary losses through
# DAPCNHeadMixin, mirroring UNetFormerDAPCNHead.
# ---------------------------------------------------------------

from mmseg.models.builder import HEADS
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin
from .pyramidmamba_modules import PyramidMambaDecoder


@HEADS.register_module()
class PyramidMambaHead(BaseDecodeHead):
    """PyramidMamba decode head (EfficientPyramidMamba decoder).

    Consumes two backbone features — a shallow (stride-4) and a deep
    (stride-32) map selected through ``in_index`` — and fuses them with a
    Pyramid-Pooling + Mamba decoder. The fused feature has
    ``decode_channels // 2`` channels at the input resolution; ``cls_seg``
    produces the ``num_classes`` logits.

    Args:
        in_channels (list[int]): Channels of the selected backbone features,
            i.e. ``[shallow_ch, deep_ch]`` (e.g. ``[256, 2048]``).
        channels (int): Kept for API compatibility; the effective
            classifier width is ``decode_channels // 2``.
        encoder_channels (tuple[int]): Same as ``in_channels`` as a tuple
            ``(shallow_ch, deep_ch)``; ``[0]`` feeds the skip ``pre_conv`` and
            ``[-1]`` feeds the Mamba block.
        decode_channels (int): Decoder hidden width. Default: 128.
        last_feat_size (int): Deepest-feature spatial size at the train crop
            (= crop_size // 32). Default: 16 (for a 512 crop).
        d_state, d_conv, expand: Mamba SSM hyper-parameters.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 encoder_channels=(256, 2048),
                 decode_channels=128,
                 last_feat_size=16,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 in_index=(0, 3),
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                                  loss_weight=1.0),
                 backbone_nhwc=False,
                 ignore_index=255,
                 init_cfg=dict(type='Normal', std=0.01,
                               override=dict(name='conv_seg'))):
        super().__init__(
            in_channels=in_channels,
            channels=decode_channels // 2,
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
        self.encoder_channels = tuple(encoder_channels)
        self.decode_channels = decode_channels
        self.last_feat_size = last_feat_size
        # NHWC backbones (e.g. timm Swin) emit (B,H,W,C); permute to NCHW in _decode.
        self.backbone_nhwc = backbone_nhwc

        self.decoder = PyramidMambaDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decode_channels,
            last_feat_size=last_feat_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def _decode(self, inputs):
        """Run the PyramidMamba decoder, returning the fused feature.

        Args:
            inputs (list[Tensor]): Raw multi-level backbone features.

        Returns:
            Tensor: Fused feature (B, decode_channels // 2, H, W) at the
            input image resolution (before cls_seg).
        """
        feats = self._transform_inputs(inputs)  # [shallow (x0), deep (x3)]
        if self.backbone_nhwc:
            # timm Swin & co. output (B, H, W, C); the decoder expects NCHW.
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        x0, x3 = feats[0], feats[-1]
        return self.decoder(x0, x3)

    def forward(self, inputs):
        """Forward pass returning segmentation logits."""
        return self.cls_seg(self._decode(inputs))


@HEADS.register_module()
class PyramidMambaDAPCNHead(DAPCNHeadMixin, PyramidMambaHead):
    """PyramidMamba head with DAPCN auxiliary losses.

    Adds the Dynamic-Anchor Prototype Contrastive auxiliary losses
    (boundary + DAPG prototype grouping + optional contrastive) on top of
    PyramidMambaHead, mirroring ``UNetFormerDAPCNHead``. With
    ``da_position='after_fusion'`` the Dynamic Anchor Module operates on the
    ``decode_channels // 2``-dim fused decoder feature.

    See ``DAPCNHeadMixin`` for the auxiliary-loss arguments.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 encoder_channels=(256, 2048),
                 decode_channels=128,
                 last_feat_size=16,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 in_index=(0, 3),
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                                  loss_weight=1.0),
                 # DAPCN params
                 da_position='after_fusion',
                 da_feature_dim=None,
                 boundary_lambda=0.15,
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
                 backbone_nhwc=False,
                 ignore_index=255,
                 init_cfg=dict(type='Normal', std=0.01,
                               override=dict(name='conv_seg'))):
        # Initialize the base PyramidMamba head.
        PyramidMambaHead.__init__(
            self,
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            last_feat_size=last_feat_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backbone_nhwc=backbone_nhwc,
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
        # Initialize DAPCN auxiliary components.
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

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg,
                      seg_weight=None):
        """Forward + loss with DAPCN auxiliary losses (single decoder pass)."""
        fused_feature = self._decode(inputs)
        seg_logits = self.cls_seg(fused_feature)
        # Standard segmentation loss.
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        # DAPCN auxiliary losses.
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)
        return losses
