# ---------------------------------------------------------------
# SegMenter Decode Head: Mask Transformer for Semantic Segmentation
# Reference: "Segmenter: Transformer for Semantic Segmentation"
# (https://arxiv.org/abs/2105.05424)
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .dapcn_head_mixin import DAPCNHeadMixin


@HEADS.register_module()
class SegMenterHead(BaseDecodeHead):
    """SegMenter decode head using Mask Transformer.

    This head implements the Mask Transformer approach where:
    1. Learnable class embeddings serve as queries
    2. Cross-attention fuses class queries with patch features
    3. Per-class masks are generated via dot product with features

    The approach is particularly effective for:
    - Fine-grained semantic segmentation
    - Learning class-specific visual patterns
    - Hierarchical feature representation

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of intermediate channels (embed_dim).
        num_classes (int): Number of semantic classes.
        embed_dim (int): Embedding dimension for class queries. Default: 768.
        num_heads (int): Number of attention heads. Default: 12.
        num_transformer_layers (int): Number of transformer layers. Default: 2.
        hidden_dim (int): Hidden dimension of MLP. Default: 3072.
        dropout_ratio (float): Dropout ratio. Default: 0.1.
        conv_cfg (dict|None): Config for convolution. Default: None.
        norm_cfg (dict|None): Config for normalization. Default: None.
        act_cfg (dict): Config for activation. Default: dict(type='GELU').
        align_corners (bool): Align corners in interpolation. Default: False.
        init_cfg (dict): Initialization config. Default: Normal init.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 embed_dim=768,
                 num_heads=12,
                 num_transformer_layers=2,
                 hidden_dim=3072,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='GELU'),
                 align_corners=False,
                 ignore_index=255,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(SegMenterHead, self).__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            ignore_index=ignore_index,
            init_cfg=init_cfg)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.hidden_dim = hidden_dim

        # Project input features to embed_dim via MLP
        self.feat_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Learnable class embeddings (queries)
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, embed_dim))
        nn.init.normal_(self.class_embeddings, std=0.02)

        # Transformer layers for cross-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_ratio,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers)

        # Output projection to num_classes
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (Tensor): Input features of shape (B, C_in, H, W).

        Returns:
            Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        x = self._transform_inputs(inputs)
        B, C, H, W = x.shape

        # Flatten spatial dimensions
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Project to embed_dim
        x_proj = self.feat_proj(x_flat)  # (B*H*W, embed_dim)
        x_proj = x_proj.view(B, H * W, self.embed_dim)

        # Prepare queries: class embeddings
        class_queries = self.class_embeddings.unsqueeze(0).expand(
            B, -1, -1)  # (B, num_classes, embed_dim)

        # Cross-attention: class queries attend to patch features
        # Concatenate queries and features for transformer
        combined = torch.cat([class_queries, x_proj], dim=1)  # (B, num_classes+H*W, embed_dim)
        attended = self.transformer_encoder(combined)

        # Extract updated class embeddings (first num_classes tokens)
        class_queries_updated = attended[:, :self.num_classes, :]  # (B, num_classes, embed_dim)

        # Generate masks: class_queries @ features^T
        # Reshape features back
        feat_updated = attended[:, self.num_classes:, :]  # (B, H*W, embed_dim)

        # Compute attention scores: (B, num_classes, embed_dim) @ (B, embed_dim, H*W)
        # -> (B, num_classes, H*W)
        masks = torch.bmm(class_queries_updated, feat_updated.transpose(1, 2))
        masks = masks / (self.embed_dim ** 0.5)  # Scale by sqrt(embed_dim)

        # Reshape to (B, num_classes, H, W)
        seg_logits = masks.view(B, self.num_classes, H, W)

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


class SegMenterHead_(SegMenterHead):
    """Alias for backward compatibility."""
    pass


@HEADS.register_module()
class SegMenterDAPCNHead(DAPCNHeadMixin, SegMenterHead):
    """SegMenter head with DAPCN support.

    Combines Mask Transformer decode strategy with Dynamic Attention-based
    Prototype Contrastive Network auxiliary losses.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of intermediate channels (embed_dim).
        num_classes (int): Number of semantic classes.
        embed_dim (int): Embedding dimension. Default: 768.
        num_heads (int): Number of attention heads. Default: 12.
        num_transformer_layers (int): Number of transformer layers. Default: 2.
        hidden_dim (int): Hidden dimension of MLP. Default: 3072.
        dropout_ratio (float): Dropout ratio. Default: 0.1.
        conv_cfg (dict|None): Config for convolution. Default: None.
        norm_cfg (dict|None): Config for normalization. Default: None.
        act_cfg (dict): Config for activation. Default: dict(type='GELU').
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
                 embed_dim=768,
                 num_heads=12,
                 num_transformer_layers=2,
                 hidden_dim=3072,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='GELU'),
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
                 ignore_index=255,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        # Initialize SegMenterHead
        SegMenterHead.__init__(
            self,
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            hidden_dim=hidden_dim,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            ignore_index=ignore_index,
            init_cfg=init_cfg)

        # Initialize DAPCN components
        # For SegMenter:
        #   before_fusion: DA operates on inputs[-1] with C=in_channels[-1]=768
        #   after_fusion:  DA operates on projected feature with C=embed_dim=768
        # Note: for SegMenter, in_channels==channels==embed_dim, so both
        # positions yield the same feature_dim. We let the mixin auto-infer.
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

        # Get fused feature for DAPCN (here, we use the projected features)
        x = self._transform_inputs(inputs)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        fused_feature = self.feat_proj(x_flat).view(B, H, W, self.embed_dim)
        fused_feature = fused_feature.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)

        # Compute DAPCN losses
        dapcn_losses = self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature)
        losses.update(dapcn_losses)

        return losses


class SegMenterDAPCNHead_(SegMenterDAPCNHead):
    """Wrapper for registry."""
    pass
