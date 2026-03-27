"""SegMenter Backbone Implementation.

SegMenter is a pure Vision Transformer-based backbone for semantic segmentation.
It uses patch embeddings, learnable positional encodings, and transformer encoder
blocks to process image patches and return multi-scale features.

All components inherit from mmcv.runner.BaseModule for proper initialization.

Reference: https://arxiv.org/abs/2105.05633
"""

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule

from ..builder import BACKBONES


class DropPath(nn.Module):
    """Drop Path (Stochastic Depth) as described in `Deep Networks with
    Stochastic Depth <https://arxiv.org/abs/1603.09382>`_.

    Args:
        drop_prob (float): Drop path probability. Default: 0.0.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward pass."""
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(
            torch.full(shape, keep_prob, device=x.device))
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchEmbedding(BaseModule):
    """Image to Patch Embedding.

    Uses a convolutional layer for patch projection. All components inherit
    from BaseModule for proper initialization support.

    Args:
        img_size (int): Input image size. Default: 512.
        patch_size (int): Patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 768.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 conv_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding via build_conv_layer for consistency
        self.proj = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with truncated normal."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Patch embeddings with class token of shape (B, num_patches+1, embed_dim).
        """
        # Project image to patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        return x


class TransformerEncoderBlock(BaseModule):
    """Transformer Encoder Block.

    Standard transformer block with LayerNorm -> Attention -> LayerNorm -> MLP.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        act_layer (nn.Module): Activation layer. Default: nn.GELU.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 embed_dim,
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 act_layer=nn.GELU,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, \
            f'embed_dim {embed_dim} should be divisible by num_heads {num_heads}'

        # Layer normalization (nn.LayerNorm directly, as in MiT reference)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop_rate,
            batch_first=True)

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, N, embed_dim).

        Returns:
            Tensor: Output tensor of shape (B, N, embed_dim).
        """
        # Self-attention block with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path(attn_out)

        # Feed-forward block with residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.drop_path(mlp_out)

        return x


@BACKBONES.register_module()
class SegMenterBackbone(BaseModule):
    """SegMenter Backbone (Vision Transformer).

    Pure ViT-based backbone for semantic segmentation with patch embeddings,
    positional encodings, and transformer encoder blocks. Extracts features
    from intermediate transformer blocks.

    Args:
        img_size (int): Input image size. Default: 512.
        patch_size (int): Patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        out_indices (tuple): Indices of intermediate layers to output features.
            Default: (2, 5, 8, 11).
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN'). Note: we use nn.LayerNorm directly.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> cfg = dict(
        ...     type='SegMenterBackbone',
        ...     img_size=512,
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12,
        ...     out_indices=(2, 5, 8, 11))
        >>> backbone = SegMenterBackbone(**cfg)
        >>> x = torch.randn(1, 3, 512, 512)
        >>> output = backbone(x)
        >>> len(output)  # 4 feature maps
        4
        >>> output[0].shape
        torch.Size([1, 768, 32, 32])
    """

    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 out_indices=(2, 5, 8, 11),
                 conv_cfg=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.out_indices = out_indices
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            conv_cfg=conv_cfg)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate)
            for _ in range(depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def init_weights(self):
        """Initialize weights."""
        # Embeddings are initialized in PatchEmbedding._init_embeddings()
        # Linear layers use default PyTorch initialization or init_cfg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            tuple: Multi-scale features from intermediate layers.
                Each output is of shape (B, embed_dim, H/patch_size, W/patch_size).
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches+1, embed_dim)

        # Extract features from intermediate layers
        outputs = []

        # Forward through encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)

            # Collect features at specified indices
            if i in self.out_indices:
                # Normalize
                feat = self.norm(x)

                # Remove class token and reshape to spatial
                feat = feat[:, 1:, :]  # (B, num_patches, embed_dim)
                feat = feat.transpose(1, 2)  # (B, embed_dim, num_patches)

                # Reshape to spatial format (B, embed_dim, H/patch_size, W/patch_size)
                H = W = int(self.num_patches ** 0.5)
                feat = feat.reshape(B, -1, H, W)
                outputs.append(feat)

        return tuple(outputs)
