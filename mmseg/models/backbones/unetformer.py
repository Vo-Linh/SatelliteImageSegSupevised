"""UNetFormer Backbone Implementation.

UNetFormer is a U-shaped Transformer backbone for semantic segmentation,
combining hierarchical transformer stages with global-local attention
mechanisms and U-Net style skip connections.

All components inherit from mmcv.runner.BaseModule for proper initialization.

Reference: https://arxiv.org/abs/2109.10202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

from ..builder import BACKBONES


class LocalWindowAttention(BaseModule):
    """Local Window Self-Attention.

    Performs multi-head attention within local windows for computational efficiency.

    Args:
        dim (int): Number of input channels.
        window_size (int): Window size for local attention. Default: 7.
        num_heads (int): Number of attention heads. Default: 8.
        attn_drop (float): Attention dropout rate. Default: 0.0.
        proj_drop (float): Projection dropout rate. Default: 0.0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 dim,
                 window_size=7,
                 num_heads=8,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, \
            f'dim {dim} should be divisible by num_heads {num_heads}'

        self.scale = (dim // num_heads) ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        """Forward pass with local window attention.

        Args:
            x (Tensor): Input features of shape (B, N, C).
            H (int): Height of spatial dimension.
            W (int): Width of spatial dimension.

        Returns:
            Tensor: Output features of shape (B, N, C).
        """
        B, N, C = x.shape
        assert N == H * W, f'Input size {N} != {H} * {W}'

        # Reshape to spatial
        x_spatial = x.reshape(B, H, W, C)

        # Apply windowed attention
        window_size = self.window_size
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            x_spatial = F.pad(x_spatial, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x_spatial.shape[1:3]
        x_spatial = x_spatial.reshape(B, Hp // window_size, window_size,
                                      Wp // window_size, window_size, C)
        x_spatial = x_spatial.permute(0, 1, 3, 2, 4, 5).reshape(
            B * (Hp // window_size) * (Wp // window_size), window_size ** 2, C)

        # Multi-head attention
        qkv = self.qkv(x_spatial)
        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output
        x_spatial = (attn @ v).transpose(1, 2).reshape(
            qkv.shape[0], qkv.shape[1], window_size ** 2, C)
        x_spatial = self.proj(x_spatial)
        x_spatial = self.proj_drop(x_spatial)

        # Reshape back
        x_spatial = x_spatial.reshape(
            B, Hp // window_size, Wp // window_size, window_size, window_size, C)
        x_spatial = x_spatial.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)

        # Remove padding
        if pad_h > 0:
            x_spatial = x_spatial[:, :H, :, :]
        if pad_w > 0:
            x_spatial = x_spatial[:, :, :W, :]

        # Reshape back to sequence
        x = x_spatial.reshape(B, H * W, C)
        return x


class GlobalTokenAttention(BaseModule):
    """Global Token Self-Attention.

    Uses learnable global tokens that attend to all spatial locations
    for global context modeling.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8.
        attn_drop (float): Attention dropout rate. Default: 0.0.
        proj_drop (float): Projection dropout rate. Default: 0.0.
        num_tokens (int): Number of global tokens. Default: 1.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 num_tokens=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dim = dim
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        assert dim % num_heads == 0, \
            f'dim {dim} should be divisible by num_heads {num_heads}'

        self.scale = (dim // num_heads) ** -0.5

        # Global tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass with global token attention.

        Args:
            x (Tensor): Input features of shape (B, N, C).

        Returns:
            Tensor: Output features of shape (B, N, C).
        """
        B, N, C = x.shape

        # Concatenate global tokens
        global_tokens = self.global_tokens.expand(B, -1, -1)
        x_with_tokens = torch.cat([global_tokens, x], dim=1)

        # Project to QKV
        qkv = self.qkv(x_with_tokens)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_heads,
                          C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output
        x_with_tokens = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_with_tokens = self.proj(x_with_tokens)
        x_with_tokens = self.proj_drop(x_with_tokens)

        # Return only spatial features (remove global tokens)
        return x_with_tokens[:, self.num_tokens:, :]


class TransformerBlock(BaseModule):
    """Transformer Block with Local-Global Attention.

    Combines local window attention and global token attention with
    feed-forward for hybrid global-local modeling.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        window_size (int): Local window size. Default: 7.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 window_size=7,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Local-window attention
        self.local_attn = LocalWindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)

        # Global token attention
        self.global_attn = GlobalTokenAttention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)

        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x, H, W):
        """Forward pass.

        Args:
            x (Tensor): Input features of shape (B, N, C).
            H (int): Height.
            W (int): Width.

        Returns:
            Tensor: Output features of shape (B, N, C).
        """
        # Local window attention with residual
        x_norm = self.norm1(x)
        local_out = self.local_attn(x_norm, H, W)
        x = x + local_out

        # Global token attention with residual
        x_norm = self.norm2(x)
        global_out = self.global_attn(x_norm)
        x = x + global_out

        # Feed-forward with residual
        x_norm = self.norm3(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class PatchMerging(BaseModule):
    """Patch Merging for downsampling.

    Merges adjacent patches and projects to higher dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reduction = nn.Linear(in_channels * 4, out_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels * 4)

    def forward(self, x, H, W):
        """Forward pass.

        Args:
            x (Tensor): Input features of shape (B, H*W, C).
            H (int): Height.
            W (int): Width.

        Returns:
            Tensor: Output features of shape (B, H/2*W/2, out_channels).
        """
        B, N, C = x.shape
        assert H * W == N, f'Input size {N} != {H} * {W}'

        x = x.reshape(B, H, W, C)

        # Merge patches: (B, H, W, C) -> (B, H/2, W/2, 4*C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.reshape(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class EncoderStage(BaseModule):
    """Encoder Stage with Patch Merging and Transformer Blocks.

    Args:
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP ratio. Default: 4.0.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        window_size (int): Local window size. Default: 7.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 embed_dim,
                 depth,
                 num_heads,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 window_size=7,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # Patch merging layer
        self.patch_merge = PatchMerging(
            in_channels, embed_dim, norm_cfg=norm_cfg)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                window_size=window_size)
            for _ in range(depth)
        ])

    def forward(self, x, H, W):
        """Forward pass.

        Args:
            x (Tensor): Input features of shape (B, N, C).
            H (int): Height.
            W (int): Width.

        Returns:
            tuple: (output features, new height, new width).
        """
        # Patch merging
        x = self.patch_merge(x, H, W)
        H, W = H // 2, W // 2

        # Transformer blocks
        for block in self.blocks:
            x = block(x, H, W)

        return x, H, W


@BACKBONES.register_module()
class UNetFormer(BaseModule):
    """UNetFormer Backbone.

    U-shaped Transformer backbone for semantic segmentation with hierarchical
    encoder stages, global-local attention, and decoder path with skip connections.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list): List of embedding dimensions for each stage.
            Default: [64, 128, 256, 512].
        depths (list): List of depths (number of transformer blocks) for each stage.
            Default: [2, 2, 6, 2].
        num_heads (list): List of number of attention heads for each stage.
            Default: [2, 4, 8, 16].
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): MLP ratio. Default: 4.0.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN'). Note: we use nn.LayerNorm directly.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> cfg = dict(
        ...     type='UNetFormer',
        ...     in_channels=3,
        ...     embed_dims=[64, 128, 256, 512],
        ...     depths=[2, 2, 6, 2],
        ...     num_heads=[2, 4, 8, 16])
        >>> backbone = UNetFormer(**cfg)
        >>> x = torch.randn(1, 3, 512, 512)
        >>> output = backbone(x)
        >>> len(output)  # 4 + 3 = 7 feature maps (encoder + decoder)
        7
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=7,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_stages = len(embed_dims)

        # Initial patch embedding (4x downsampling)
        self.patch_embed = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                embed_dims[0],
                kernel_size=4,
                stride=4),
            nn.LayerNorm(embed_dims[0]),
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            if i == 0:
                # First stage: no patch merging, just transformer blocks
                stage = nn.ModuleList([
                    TransformerBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        window_size=window_size)
                    for _ in range(depths[i])
                ])
            else:
                # Subsequent stages with patch merging
                stage = EncoderStage(
                    in_channels=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    window_size=window_size,
                    norm_cfg=norm_cfg)
            self.encoder_stages.append(stage)

        # Decoder stages with skip connections
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages - 2, -1, -1):
            # Linear projection for skip connection fusion
            self.decoder_stages.append(
                nn.Linear(embed_dims[i + 1], embed_dims[i]))

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            tuple: Multi-scale features from encoder and decoder paths.
                Each feature is of shape (B, embed_dims[i], H_i, W_i).
        """
        B, C, H, W = x.shape

        # Initial patch embedding (4x downsampling)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, HW/16, embed_dims[0])
        cur_h, cur_w = H // 4, W // 4

        # Encoder forward pass
        encoder_outputs = []

        # First stage (no patch merging)
        for block in self.encoder_stages[0]:
            x = block(x, cur_h, cur_w)
        encoder_outputs.append(x)

        # Subsequent stages (with patch merging)
        for stage_idx in range(1, self.num_stages):
            stage = self.encoder_stages[stage_idx]
            x, cur_h, cur_w = stage(x, cur_h, cur_w)
            encoder_outputs.append(x)

        # Decoder path with skip connections
        decoder_outputs = [encoder_outputs[-1]]
        for i in range(len(self.decoder_stages)):
            # Get skip connection from encoder
            skip_feature = encoder_outputs[-(i + 2)]

            # Project decoder features
            x = self.decoder_stages[i](decoder_outputs[-1])

            # Fuse with skip connection
            x = x + skip_feature
            decoder_outputs.append(x)

        # Return multi-scale features (encoder + decoder)
        all_outputs = encoder_outputs + decoder_outputs[1:]
        return tuple(all_outputs)
