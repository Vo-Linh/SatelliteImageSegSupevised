"""
PIDNet (PID Controller Inspired Network) Backbone

Implements a three-branch architecture inspired by PID controllers for semantic
segmentation. The P (proportional/detail), I (integral/context), and D (derivative/
boundary) branches work in concert to capture complementary information.

- P branch: High-resolution detail stream
- I branch: Low-resolution context stream
- D branch: Boundary detail stream

Light-weight fusion modules combine branch features for efficient segmentation.

Reference: https://arxiv.org/abs/2206.02066
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES


class BasicBlock(BaseModule):
    """Basic residual block with 3x3 convolutions."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck residual block with 1x1-3x3-1x1 convolutions."""

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class PagFM(BaseModule):
    """Pixel-Attention-Guided Fusion Module.

    A light-weight fusion module that applies channel attention before
    combining two feature maps.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(PagFM, self).__init__(init_cfg)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels, postfix=0)

        # Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Fusion convolution
        self.fuse_conv = build_conv_layer(
            conv_cfg,
            in_channels * 2,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.add_module(self.norm_name, norm)
        self.relu = nn.ReLU(inplace=True)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x1, x2):
        """Fuse two feature maps with pixel attention guidance.

        Args:
            x1: First feature map (typically high-res)
            x2: Second feature map (typically low-res or boundary)

        Returns:
            Fused feature map
        """
        # Channel attention on x2
        att = self.gap(x2)
        att = self.fc(att)
        x2_att = x2 * att

        # Concatenate and fuse
        x_cat = torch.cat([x1, x2_att], dim=1)
        out = self.fuse_conv(x_cat)
        out = self.norm(out)
        out = self.relu(out)

        return out


class BAG(BaseModule):
    """Boundary Attention Guided Module.

    Applies sigmoid attention on boundary features to weight context
    and detail features during final fusion.
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BAG, self).__init__(init_cfg)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, in_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.sigmoid = nn.Sigmoid()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x_detail, x_context, x_boundary):
        """Apply boundary attention guided fusion.

        Args:
            x_detail: Detail features (P branch)
            x_context: Context features (I branch)
            x_boundary: Boundary features (D branch)

        Returns:
            Fused feature map with boundary-guided attention
        """
        # Generate attention gate from boundary features
        gate = self.conv1(x_boundary)
        gate = self.norm1(gate)
        gate = self.relu(gate)
        gate = self.conv2(gate)
        gate = self.norm2(gate)
        gate = self.sigmoid(gate)

        # Apply gating: emphasize boundaries
        x_detail_gated = x_detail * gate
        x_context_gated = x_context * (1 - gate)

        # Combine
        out = x_detail_gated + x_context_gated

        return out


@BACKBONES.register_module()
class PIDNet(BaseModule):
    """PIDNet (PID Controller Inspired Network) for semantic segmentation.

    A three-branch architecture where:
    - P (Proportional) branch: high-resolution detail stream
    - I (Integral) branch: low-resolution context stream
    - D (Derivative) branch: boundary detail stream

    Branches are fused with light-weight attention modules for efficient
    semantic segmentation.

    Args:
        in_channels (int): Number of input channels. Default: 3
        channels (int or list): Base channel widths. If int, creates [c, 2c, 4c, 8c].
            Default: 64
        ppm_channels (int): Channels for context aggregation. Default: 96
        num_blocks (list): Number of blocks in each layer. Default: [2, 2, 2, 2]
        conv_cfg (dict): Config for convolution layer. Default: None
        norm_cfg (dict): Config for normalization. Default: dict(type='BN')
        init_cfg (dict): Config for initialization. Default: None
    """

    def __init__(self,
                 in_channels=3,
                 channels=64,
                 ppm_channels=96,
                 num_blocks=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(PIDNet, self).__init__(init_cfg=init_cfg)

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = in_channels
        self.channels = channels if isinstance(channels, list) else [
            channels, channels * 2, channels * 4, channels * 8
        ]
        self.ppm_channels = ppm_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # Shared stem
        self.norm_stem_name, norm_stem = build_norm_layer(
            norm_cfg, self.channels[0], postfix='stem')
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.channels[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.add_module(self.norm_stem_name, norm_stem)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # P Branch: High-resolution proportional (detail) stream
        self.p_stage1 = self._make_layer(
            BasicBlock,
            self.channels[0],
            self.channels[0],
            num_blocks[0],
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.p_stage2 = self._make_layer(
            BasicBlock,
            self.channels[0],
            self.channels[1],
            num_blocks[1],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.p_stage3 = self._make_layer(
            BasicBlock,
            self.channels[1],
            self.channels[2],
            num_blocks[2],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.p_stage4 = self._make_layer(
            BasicBlock,
            self.channels[2],
            self.channels[3],
            num_blocks[3],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # I Branch: Low-resolution integral (context) stream
        # Use BasicBlock (expansion=1) to keep channel dimensions consistent
        self.i_stage3 = self._make_layer(
            BasicBlock,
            self.channels[2],
            self.channels[2],
            num_blocks[2],
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.i_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.i_stage4 = self._make_layer(
            BasicBlock,
            self.channels[2],
            self.channels[3],
            num_blocks[3],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # D Branch: Boundary detail stream (derived from P)
        self.norm_d_name, norm_d = build_norm_layer(
            norm_cfg, self.channels[0], postfix='d')
        self.d_conv = build_conv_layer(
            conv_cfg,
            self.channels[0],
            self.channels[0],
            kernel_size=3,
            padding=1,
            bias=False)
        self.add_module(self.norm_d_name, norm_d)

        self.d_stage2 = self._make_layer(
            BasicBlock,
            self.channels[0],
            self.channels[1],
            num_blocks[1],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.d_stage3 = self._make_layer(
            BasicBlock,
            self.channels[1],
            self.channels[2],
            num_blocks[2],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.d_stage4 = self._make_layer(
            BasicBlock,
            self.channels[2],
            self.channels[3],
            num_blocks[3],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # Fusion modules
        self.pagfm2 = PagFM(
            self.channels[1],
            self.channels[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.pagfm3 = PagFM(
            self.channels[2],
            self.channels[2],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.pagfm4 = PagFM(
            self.channels[3],
            self.channels[3],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.bag2 = BAG(
            self.channels[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.bag3 = BAG(
            self.channels[2],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.bag4 = BAG(
            self.channels[3],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    @property
    def norm_stem(self):
        return getattr(self, self.norm_stem_name)

    @property
    def norm_d(self):
        return getattr(self, self.norm_d_name)

    def _make_layer(self,
                    block,
                    inplanes,
                    planes,
                    blocks,
                    stride=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN')):
        """Create a residual layer."""
        downsample = None
        if stride != 1 or inplanes != planes:
            norm_name, norm = build_norm_layer(norm_cfg, planes, postfix=0)
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                norm,
            )
            downsample.add_module(norm_name, norm)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes,
                    planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        if isinstance(pretrained, str):
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError(f'pretrained must be a str or None. But received {type(pretrained)}')

    def forward(self, x):
        """Forward pass returning multi-scale features from three branches.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            tuple: Multi-scale fused feature maps from PIDNet branches
        """
        # Shared stem
        stem_out = self.conv1(x)
        stem_out = self.norm_stem(stem_out)
        stem_out = self.relu(stem_out)
        stem_out = self.maxpool(stem_out)

        # ========== P Branch (Proportional - Detail) ==========
        p_s1 = self.p_stage1(stem_out)
        p_s2 = self.p_stage2(p_s1)
        p_s3 = self.p_stage3(p_s2)
        p_s4 = self.p_stage4(p_s3)

        # ========== I Branch (Integral - Context) ==========
        i_s3 = self.i_stage3(p_s3)
        i_pool = self.i_pool(i_s3)
        i_s4 = self.i_stage4(i_pool)

        # ========== D Branch (Derivative - Boundary) ==========
        d_s1 = self.d_conv(stem_out)
        d_s1 = self.norm_d(d_s1)
        d_s1 = self.relu(d_s1)
        d_s2 = self.d_stage2(d_s1)
        d_s3 = self.d_stage3(d_s2)
        d_s4 = self.d_stage4(d_s3)

        # ========== Fusion at Stage 2 ==========
        d_s2_up = nn.functional.interpolate(
            d_s2, size=p_s2.shape[2:], mode='bilinear', align_corners=True)
        fused_s2 = self.pagfm2(p_s2, d_s2_up)
        fused_s2 = self.bag2(fused_s2, p_s2, d_s2_up)

        # ========== Fusion at Stage 3 ==========
        d_s3_up = nn.functional.interpolate(
            d_s3, size=p_s3.shape[2:], mode='bilinear', align_corners=True)
        i_s3_up = nn.functional.interpolate(
            i_s3, size=p_s3.shape[2:], mode='bilinear', align_corners=True)
        fused_s3 = self.pagfm3(p_s3, d_s3_up)
        fused_s3 = self.bag3(fused_s3, i_s3_up, d_s3_up)

        # ========== Fusion at Stage 4 ==========
        d_s4_up = nn.functional.interpolate(
            d_s4, size=p_s4.shape[2:], mode='bilinear', align_corners=True)
        i_s4_up = nn.functional.interpolate(
            i_s4, size=p_s4.shape[2:], mode='bilinear', align_corners=True)
        fused_s4 = self.pagfm4(p_s4, d_s4_up)
        fused_s4 = self.bag4(fused_s4, i_s4_up, d_s4_up)

        return (p_s1, fused_s2, fused_s3, fused_s4)
