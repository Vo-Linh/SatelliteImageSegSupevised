"""
DDRNet (Deep Dual-Resolution Networks) Backbone

Implements a dual-branch architecture with high-resolution and low-resolution
pathways for semantic segmentation. The high-resolution branch preserves spatial
detail while the low-resolution branch captures global context. Bilateral fusion
modules combine features across branches.

Reference: https://arxiv.org/abs/2101.06085
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


class DAPPM(BaseModule):
    """Deep Aggregation Pyramid Pooling Module.

    Aggregates features at multiple scales using pyramid pooling.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(DAPPM, self).__init__(init_cfg)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, out_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, out_channels, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(norm_cfg, out_channels, postfix=4)

        self.relu = nn.ReLU(inplace=True)

        # Identity branch: project input to out_channels
        self.norm0_name, norm0 = build_norm_layer(
            norm_cfg, out_channels, postfix=0)
        self.branch0 = build_conv_layer(
            conv_cfg, in_channels, out_channels, 1, bias=False)
        self.add_module(self.norm0_name, norm0)

        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            build_conv_layer(conv_cfg, in_channels, out_channels, 1, bias=False),
        )
        self.add_module(self.norm1_name, norm1)

        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            build_conv_layer(conv_cfg, in_channels, out_channels, 1, bias=False),
        )
        self.add_module(self.norm2_name, norm2)

        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            build_conv_layer(conv_cfg, in_channels, out_channels, 1, bias=False),
        )
        self.add_module(self.norm3_name, norm3)

        # Fuse 4 branches (identity + 3 pooling), each with out_channels
        self.conv_out = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                out_channels * 4,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
        )
        self.add_module(self.norm4_name, norm4)

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        return getattr(self, self.norm4_name)

    def forward(self, x):
        h, w = x.shape[2:]

        # Identity branch: project to out_channels
        branch0 = self.branch0(x)
        branch0 = self.norm0(branch0)
        branch0 = self.relu(branch0)

        branch1 = self.branch1[0](x)
        branch1 = self.branch1[1](branch1)
        branch1 = self.norm1(branch1)
        branch1 = self.relu(branch1)
        branch1 = nn.functional.interpolate(
            branch1, size=(h, w), mode='bilinear', align_corners=True)

        branch2 = self.branch2[0](x)
        branch2 = self.branch2[1](branch2)
        branch2 = self.norm2(branch2)
        branch2 = self.relu(branch2)
        branch2 = nn.functional.interpolate(
            branch2, size=(h, w), mode='bilinear', align_corners=True)

        branch3 = self.branch3[0](x)
        branch3 = self.branch3[1](branch3)
        branch3 = self.norm3(branch3)
        branch3 = self.relu(branch3)
        branch3 = nn.functional.interpolate(
            branch3, size=(h, w), mode='bilinear', align_corners=True)

        # All 4 branches have out_channels dimensions
        out = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        out = self.conv_out[0](out)
        out = self.norm4(out)
        out = self.relu(out)

        return out


class BilateralFusion(BaseModule):
    """Bilateral fusion module for combining high-res and low-res branches."""

    def __init__(self,
                 high_channels,
                 low_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BilateralFusion, self).__init__(init_cfg)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, out_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channels, postfix=2)

        # High-res path
        self.high_conv = build_conv_layer(
            conv_cfg, high_channels, out_channels, 1, bias=False)
        self.add_module(self.norm1_name, norm1)

        # Low-res path (upsample)
        self.low_conv = build_conv_layer(
            conv_cfg, low_channels, out_channels, 1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, high_x, low_x):
        """Fuse high-resolution and low-resolution features.

        Args:
            high_x: high-resolution feature map
            low_x: low-resolution feature map

        Returns:
            Fused feature map
        """
        h, w = high_x.shape[2:]

        # Process high-res branch
        high_out = self.high_conv(high_x)
        high_out = self.norm1(high_out)

        # Process low-res branch with upsampling
        low_out = self.low_conv(low_x)
        low_out = self.norm2(low_out)
        low_out = nn.functional.interpolate(
            low_out, size=(h, w), mode='bilinear', align_corners=True)

        # Fuse
        out = high_out + low_out
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class DDRNet(BaseModule):
    """DDRNet (Deep Dual-Resolution Networks) for semantic segmentation.

    A dual-branch architecture combining high-resolution and low-resolution
    pathways with bilateral fusion for efficient semantic segmentation.

    Args:
        in_channels (int): Number of input channels. Default: 3
        channels (int or list): Base channel widths. If int, creates [c, 2c, 4c, 8c].
            Default: 64
        ppm_channels (int): Channels for DAPPM module. Default: 128
        num_blocks (list): Number of blocks in each layer. Default: [2, 2, 2, 2]
        conv_cfg (dict): Config for convolution layer. Default: None
        norm_cfg (dict): Config for normalization. Default: dict(type='BN')
        init_cfg (dict): Config for initialization. Default: None
    """

    def __init__(self,
                 in_channels=3,
                 channels=64,
                 ppm_channels=128,
                 num_blocks=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(DDRNet, self).__init__(init_cfg=init_cfg)

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = in_channels
        self.channels = channels if isinstance(channels, list) else [
            channels, channels * 2, channels * 4, channels * 8
        ]
        self.ppm_channels = ppm_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # Stem: stride-2 convolution with maxpool
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

        # High-resolution branch stages
        self.hr_stage1 = self._make_layer(
            BasicBlock,
            self.channels[0],
            self.channels[0],
            num_blocks[0],
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.hr_stage2 = self._make_layer(
            BasicBlock,
            self.channels[0],
            self.channels[1],
            num_blocks[1],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.hr_stage3 = self._make_layer(
            BasicBlock,
            self.channels[1],
            self.channels[2],
            num_blocks[2],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.hr_stage4 = self._make_layer(
            BasicBlock,
            self.channels[2],
            self.channels[3],
            num_blocks[3],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # Low-resolution branch (parallel pathway from stage3)
        # Use BasicBlock (expansion=1) to keep channel dimensions consistent
        self.lr_stage3 = self._make_layer(
            BasicBlock,
            self.channels[2],
            self.channels[2],
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.dappm = DAPPM(
            self.channels[2],
            ppm_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.lr_stage4 = self._make_layer(
            BasicBlock,
            ppm_channels,
            self.channels[3],
            num_blocks[3],
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # Bilateral fusion modules
        self.fusion3 = BilateralFusion(
            self.channels[2],
            ppm_channels,
            self.channels[2],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.fusion4 = BilateralFusion(
            self.channels[3],
            self.channels[3],
            self.channels[3],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    @property
    def norm_stem(self):
        return getattr(self, self.norm_stem_name)

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
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            tuple: Multi-scale feature maps
        """
        # Stem
        x = self.conv1(x)
        x = self.norm_stem(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # High-resolution branch
        hr_s1 = self.hr_stage1(x)
        hr_s2 = self.hr_stage2(hr_s1)
        hr_s3 = self.hr_stage3(hr_s2)
        hr_s4 = self.hr_stage4(hr_s3)

        # Low-resolution branch (parallel from stage3)
        lr_s3 = self.lr_stage3(hr_s3)
        lr_s3 = self.dappm(lr_s3)
        lr_s4 = self.lr_stage4(lr_s3)

        # Bilateral fusion
        fused_s3 = self.fusion3(hr_s3, lr_s3)
        fused_s4 = self.fusion4(hr_s4, lr_s4)

        return (hr_s1, hr_s2, fused_s3, fused_s4)
