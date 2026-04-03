"""KNet Backbone Implementation.

KNet (Kernel Update Network) combines a standard backbone with kernel-based
dynamic feature refinement. K-Net's key insight: the backbone is a STANDARD
backbone (ResNet or Swin), and kernel-based processing happens in the HEAD.
This backbone is a thin wrapper that delegates to the base backbone.

Reference: https://github.com/open-mmlab/mmseg/tree/main/mmseg/models/backbones
"""

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES, build_backbone


@BACKBONES.register_module()
class KNetBackbone(BaseModule):
    """KNet Backbone.

    Wrapper backbone that uses a standard backbone (ResNet, Swin, etc.)
    as the feature extractor. K-Net's kernel update mechanism is handled
    in the decode head, not the backbone.

    The backbone simply delegates forward pass to the base backbone and
    returns its multi-scale features.

    Args:
        base_backbone (dict): Config dict for base backbone.
            Example: dict(type='ResNetV1c', depth=50, num_stages=4,
            out_indices=(0, 1, 2, 3), dilations=(1, 1, 2, 4), ...)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.

    Example:
        >>> import torch
        >>> cfg = dict(
        ...     type='KNetBackbone',
        ...     base_backbone=dict(
        ...         type='ResNetV1c',
        ...         depth=50,
        ...         num_stages=4,
        ...         out_indices=(0, 1, 2, 3)))
        >>> backbone = KNetBackbone(**cfg)
        >>> x = torch.randn(1, 3, 512, 512)
        >>> output = backbone(x)
        >>> len(output)  # 4 scales
        4
    """

    def __init__(self,
                 base_backbone,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.base_backbone = base_backbone
        if pretrained is not None:
            base_backbone['pretrained'] = pretrained
        self.backbone = build_backbone(base_backbone)

    def init_weights(self):
        """Initialize weights."""
        self.backbone.init_weights()

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            tuple: Multi-scale features from base backbone.
                Each feature map is of shape (B, C_i, H_i, W_i).
        """
        return self.backbone(x)
