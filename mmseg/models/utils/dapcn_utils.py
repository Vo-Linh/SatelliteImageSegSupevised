# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# DAPCN: Dynamic Attention-based Prototype Contrastive Network
# Reference: docs/DAPCN.md
# Modifications: Adapted for UDA
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_boundary_map(logits, mode='sobel'):
    """Extract boundary map from segmentation logits using gradient operators.

    Args:
        logits (torch.Tensor): Segmentation logits of shape (N, C, H, W)
        mode (str): Boundary extraction mode ('sobel', 'laplacian', 'diff')

    Returns:
        torch.Tensor: Boundary map of shape (N, 1, H, W), values in [0, 1]
    """
    if mode == 'sobel':
        # Compute gradients using Sobel filters
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=logits.dtype, device=logits.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=logits.dtype, device=logits.device)

        # Add channel and batch dimensions
        num_channels = logits.shape[1]
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1)

        # Apply to each class separately (depthwise convolution)
        grad_x = F.conv2d(logits, sobel_x, padding=1, groups=num_channels)
        grad_y = F.conv2d(logits, sobel_y, padding=1, groups=num_channels)

        # Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Sum over classes and normalize
        boundary = torch.sum(grad_mag, dim=1, keepdim=True)
        boundary = boundary / (boundary.amax(dim=(2, 3), keepdim=True) + 1e-8)
        boundary = torch.clamp(boundary, 0, 1)

    elif mode == 'diff':
        # Simple neighbor difference
        # Pad logits
        logits_pad = F.pad(logits, (1, 1, 1, 1), mode='replicate')

        # Compute differences in 4 directions
        diff_left = torch.abs(logits_pad[..., :-2, 1:-1] - logits_pad[..., 2:, 1:-1])
        diff_top = torch.abs(logits_pad[..., 1:-1, :-2] - logits_pad[..., 1:-1, 2:])

        # Sum over classes and directions
        boundary = torch.sum(diff_left + diff_top, dim=1, keepdim=True)
        boundary = torch.clamp(boundary / (boundary.max() + 1e-8), 0, 1)

    elif mode == 'laplacian':
        # Laplacian filter
        num_channels = logits.shape[1]
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=logits.dtype, device=logits.device)
        laplacian = laplacian.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1)

        # Apply to each class
        lap = F.conv2d(logits, laplacian, padding=1, groups=num_channels)

        # Compute absolute values, sum over classes, normalize
        boundary = torch.sum(torch.abs(lap), dim=1, keepdim=True)
        boundary = torch.clamp(boundary / (boundary.max() + 1e-8), 0, 1)

    else:
        raise ValueError(f"Unknown boundary extraction mode: {mode}")

    return boundary


def compute_boundary_gt(seg_label, ignore_index=255):
    """Compute boundary ground truth from segmentation labels.

    Args:
        seg_label (torch.Tensor): Segmentation labels of shape (N, H, W) or (N, 1, H, W)
        ignore_index (int): Ignore index for label

    Returns:
        torch.Tensor: Boundary map of shape (N, 1, H, W), values in {0, 1}
    """
    if seg_label.dim() == 4:
        seg_label = seg_label.squeeze(1)

    N, H, W = seg_label.shape
    seg_label = seg_label.unsqueeze(1)

    label_pad = torch.cat([
        seg_label[:, :, :1, :],
        seg_label,
        seg_label[:, :, -1:, :],
    ], dim=2)
    label_pad = torch.cat([
        label_pad[:, :, :, :1],
        label_pad,
        label_pad[:, :, :, -1:],
    ], dim=3)

    diff_h = (label_pad[:, :, 1:-1, :-2] != label_pad[:, :, 1:-1, 2:])
    diff_v = (label_pad[:, :, :-2, 1:-1] != label_pad[:, :, 2:, 1:-1])

    mask_ignore = (seg_label != ignore_index).float()

    boundary = torch.logical_or(diff_h, diff_v).float()
    boundary = boundary * mask_ignore

    return boundary


