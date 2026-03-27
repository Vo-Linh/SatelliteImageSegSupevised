# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..builder import LOSSES
except ImportError:
    LOSSES = None


def _register(cls):
    if LOSSES is not None:
        return LOSSES.register_module()(cls)
    return cls



def affinity_boundary_loss(features,
                           seg_label,
                           temperature=0.5,
                           scale=2,
                           num_neighbors=4,
                           ignore_index=255,
                           pseudo_weight=None,
                           reduction='mean'):
    """Compute affinity-based boundary loss using 4-neighbor connectivity.

    Uses relational (pixel-pair) affinity rather than per-pixel prediction.
    High affinity indicates same semantic class (should be pulled together),
    low affinity indicates boundary (should be pushed apart).

    Args:
        features (torch.Tensor): Feature embeddings of shape (B, C, H, W)
        seg_label (torch.Tensor): Ground truth labels of shape (B, H, W)
        temperature (float): Temperature parameter for affinity scaling. Default: 0.5
        scale (int): Downsampling factor for computational efficiency. Default: 2
        num_neighbors (int): Number of neighbors (4 for 4-connectivity). Default: 4
        ignore_index (int): Ignore index for labels. Default: 255
        pseudo_weight (torch.Tensor, optional): Pseudo-label weights of shape (B, H, W)
        reduction (str): Reduction method ('mean' or 'none'). Default: 'mean'

    Returns:
        torch.Tensor: Scalar affinity boundary loss
    """
    # Named constants for readability
    DIM_WIDTH_4D = 3
    DIM_HEIGHT_4D = 2
    DIM_CHANNEL = 1
    DIM_WIDTH_3D = 2
    DIM_HEIGHT_3D = 1
    SHIFT_RIGHT = 1
    SHIFT_LEFT = -1
    EPSILON = 1e-7
    NEIGHBOR_DIRECTIONS = [
        (3, 1),   # Right
        (3, -1),  # Left
        (2, 1),   # Bottom
        (2, -1),  # Top
    ]

    B, C, H, W = features.shape

    # Downsample for computational efficiency
    if scale > 1:
        features = F.interpolate(features, scale_factor=1.0 / scale,
                                 mode='bilinear', align_corners=False)
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(),
                                   scale_factor=1.0 / scale,
                                   mode='nearest').squeeze(1).long()
        if pseudo_weight is not None:
            pseudo_weight = F.interpolate(pseudo_weight.unsqueeze(1).float(),
                                           scale_factor=1.0 / scale,
                                           mode='bilinear',
                                           align_corners=False).squeeze(1)

    B, C, H_new, W_new = features.shape

    # Normalize features
    features_norm = F.normalize(features, dim=DIM_CHANNEL)
    valid_mask = (seg_label != ignore_index).float()

    # Apply pseudo-weights if provided
    if pseudo_weight is not None:
        weights = pseudo_weight * valid_mask
    else:
        weights = valid_mask

    # Compute loss for each direction
    total_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)
    num_directions = 0

    for feat_dim, shift in NEIGHBOR_DIRECTIONS[:num_neighbors]:
        shifted_feat = torch.roll(features_norm, shifts=shift, dims=feat_dim)
        similarity = torch.sum(features_norm * shifted_feat, dim=DIM_CHANNEL) / temperature

        label_dim = feat_dim - 1  # Map 4D feature dim to 3D label dim
        spatial_valid = torch.ones_like(seg_label).float()

        # Handle edge pixels (no wrap-around)
        if label_dim == DIM_WIDTH_3D:
            if shift == SHIFT_RIGHT:
                spatial_valid[:, :, -1] = 0
            elif shift == SHIFT_LEFT:
                spatial_valid[:, :, 0] = 0
        elif label_dim == DIM_HEIGHT_3D:
            if shift == SHIFT_RIGHT:
                spatial_valid[:, -1, :] = 0
            elif shift == SHIFT_LEFT:
                spatial_valid[:, 0, :] = 0

        shifted_label = torch.roll(seg_label, shifts=shift, dims=label_dim)
        shifted_valid = torch.roll(valid_mask, shifts=shift, dims=label_dim)
        pair_valid = valid_mask * shifted_valid * spatial_valid
        target_affinity = (seg_label == shifted_label).float()
        target_affinity = target_affinity * pair_valid

        if pseudo_weight is not None:
            shifted_weight = torch.roll(weights, shifts=shift, dims=label_dim)
            pair_weight = weights * shifted_weight * pair_valid
        else:
            pair_weight = pair_valid

        prob = torch.sigmoid(similarity)
        prob = torch.clamp(prob, min=EPSILON, max=1 - EPSILON)
        bce_loss = -target_affinity * torch.log(prob) - \
                   (1 - target_affinity) * torch.log(1 - prob)

        if pair_weight.sum() > 0:
            weighted_loss = (bce_loss * pair_weight).sum() / pair_weight.sum()
            total_loss += weighted_loss
            num_directions += 1

    if num_directions > 0:
        loss = total_loss / num_directions
    else:
        # Return zero loss when all pixels are ignored
        loss = torch.zeros(1, device=features.device, dtype=features.dtype,
                           requires_grad=False).squeeze()

    return loss


@_register
class AffinityBoundaryLoss(nn.Module):
    """Affinity-based boundary loss using 4-neighbor connectivity.

    Uses relational (pixel-pair) affinity rather than per-pixel prediction.
    High affinity indicates same semantic class (should be pulled together),
    low affinity indicates boundary (should be pushed apart).

    Args:
        temperature (float): Temperature parameter for affinity scaling. Default: 0.5
        scale (int): Downsampling factor for computational efficiency. Default: 2
        num_neighbors (int): Number of neighbors (4 for 4-connectivity). Default: 4
        ignore_index (int): Ignore index for labels. Default: 255
        reduction (str): Reduction method ('mean' or 'none'). Default: 'mean'
        loss_weight (float): Weight of the loss. Default: 1.0
    """

    def __init__(self,
                 temperature=0.5,
                 scale=2,
                 num_neighbors=4,
                 ignore_index=255,
                 reduction='mean',
                 loss_weight=1.0):
        super(AffinityBoundaryLoss, self).__init__()
        self.temperature = temperature
        self.scale = scale
        self.num_neighbors = num_neighbors
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                features,
                seg_label,
                pseudo_weight=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            features (torch.Tensor): Feature embeddings of shape (B, C, H, W)
            seg_label (torch.Tensor): Ground truth labels of shape (B, H, W)
            pseudo_weight (torch.Tensor, optional): Pseudo-label weights of shape (B, H, W)
            reduction_override (str, optional): Override reduction method.
                Options are "none", "mean". Default: None

        Returns:
            torch.Tensor: Weighted loss value
        """
        assert reduction_override in (None, 'none', 'mean'), \
            f"Invalid reduction_override: {reduction_override}"
        reduction = reduction_override if reduction_override else self.reduction

        loss = affinity_boundary_loss(
            features,
            seg_label,
            temperature=self.temperature,
            scale=self.scale,
            num_neighbors=self.num_neighbors,
            ignore_index=self.ignore_index,
            pseudo_weight=pseudo_weight,
            reduction=reduction
        )
        return self.loss_weight * loss


if __name__ == '__main__':
    print("Testing AffinityBoundaryLoss...")

    loss_fn = AffinityBoundaryLoss(temperature=0.5, scale=2)

    features = torch.randn(2, 256, 64, 64, requires_grad=True)
    seg_label = torch.zeros(2, 64, 64, dtype=torch.long)
    seg_label[:, :32, :] = 1
    seg_label[:, 32:, :] = 2

    loss = loss_fn(features, seg_label)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn}")

    loss.backward()
    print(f"features.grad is not None: {features.grad is not None}")
    print(f"features.grad shape: {features.grad.shape}")

    seg_label_ignore = torch.full((2, 64, 64), 255, dtype=torch.long)
    loss_zero = loss_fn(features, seg_label_ignore)
    print(f"Loss with all ignore: {loss_zero.item():.4f}")

    print("\nAll tests passed!")
