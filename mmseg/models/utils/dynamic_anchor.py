# ---------------------------------------------------------------
# DynamicAnchorModule: Dataset-Level Learnable Prototype Discovery
# Reference: docs/DAPCN.md
#
# Revision history:
#   v1 — Per-batch transient prototypes (FPS init each forward pass)
#   v2 — Persistent learnable prototypes (nn.Parameter) with
#         per-batch EM refinement.  Gradients from DAPGLoss flow
#         back through the differentiable EM chain to update the
#         persistent prototypes via the optimiser.
# ---------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_func

from mmcv.runner import BaseModule
from mmseg.models.builder import MODELS


@MODELS.register_module(force=True)
class DynamicAnchorModule(BaseModule):
    """Dataset-level learnable prototype discovery with EM refinement.

    Unlike the original per-batch design where prototypes were
    re-initialised from features every forward pass, this version
    maintains **persistent learnable prototypes** as ``nn.Parameter``.
    They evolve across the entire training trajectory, capturing
    dataset-wide structure rather than batch-specific geometry.

    Each forward pass performs differentiable EM refinement starting
    from the persistent prototypes.  Because both the E-step (softmax
    assignment) and M-step (weighted mean) are differentiable, loss
    gradients propagate back through all EM iterations to the
    ``nn.Parameter``, allowing the optimiser (e.g. AdamW) to update
    them jointly with the rest of the network.

    Computation graph (single EM iteration):

        proto_t  ──► sim = feats @ proto_t.T / tau
                         │
                         ▼
                     assign = softmax(sim)
                         │
                         ▼
                     proto_{t+1} = assign.T @ feats / sizes
                         │
                         ▼
                     normalise(proto_{t+1})

    After ``num_iters`` iterations, the refined prototypes are passed
    to DAPGLoss.  Gradients flow:
        L_dapg ─► proto_refined ─► ... ─► assign_0 ─► self.prototypes

    Key differences from v1 (per-batch):
        - Prototypes are ``nn.Parameter``, not transient tensors.
        - No FPS/random/importance re-init each batch.
        - ``init_method`` controls the one-time parameter initialisation
          (Xavier, Kaiming, or normal).
        - Prototypes accumulate dataset knowledge across iterations.
        - Quality-net and mask-predictor remain for adaptive K and
          efficiency, but operate on the refined (not initial) protos.

    Args:
        feature_dim (int): Dimension of input features.
        max_groups (int): Number of persistent prototypes (K).
            Default: 64.
        min_quality (float): Quality threshold for filtering.
            Default: 0.1.
        num_iters (int): Number of EM refinement iterations per
            forward pass. Default: 3.
        temperature (float): Temperature for soft assignment.
            Default: 0.1.
        init_method (str): One-time prototype initialisation strategy.
            'xavier' — Xavier uniform (default, good for normalised
                       features).
            'kaiming' — Kaiming normal (good for ReLU-style features).
            'normal'  — N(0, 1/sqrt(D)).
            Default: 'xavier'.
        use_quality_gate (bool): Enable quality-net filtering.
            Default: True.
        use_mask_predictor (bool): Enable per-pixel attention mask.
            Default: False (removes the learnable predictor concern
            from the original TODO).
        ema_decay (float): If > 0, apply additional EMA smoothing on
            prototype parameters after EM refinement (provides extra
            stability).  0 disables EMA. Default: 0.0.
        init_cfg (dict, optional): MMEngine init config. Default: None.
    """

    EPS = 1e-6

    def __init__(self,
                 feature_dim,
                 max_groups=64,
                 min_quality=0.1,
                 num_iters=3,
                 temperature=0.1,
                 init_method='xavier',
                 use_quality_gate=True,
                 use_mask_predictor=False,
                 ema_decay=0.0,
                 init_cfg=None):
        super(DynamicAnchorModule, self).__init__(init_cfg)
        self.feature_dim = feature_dim
        self.max_groups = max_groups
        self.min_quality = min_quality
        self.num_iters = num_iters
        self.temperature = temperature
        self.init_method = init_method
        self.use_quality_gate = use_quality_gate
        self.use_mask_predictor = use_mask_predictor
        self.ema_decay = ema_decay

        # ---- Persistent learnable prototypes ----
        self.prototypes = nn.Parameter(
            torch.empty(max_groups, feature_dim))
        self._init_prototypes()

        # ---- Quality estimator (optional) ----
        if use_quality_gate:
            self.quality_net = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )

        # ---- Mask predictor (optional) ----
        if use_mask_predictor:
            self.mask_predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )

        # ---- EMA shadow (non-learnable copy for smoothing) ----
        if ema_decay > 0:
            self.register_buffer(
                '_ema_prototypes',
                torch.zeros(max_groups, feature_dim))
            self.register_buffer(
                '_ema_initialised', torch.tensor(False))

    def _init_prototypes(self):
        """One-time initialisation of the persistent prototype tensor."""
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif self.init_method == 'kaiming':
            # fan_out = feature_dim (each prototype is a row)
            nn.init.kaiming_normal_(
                self.prototypes, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'normal':
            std = 1.0 / math.sqrt(self.feature_dim)
            nn.init.normal_(self.prototypes, mean=0.0, std=std)
        else:
            raise ValueError(
                f"Unknown init_method: '{self.init_method}'. "
                f"Choose from 'xavier', 'kaiming', 'normal'.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features):
        """Forward pass with EM refinement of persistent prototypes.

        Args:
            features (torch.Tensor): Encoder/decoder features of shape
                (B, C, H, W).

        Returns:
            tuple:
                - assign_valid (Tensor): Soft assignments (N, K') where
                  K' <= max_groups after quality filtering.
                - proto_valid (Tensor): Refined prototype vectors (K', C).
                - quality_valid (Tensor): Quality scores (K',).
        """
        B, C, H, W = features.shape
        N = B * H * W
        feats = features.permute(0, 2, 3, 1).reshape(N, C)

        # Normalise input features
        feats_norm = F_func.normalize(feats, dim=1)

        # Start from the persistent prototypes (normalised)
        proto = F_func.normalize(self.prototypes, dim=1)

        # ---- Differentiable EM refinement ----
        assign = None
        for _ in range(self.num_iters):
            # E-step: soft assignment via scaled dot-product
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            assign = torch.softmax(sim, dim=1)       # (N, K)

            # M-step: update prototypes from features
            # Note: ``feats`` (un-normalised) is used so that the
            # M-step captures magnitude information; normalisation
            # is applied afterwards.
            group_sizes = assign.sum(dim=0).clamp(min=self.EPS)
            proto = torch.mm(assign.t(), feats) / group_sizes.unsqueeze(1)
            proto = F_func.normalize(proto, dim=1)

        # Edge case: num_iters == 0 (skip EM, use persistent protos directly)
        if assign is None:
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            assign = torch.softmax(sim, dim=1)

        # ---- Optional EMA smoothing of persistent prototypes ----
        if self.ema_decay > 0 and self.training:
            self._update_ema(proto.detach())

        # ---- Quality-guided filtering (adaptive K) ----
        if self.use_quality_gate:
            group_quality = self.quality_net(proto).squeeze(-1)
            valid_mask = group_quality > self.min_quality

            # Safety: ensure at least 1 prototype survives
            if valid_mask.sum() == 0:
                valid_mask[group_quality.argmax()] = True

            assign_valid = assign[:, valid_mask]
            proto_valid = proto[valid_mask]
            quality_valid = group_quality[valid_mask]
        else:
            # No filtering — all prototypes are valid
            assign_valid = assign
            proto_valid = proto
            quality_valid = torch.ones(
                proto.shape[0], device=proto.device)

        # ---- Optional per-pixel attention mask ----
        if self.use_mask_predictor:
            mask = self.mask_predictor(feats).squeeze(-1)  # (N,)
            assign_valid = assign_valid * mask.unsqueeze(1)
            assign_valid = assign_valid / (
                assign_valid.sum(dim=1, keepdim=True) + self.EPS)

        return assign_valid, proto_valid, quality_valid

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_ema(self, proto_refined):
        """Exponentially smooth the persistent prototypes toward the
        batch-refined values.  This provides stability while still
        allowing gradient-based learning.
        """
        if not self._ema_initialised:
            self._ema_prototypes.copy_(proto_refined)
            self._ema_initialised.fill_(True)
        else:
            self._ema_prototypes.mul_(self.ema_decay).add_(
                proto_refined, alpha=1.0 - self.ema_decay)

        # Soft nudge: blend the persistent parameter toward EMA shadow.
        # This does NOT break the gradient graph because we only modify
        # .data (the parameter's value, not its computational graph).
        blend = 0.01  # small blend factor for stability
        self.prototypes.data.mul_(1.0 - blend).add_(
            self._ema_prototypes, alpha=blend)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def extra_repr(self):
        return (
            f'feature_dim={self.feature_dim}, '
            f'max_groups={self.max_groups}, '
            f'num_iters={self.num_iters}, '
            f'temperature={self.temperature}, '
            f'init_method={self.init_method}, '
            f'quality_gate={self.use_quality_gate}, '
            f'mask_predictor={self.use_mask_predictor}, '
            f'ema_decay={self.ema_decay}'
        )
