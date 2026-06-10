# E3 — E1 (weighted CE + DAPG off) + Lovasz-Softmax auxiliary term.
#
# Lovasz-Softmax is a direct (sub-)differentiable surrogate of mean-IoU, the
# exact eval metric, and is the strongest known lever for rare/volatile-class
# IoU (Bareland/Road). Total objective becomes:
#       L = CE_weighted  +  0.15 * boundary  +  0.5 * Lovasz
#
# Wiring: this fork's BaseDecodeHead.losses() builds a SINGLE loss_decode, so
# Lovasz is injected as a parallel gradient-bearing term inside
# DAPCNHeadMixin.dapcn_forward_train (key 'loss_lovasz'), computed on the
# FULL-resolution (512) upsampled logits so its IoU surrogate matches the CE /
# eval resolution — NOT on the raw H/4 logits.
#
# This is the designated "closer" to clear 0.75 if E1/E2 plateau ~0.745.
# Lovasz overlaps with class-weighting gains — do NOT expect additive stacking.
# If it regresses, drop lovasz_lambda 0.5 -> 0.25. Report at peak ckpt + TTA.

_base_ = ['./oem_woxbd_E1_clsweight_nodapg.py']

work_dir = './work_dirs/openearthmap/oem_woxbd_E3_lovasz'

model = dict(
    decode_head=dict(
        lovasz_lambda=0.5,
        lovasz_loss=dict(
            type='LovaszLoss',
            loss_type='multi_class',
            classes='present',
            per_image=False),
    ))
