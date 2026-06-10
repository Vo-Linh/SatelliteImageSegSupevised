# E2 — E1 (weighted CE + DAPG off) + non-aggressive OHEM hard-pixel mining.
#
# OHEMPixelSampler emits a per-pixel weight that MULTIPLIES the class_weight
# inside F.cross_entropy, so the two compose. Because OHEM already amplifies
# hard rare-class / boundary pixels, the class_weight is SOFTENED vs E1 to
# avoid double-counting the same pixels (which would amplify Bareland variance).
#
#   thresh=0.7    -> only pixels whose predicted prob on their GT class < 0.7
#                    are forced to count (adapts to training progress).
#   min_kept=131072 (~50% of 512*512) -> large pool; does NOT starve easy
#                    frequent-class pixels. Raise it if frequent-class IoU drops.
#
# Run AFTER E1 as a second step. Report at peak ckpt with TTA.
# If c2/c5/c8 regress > ~1.5 IoU or c1 variance climbs, drop OHEM, keep E1.

_base_ = ['./oem_woxbd_E1_clsweight_nodapg.py']

work_dir = './work_dirs/openearthmap/oem_woxbd_E2_ohem'

model = dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=131072),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 2.0, 0.7, 0.85, 1.3, 0.75, 1.2, 0.9, 0.9]),
    ))
