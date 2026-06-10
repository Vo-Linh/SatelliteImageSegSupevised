# PyramidMamba-Swin-B (plain baseline) — train 500, val 2000.
# Inherits the full baseline train1500 swinb config; only split + work_dir change.

_base_ = ['./pyramidmamba_baseline_openearthmap_train1500_40k_swinb.py']

data = dict(train=dict(split='train_500_fixed.txt'))

work_dir = './work_dirs/openearthmap/pyramidmamba_baseline_train500_swinb'
