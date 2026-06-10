# PyramidMamba-Swin-B + DAPCN — train 1000, val 2000.
# Inherits the full train1500 swinb config; only the train split + work_dir change.

_base_ = ['./pyramidmamba_openearthmap_train1500_40k_swinb.py']

data = dict(train=dict(split='train_1000_fixed.txt'))

work_dir = './work_dirs/openearthmap/pyramidmamba_dapcn_train1000_swinb'
