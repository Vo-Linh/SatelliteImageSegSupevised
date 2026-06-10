# PyramidMamba (plain baseline) + ResNeXt101_32x16d — train 500, val 2000.
# Inherits the full baseline train1500 config; only split + work_dir change.

_base_ = ['./pyramidmamba_baseline_openearthmap_train1500_40k_resnext101_32x16d.py']

data = dict(train=dict(split='train_500_fixed.txt'))

work_dir = './work_dirs/openearthmap/pyramidmamba_baseline_train500_resnext101_32x16d'
