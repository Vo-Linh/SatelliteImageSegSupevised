# PyramidMamba-DAPCN + ResNeXt101_32x16d — train on 500 samples, val on 2000.
# Inherits the full train1500 config; only the train split + work_dir change.

_base_ = ['./pyramidmamba_openearthmap_train1500_40k_resnext101_32x16d.py']

data = dict(train=dict(split='train_500_fixed.txt'))

work_dir = './work_dirs/openearthmap/pyramidmamba_dapcn_train500_resnext101_32x16d'
