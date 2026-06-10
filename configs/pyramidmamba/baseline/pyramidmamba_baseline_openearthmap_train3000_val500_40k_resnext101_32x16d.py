# PyramidMamba (plain baseline) + ResNeXt101_32x16d on the OpenEarthMap wo_xBD split.
#   train: train_woxbd_3000.txt (3000)
#   val:   val_woxbd_500.txt    (500)
# Inherits the full baseline train1500 config; only the splits + work_dir change.

_base_ = ['./pyramidmamba_baseline_openearthmap_train1500_40k_resnext101_32x16d.py']

data = dict(
    samples_per_gpu=8,
    train=dict(split='train_woxbd_3000.txt'),
    val=dict(split='val_woxbd_500.txt'),
)

work_dir = './work_dirs/openearthmap/pyramidmamba_baseline_woxbd_train3000_val500_resnext101_32x16d'
