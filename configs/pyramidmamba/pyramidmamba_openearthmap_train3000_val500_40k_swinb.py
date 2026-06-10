# PyramidMamba-Swin-B + DAPCN on the OpenEarthMap wo_xBD split.
#   train: train_woxbd_3000.txt (3000)
#   val:   val_woxbd_500.txt    (500)
# Inherits the full train1500 swinb config; only the splits + work_dir change.

_base_ = ['./pyramidmamba_openearthmap_train1500_40k_swinb.py']

data = dict(
    samples_per_gpu=12,
    train=dict(split='train_woxbd_3000.txt'),
    val=dict(split='val_woxbd_500.txt'),
)

work_dir = './work_dirs/openearthmap/pyramidmamba_dapcn_woxbd_train3000_val500_swinb'
