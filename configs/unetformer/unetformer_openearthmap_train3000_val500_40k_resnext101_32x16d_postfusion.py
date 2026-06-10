# UNetFormer + ResNeXt101_32x16d on OpenEarthMap (wo_xBD split)
#
# Same model / optimizer / schedule as
#   unetformer_openearthmap_train3500_40k_resnext101_32x16d_postfusion.py
# but trains on the OpenEarthMap_wo_xBD splits:
#   train: /home/ubuntu/data/OpenEarthMap/OpenEarthMap_wo_xBD/train.txt (3000)
#   val:   /home/ubuntu/data/OpenEarthMap/OpenEarthMap_wo_xBD/val.txt   (500)
#
# Note: the OpenEarthMapDataset loader strips the basename and re-appends
# ``img_suffix`` (.tif), so the split files below are the ``.tif``-stripped
# versions of the wo_xBD lists, materialized under the flat data_root.

_base_ = ['./unetformer_openearthmap_train3500_40k_resnext101_32x16d_postfusion.py']

work_dir = './work_dirs/openearthmap/unetformer_woxbd_resnext101_32x16d_postfusion'

data = dict(
    samples_per_gpu=8,
    train=dict(split='train_woxbd_3000.txt'),
    val=dict(split='val_woxbd_500.txt'),
)
