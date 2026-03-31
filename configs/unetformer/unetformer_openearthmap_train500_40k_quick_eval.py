_base_ = './unetformer_openearthmap_train500_40k_stable.py'

data = dict(
    val=dict(
        split='val_200_quick.txt',
    ),
    test=dict(
        split='val_200_quick.txt',
    )
)

evaluation = dict(interval=500, metric='mIoU', pre_eval=True)
