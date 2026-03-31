optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=2.0, decay_mult=1.0),
            'prototypes': dict(lr_mult=1.0, decay_mult=0.01),
            'quality': dict(lr_mult=1.0, decay_mult=1.0),
        }))
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(
    by_epoch=False,
    interval=4000,
    max_keep_ckpts=2)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
