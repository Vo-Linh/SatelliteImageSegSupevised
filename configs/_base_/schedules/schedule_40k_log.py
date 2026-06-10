# Logarithmic Learning Rate Schedule for OpenEarthMap
# This schedule uses logarithmic decay instead of polynomial
# The curve is steeper at the start and flattens towards the end

optimizer = dict(
    type='AdamW',
    lr=3e-5,
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

# Logarithmic LR decay config
# Formula: lr = base_lr * (1 - log(progress) / log(max_iters))^power
lr_config = dict(
    policy='Log',           # Use LogLrUpdaterHook (registered as 'Log')
    power=1.0,              # Power of the log decay (1.0 = standard log)
    warmup='linear',        # Warmup strategy
    warmup_iters=1500,      # Warmup iterations
    warmup_ratio=1e-6,      # Initial warmup LR ratio
    min_lr=1e-7,            # Minimum learning rate
    by_epoch=False)         # Iteration-based training

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(
    by_epoch=False,
    interval=4000)

evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
