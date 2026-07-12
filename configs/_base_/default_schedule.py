# Shared training schedule: optimizer, LR policy, runner length, and evaluation/checkpoint cadence.

# optimizer
optimizer = dict(type='AdamW', lr=4e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=None)
fp16 = None  # dict(loss_scale='dynamic')
auto_scale_lr = dict(enable=False, base_batch_size=4)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 100,
    min_lr_ratio=1e-2
)
runner = dict(type='IterBasedRunner', max_iters=100000)

# evaluation / checkpointing
evaluation = dict(
    start=10000,
    interval=10000,
    num_classes=5,
    save_best='ChamDis',
    metrics=['ChamDis', 'PointIoU'],
    resolution=128
)
checkpoint_config = dict(interval=10000, max_keep_ckpts=2)
