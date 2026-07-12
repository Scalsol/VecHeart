# Shared runtime: logging and misc flags.

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [
    dict(type='VisualizationHook', resolution=128, by_epoch=False, interval=2500, return_gt=True),
]

log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
work_dir = "work_dirs/${cfg_name}"
