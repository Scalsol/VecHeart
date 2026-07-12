import copy

from reimu import ConfigDict


def compat_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg = compat_loader_args(cfg)
    return cfg


def compat_loader_args(cfg):
    """Deprecated sample_per_gpu in cfg.data."""

    cfg = copy.deepcopy(cfg)
    if 'train_dataloader' not in cfg.data:
        cfg.data['train_dataloader'] = ConfigDict()
    if 'val_dataloader' not in cfg.data:
        cfg.data['val_dataloader'] = ConfigDict()
    if 'test_dataloader' not in cfg.data:
        cfg.data['test_dataloader'] = ConfigDict()

    # special process for train_dataloader
    if 'samples_per_gpu' in cfg.data:
        samples_per_gpu = cfg.data.pop('samples_per_gpu')
        cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu

    if 'persistent_workers' in cfg.data:
        persistent_workers = cfg.data.pop('persistent_workers')
        cfg.data.train_dataloader['persistent_workers'] = persistent_workers

    if 'workers_per_gpu' in cfg.data:
        workers_per_gpu = cfg.data.pop('workers_per_gpu')
        cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu

    return cfg
