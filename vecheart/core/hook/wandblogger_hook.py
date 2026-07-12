import importlib
import os.path as osp
import sys

from reimu.runner import HOOKS
from reimu.runner.dist_utils import master_only
from reimu.runner.hooks.checkpoint import CheckpointHook
from reimu.runner.hooks.logger.wandb import WandbLoggerHook

from .eval_hooks import EvalHook


@HOOKS.register_module()
class WandbHook(WandbLoggerHook):
    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 **kwargs):
        super(WandbHook, self).__init__(init_kwargs, interval, **kwargs)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                if self.eval_hook._should_evaluate(runner) and self.get_mode(runner) == 'train':
                    self.wandb.log(
                        tags, step=self.get_iter(runner), commit=False)
                else:
                    self.wandb.log(
                        tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)

    @master_only
    def before_run(self, runner):
        super(WandbHook, self).before_run(runner)

        # Save and Log config.
        if runner.meta is not None:
            src_cfg_path = osp.join(runner.work_dir,
                                    runner.meta.get('exp_name', None))
            if osp.exists(src_cfg_path):
                self.wandb.save(src_cfg_path, base_path=runner.work_dir)
                self._update_wandb_config(runner)
        else:
            runner.logger.warning('No meta information found in the runner. ')

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, EvalHook):
                self.eval_hook = hook

    def _update_wandb_config(self, runner):
        """Update wandb config."""
        # Import the config file.
        sys.path.append(runner.work_dir)
        config_filename = runner.meta['exp_name'][:-3]
        configs = importlib.import_module(config_filename)
        # Prepare a nested dict of config variables.
        config_keys = [key for key in dir(configs) if not key.startswith('__')]
        config_dict = {key: getattr(configs, key) for key in config_keys}
        # Update the W&B config.
        self.wandb.config.update(config_dict)