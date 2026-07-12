# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import time

import reimu
import torch
from reimu import Config, DictAction
from reimu.runner import (finalize_dist, get_dist_info, init_dist,
                          load_checkpoint, wrap_fp16_model)
from reimu.utils import replace_cfg_vals

from vecheart.apis import multi_gpu_test, single_gpu_test
from vecheart.datasets import build_dataloader, build_dataset
from vecheart.models import build_reconstructor
from vecheart.utils import (build_ddp, build_dp, compat_cfg,
                            setup_multi_processes)


def parse_args():
    parser = argparse.ArgumentParser(
        description='VecHeart test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--eval',
        action='store_true',
        help='whether to evaluate the results')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--resolution', type=int, default=128, help='resolution for reconstruction')
    parser.add_argument(
        '--return-volume', action='store_true', help='whether evaluate volume')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def load_cfg(args):
    cfg = Config.fromfile(args.config)
    cfg.cfg_name = osp.splitext(osp.basename(cfg.filename))[0]

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)
    cfg = compat_cfg(cfg)

    # allows not to create
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.show:
        if not args.show_dir:
            args.show_dir = cfg.work_dir.replace('work_dirs', 'vis_dir')

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    return cfg


def main():
    args = parse_args()

    assert args.eval or args.show, \
        ('Please specify at least one operation (eval/show the '
         'results / save the results) with the argument "--eval"'
         ', "--show" or "--show-dir"')

    cfg = load_cfg(args)

    cfg.device = "cuda"
    # gpu_ids is for single-gpu evaluation
    cfg.gpu_ids = [0]
    # below is for multi-gpu evaluation
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher)

    test_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=distributed,
        shuffle=False,
        seed=1
    )

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    if rank == 0:
        reimu.mkdir_or_exist(osp.abspath(cfg.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(cfg.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_reconstructor(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model,
                                  data_loader,
                                  return_label=args.eval,
                                  show=args.show,
                                  show_dir=args.show_dir,
                                  **{'resolution': args.resolution,
                                     'return_volume': args.return_volume})
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(
            model, data_loader, gpu_collect=cfg.evaluation.get('gpu_collect', False), **{'resolution': args.resolution})

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(show_instance_metrics=True, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if cfg.work_dir is not None:
                reimu.dump(metric_dict, json_file, indent=4)

    if distributed:
        finalize_dist()


if __name__ == '__main__':
    main()
