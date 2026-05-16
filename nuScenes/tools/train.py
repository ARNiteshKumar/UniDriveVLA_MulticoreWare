# Copyright (c) OpenMMLab. All rights reserved.
# Sourced from: https://github.com/xiaomi-research/unidrivevla/blob/main/nuScenes/tools/train.py
from __future__ import division
import sys
import os

print(sys.executable, os.path.abspath(__file__))
import argparse
import copy
import mmcv
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from torch import distributed as dist
from datetime import timedelta

import cv2
cv2.setNumThreads(8)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='number of gpus to use (only for non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='ids of gpus to use (only for non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--dist-url', type=str, default='auto',
                        help='dist url for init process, such as tcp://localhost:8000')
    parser.add_argument('--gpus-per-machine', type=int, default=8)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi', 'mpi_nccl'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true',
                        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir).split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config).split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    if args.launcher == 'none':
        distributed = False
    elif args.launcher == 'mpi_nccl':
        distributed = True
        import mpi4py.MPI as MPI
        comm = MPI.COMM_WORLD
        mpi_local_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()
        device_ids_on_machines = list(range(args.gpus_per_machine))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids_on_machines))
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=mpi_world_size, rank=mpi_local_rank,
                                timeout=timedelta(seconds=3600))
        cfg.gpu_ids = range(mpi_world_size)
    else:
        distributed = True
        init_dist(args.launcher, timeout=timedelta(seconds=3600), **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    logger.info(f'Model:\n{model}')

    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )
    model.CLASSES = datasets[0].CLASSES
    if hasattr(cfg, 'plugin'):
        custom_train_model(model, datasets, cfg, distributed=distributed,
                           validate=(not args.no_validate), timestamp=timestamp, meta=meta)
    else:
        train_detector(model, datasets, cfg, distributed=distributed,
                       validate=(not args.no_validate), timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
