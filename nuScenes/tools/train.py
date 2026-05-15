#!/usr/bin/env python3
"""
Training entry point for UniDriveVLA.

Usage
-----
Single GPU:
    python tools/train.py <config> [--work-dir DIR] [--load-from CKPT]

Multi-GPU (via dist_train.sh):
    bash tools/dist_train.sh <config> <N_GPUS> [--work-dir DIR]
"""

import argparse
import os
import sys
import time

# Ensure the project package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# Register custom modules
import projects.mmdet3d_plugin  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='UniDriveVLA training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument('--load-from', help='load weights from a checkpoint file')
    parser.add_argument('--resume-from', help='resume training from a checkpoint')
    parser.add_argument('--auto-resume', action='store_true',
                        help='resume from latest checkpoint in work_dir')
    parser.add_argument('--no-validate', action='store_true',
                        help='skip validation during training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir') is None:
        cfg.work_dir = os.path.join(
            'work_dirs',
            os.path.splitext(os.path.basename(args.config))[0],
        )

    if args.load_from:
        cfg.load_from = args.load_from
    if args.resume_from:
        cfg.resume_from = args.resume_from

    # Distributed init
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, backend='nccl')
    rank, world_size = get_dist_info()

    # Create work dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # Dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    # Logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = mmcv.get_logger('mmdet', log_file=log_file)
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Reproducibility
    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # Build model
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]

    # Train
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=dict(seed=args.seed),
    )


if __name__ == '__main__':
    main()
