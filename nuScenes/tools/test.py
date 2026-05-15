#!/usr/bin/env python3
"""
Test / evaluation entry point for UniDriveVLA.

Usage
-----
Single GPU:
    python tools/test.py <config> <checkpoint> [--eval bbox map] [--out results.pkl]

Multi-GPU (via dist_eval.sh):
    bash tools/dist_eval.sh <config> <checkpoint> <N_GPUS>
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

import projects.mmdet3d_plugin  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='UniDriveVLA evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file (.pkl)')
    parser.add_argument('--json-out', help='output result file (.json) for submission')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics: bbox, map, planning')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory to save visualisations')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
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

    # Distributed setup
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, backend='nccl')
    rank, world_size = get_dist_info()

    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=distributed,
        shuffle=False,
    )

    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    if not distributed:
        model = MMDataParallel(model, device_ids=args.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=True)

    # Save results
    if rank == 0:
        if args.out:
            mmcv.dump(outputs, args.out)
            print(f'Results saved to {args.out}')

        if args.json_out:
            result_dict = {'results': outputs}
            with open(args.json_out, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            print(f'JSON results saved to {args.json_out}')

        # Run evaluation
        if args.eval:
            eval_kwargs = {}
            metrics = dataset.evaluate(outputs, metric=args.eval, **eval_kwargs)
            print('\n=== Evaluation Results ===')
            for k, v in metrics.items():
                print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
