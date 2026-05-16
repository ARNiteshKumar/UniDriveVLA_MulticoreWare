# Sourced from: https://github.com/xiaomi-research/unidrivevla/blob/main/nuScenes/tools/test.py
import argparse
import cv2
import numpy as np
import torch
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl',
                        help='output result file in pickle format')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, e.g., "bbox" "map" "motion" "planning"')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir', help='directory where results will be saved')
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir', help='tmp directory for collecting results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--eval-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch')
    parser.add_argument('--local_rank', type=int, default=0)
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

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            # Save pred_trajs for nuScenes planning eval
            pred_trajs_path = args.out.replace('.pkl', '_pred_trajs.pkl')
            pred_trajs_dict = {}
            data_infos = dataset.data_infos
            for i, out in enumerate(outputs):
                if i >= len(data_infos):
                    break
                token = data_infos[i]['token']
                img_bbox = out.get('img_bbox', out)
                traj = img_bbox.get('final_planning')
                if traj is not None:
                    if hasattr(traj, 'numpy'):
                        traj = traj.numpy()
                    pred_trajs_dict[token] = traj[:, :2]  # (T, 2)
            mmcv.dump(pred_trajs_dict, pred_trajs_path)
            print(f'pred_trajs saved to {pred_trajs_path} ({len(pred_trajs_dict)} samples)')

        kwargs = {} if args.eval_options is None else args.eval_options
        if args.show_dir:
            kwargs['jsonfile_prefix'] = args.show_dir
        else:
            kwargs['jsonfile_prefix'] = osp.join(
                'test', args.config.split('/')[-1].split('.')[-2],
                time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
