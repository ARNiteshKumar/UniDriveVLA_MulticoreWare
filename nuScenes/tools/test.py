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
    # ── export flags ──────────────────────────────────────────────────────────
    parser.add_argument('--export', action='store_true',
                        help='Export the loaded model (planning head + decoder) to '
                             'ONNX and TorchScript after loading the checkpoint.')
    parser.add_argument('--export-dir', default='exports',
                        help='Directory to save exported ONNX / TorchScript files.')
    parser.add_argument('--export-only', action='store_true',
                        help='Run export then exit without running evaluation.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def export_model(model, export_dir: str):
    """Export planning_head (QwenVL3APlanningHead) + decoder to ONNX and TorchScript.

    Called when --export or --export-only is passed to test.py.
    The model must already have a checkpoint loaded.

    Exports:
        <export_dir>/planning_head.onnx       ONNX opset 14
        <export_dir>/planning_head.pt         TorchScript (torch.jit.trace)
        <export_dir>/planning_head_info.json  metadata
    """
    import json
    from pathlib import Path

    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── unwrap DDP if needed ──────────────────────────────────────────────────
    raw = model.module if hasattr(model, 'module') else model

    # ── locate planning_head ──────────────────────────────────────────────────
    planning_head = getattr(raw, 'planning_head', None)
    if planning_head is None:
        print('[export] WARNING: model has no planning_head — nothing to export.')
        return

    print('\n' + '=' * 58)
    print('  Export: QwenVL3APlanningHead (from loaded checkpoint)')
    print('=' * 58)

    # ── build a thin trace wrapper around planning_head.perception_decoder ────
    # Input to the decoder is BEV tokens: (B, bev_h*bev_w, embed_dim)
    decoder = getattr(planning_head, 'perception_decoder', None)
    if decoder is None:
        print('[export] WARNING: planning_head has no perception_decoder.')
        return

    class _DecodeWrapper(torch.nn.Module):
        """Wraps UnifiedPerceptionDecoder for export: BEV tokens → 6 outputs."""
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, bev_tokens):
            s1 = self.dec.forward_stage1(bev_tokens)
            preds = self.dec.predict(s1)
            return (
                preds['det_cls'],
                preds['det_bbox'],
                preds['map_cls'],
                preds['map_pts'],
                preds['plan_trajs'],
                preds['plan_scores'],
            )

    wrapper = _DecodeWrapper(decoder).eval()
    n_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
    print(f'  Decoder params : {n_params:.1f} M')

    bev_h = getattr(planning_head, '_bev_h', 50)
    bev_w = getattr(planning_head, '_bev_w', 50)
    embed_dim = decoder.embed_dim if hasattr(decoder, 'embed_dim') else 256
    dummy = torch.zeros(1, bev_h * bev_w, embed_dim, device='cpu')
    print(f'  Dummy input    : {list(dummy.shape)}  (1 batch, {bev_h}×{bev_w} BEV, {embed_dim}-dim)')

    # reference forward
    with torch.no_grad():
        ref = wrapper(dummy)
    out_names = ['det_cls', 'det_bbox', 'map_cls', 'map_pts', 'plan_trajs', 'plan_scores']
    print('  Output shapes:')
    for name, out in zip(out_names, ref):
        print(f'    {name:<14}: {list(out.shape)}')

    # ── ONNX ─────────────────────────────────────────────────────────────────
    onnx_path = out_dir / 'planning_head.onnx'
    print(f'\n  Exporting ONNX → {onnx_path}')
    t0 = time.time()
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=['bev_tokens'],
        output_names=out_names,
        dynamic_axes={'bev_tokens': {0: 'batch'}, **{n: {0: 'batch'} for n in out_names}},
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )
    onnx_mb = onnx_path.stat().st_size / 1e6
    print(f'  Done {time.time()-t0:.1f}s  |  {onnx_mb:.1f} MB')

    # verify
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        ort_outs = sess.run(None, {'bev_tokens': dummy.numpy()})
        ok = True
        for name, r, o in zip(out_names, ref, ort_outs):
            diff = abs(r.detach().numpy() - o).max()
            sym  = '✓' if diff < 1e-4 else '✗'
            print(f'  {sym}  {name:<14} max_diff={diff:.2e}')
            if diff >= 1e-4:
                ok = False
        print('  ONNX verified ✓' if ok else '  WARNING: outputs differ')
    except ImportError:
        print('  onnxruntime not installed — skipping verify')

    # ── TorchScript ──────────────────────────────────────────────────────────
    ts_path = out_dir / 'planning_head.pt'
    print(f'\n  Exporting TorchScript → {ts_path}')
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False)
    traced.save(str(ts_path))
    ts_mb = ts_path.stat().st_size / 1e6
    print(f'  Done {time.time()-t0:.1f}s  |  {ts_mb:.1f} MB')

    # ── metadata ──────────────────────────────────────────────────────────────
    info = {
        'component': 'QwenVL3APlanningHead → UnifiedPerceptionDecoder',
        'exported_from': 'nuScenes/tools/test.py --export',
        'input': {'name': 'bev_tokens',
                  'shape': [1, bev_h * bev_w, embed_dim],
                  'description': f'(batch, {bev_h}×{bev_w} BEV tokens, {embed_dim}-dim)'},
        'outputs': {n: {'shape': list(r.shape)} for n, r in zip(out_names, ref)},
        'parameters_M': round(n_params, 1),
        'file_sizes_MB': {'onnx': round(onnx_mb, 1), 'torchscript': round(ts_mb, 1)},
    }
    info_path = out_dir / 'planning_head_info.json'
    info_path.write_text(json.dumps(info, indent=2))

    print(f'\n  Metadata → {info_path}')
    print('=' * 58)
    print('  Export complete.')
    print(f'  {onnx_path}')
    print(f'  {ts_path}')
    print('=' * 58)


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

    # ── export (optional) ─────────────────────────────────────────────────────
    if args.export or args.export_only:
        export_model(model, args.export_dir)
        if args.export_only:
            return

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
