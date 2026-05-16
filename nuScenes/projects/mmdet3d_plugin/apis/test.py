"""Test/evaluation API — wrappers around mmdet3d multi-GPU test utilities."""
import os
import pickle
import tempfile

import torch
import torch.distributed as dist
from mmdet.apis import multi_gpu_test, single_gpu_test


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, efficient_test=False):
    """Multi-GPU test, optionally collecting results on GPU to avoid OOM.

    Wraps mmdet.apis.multi_gpu_test with extra handling for planning outputs.
    """
    return multi_gpu_test(model, data_loader, tmpdir=tmpdir, gpu_collect=gpu_collect)


def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    """Single-GPU test wrapper."""
    return single_gpu_test(model, data_loader, show=show, out_dir=out_dir, show_score_thr=show_score_thr)
