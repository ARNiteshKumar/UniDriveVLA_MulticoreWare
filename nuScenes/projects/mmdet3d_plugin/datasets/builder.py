"""Dataset builder — thin wrapper used by nuScenes/tools/test.py."""
from mmdet.datasets import build_dataloader as _mmdet_build_dataloader


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    **kwargs,
):
    return _mmdet_build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        num_gpus=num_gpus,
        dist=dist,
        shuffle=shuffle,
        seed=seed,
        **kwargs,
    )
