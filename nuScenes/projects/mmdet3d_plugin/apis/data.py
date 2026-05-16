"""Data loading API — wraps mmdet3d's build_dataloader."""
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
    """Build a DataLoader for UniDriveVLA datasets.

    Thin wrapper around mmdet.datasets.build_dataloader that passes all
    arguments through; exists as an extension point for future customisation
    (e.g. temporal queue collation, BEV augmentation).
    """
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
