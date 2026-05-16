"""Training API — thin wrapper around mmdet3d's train_detector."""
from mmdet.apis import train_detector


def custom_train_model(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Wrapper around mmdet train_detector that supports the UniDriveVLA plugin.

    Identical to mmdet.apis.train_detector; exists as an extension point for
    custom training loops (e.g. AR co-training, DeepSpeed ZeRO).
    """
    train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
    )
