"""Standard mmdet3d runtime configuration used by all UniDriveVLA experiments."""

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

# Use the same cudnn determinism as BEVFormer tiny
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None

# Load from a previous checkpoint (set in child configs or CLI)
load_from = None
resume_from = None

# Workflow: [(phase, num_epochs), ...]
workflow = [('train', 1)]

# Disable OpenCV multi-threading to avoid dataloader conflicts
opencv_num_threads = 0
mp_start_method = 'fork'

# Evaluation interval (epochs)
evaluation = dict(interval=1, pipeline=None)
