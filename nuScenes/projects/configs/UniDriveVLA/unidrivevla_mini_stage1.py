"""
UniDriveVLA Stage 1 — nuScenes mini (v1.0-mini)
Perception-only training: detection + online mapping + planning.

Key adaptations from BEVFormer tiny for nuScenes mini:
  - version     : v1.0-mini
  - input_size  : (450, 800)   # H × W
  - bev_h/bev_w : 50 × 50
  - queue_length: 2
  - batch_size  : 1 per GPU
  - encoder_layers: 3
  - feature_scales: single (C5 only)
  - num_epochs  : 24
"""

_base_ = ['../_base_/default_runtime.py']

# ---------------------------------------------------------------------------
# Dataset / path settings
# ---------------------------------------------------------------------------
dataset_type = 'NuScenes3DDataset'
data_root = 'data/nuscenes/'
ann_file_train = 'data/infos/nuscenes_infos_mini_train.pkl'
ann_file_val   = 'data/infos/nuscenes_infos_mini_val.pkl'
version        = 'v1.0-mini'

# ---------------------------------------------------------------------------
# Image / BEV settings
# ---------------------------------------------------------------------------
input_size    = (450, 800)   # (H, W)
bev_h         = 50
bev_w         = 50
queue_length  = 2
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size    = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
detection_classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
map_classes = ['divider', 'ped_crossing', 'boundary']
num_det_classes = len(detection_classes)
num_map_classes = len(map_classes)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_dim_         = 256
_pos_dim_     = _dim_ // 2
_ffn_dim_     = _dim_ * 2
_num_levels_  = 1   # single-scale (C5 only)
_num_cams_    = 6
_num_enc_layers_ = 3  # BEVFormer tiny uses 3

model = dict(
    type='UniDriveVLA',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),          # C5 only
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
        ),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    planning_head=dict(
        type='UnifiedPerceptionDecoder',
        # BEV config
        bev_h=bev_h,
        bev_w=bev_w,
        num_cam=_num_cams_,
        num_feature_levels=_num_levels_,
        embed_dims=_dim_,
        # Detection
        num_det_classes=num_det_classes,
        # Map
        num_map_classes=num_map_classes,
        # BEV encoder
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=_num_enc_layers_,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1,
                    ),
                    dict(
                        type='SpatialCrossAttention',
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_,
                        ),
                        embed_dims=_dim_,
                    ),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=(
                    'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'
                ),
            ),
        ),
        # Decoder (detection + map)
        decoder=dict(
            type='DetectionTransformerDecoder',
            num_layers=6,
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=8,
                    dropout=0.1,
                ),
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=(
                    'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'
                ),
            ),
        ),
        # Loss weights
        loss_det_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_det_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_det_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=5.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_planning=dict(type='L1Loss', loss_weight=1.0),
        # Positional encoding
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
        ),
        # Task weights
        task_loss_weight=dict(
            detection=1.0,
            mapping=1.0,
            planning=1.0,
        ),
    ),
    # Training cfg
    train_cfg=dict(
        det=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
            ),
        ),
    ),
    test_cfg=dict(
        det=dict(
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
        ),
    ),
)

# ---------------------------------------------------------------------------
# Data pipelines
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=detection_classes),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=detection_classes,
        with_ego=True,
    ),
    dict(
        type='CustomCollect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_ego_his_trajs',
              'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
              'gt_agent_his_trajs', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
              'sdc_planning', 'sdc_planning_mask', 'command'],
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=input_size,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=detection_classes, with_label=False),
            dict(type='CustomCollect3D', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=detection_classes,
        map_classes=map_classes,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True,
        ),
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h, bev_w),
        queue_length=queue_length,
        point_cloud_range=point_cloud_range,
        version=version,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=detection_classes,
        map_classes=map_classes,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True,
        ),
        test_mode=True,
        bev_size=(bev_h, bev_w),
        version=version,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=detection_classes,
        map_classes=map_classes,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True,
        ),
        test_mode=True,
        bev_size=(bev_h, bev_w),
        version=version,
    ),
)

# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1)},
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# ---------------------------------------------------------------------------
# Logging / checkpointing
# ---------------------------------------------------------------------------
checkpoint_config = dict(interval=4, max_keep_ckpts=5)
evaluation = dict(interval=4, pipeline=test_pipeline)
find_unused_parameters = True
