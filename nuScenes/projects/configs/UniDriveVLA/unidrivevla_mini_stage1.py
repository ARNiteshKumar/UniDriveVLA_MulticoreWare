# UniDriveVLA — Stage 1 config for nuScenes v1.0-mini
#
# Follows BEVFormer-tiny style:
#   backbone  : ResNet-50, frozen_stages=1, out_indices=(3,) [C5 only]
#   BEV grid  : 50 x 50  (vs 200x200 in full model)
#   img size  : 800 x 450  via RandomScaleImageMultiViewImage(scales=[0.5])
#               applied to 1600x900 native nuScenes images
#   enc layers: 3  (BEVFormerEncoder: TemporalSelfAttn + SpatialCrossAttn)
#   queue len : 3  (temporal frames per sample)
#   dataset   : v1.0-mini  (~323 samples, 7 train / 3 val scenes)
#   epochs    : 24
#
# Stage 1 trains the perception stack only (detection + map + planning).
# No VLM is active at this stage.

_base_ = ['../_base_/default_runtime.py']

plugin     = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ── spatial config ────────────────────────────────────────────
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size        = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ── classes ───────────────────────────────────────────────────
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
map_classes = ['divider', 'ped_crossing', 'boundary']

input_modality = dict(
    use_lidar=False, use_camera=True,
    use_radar=False, use_map=False, use_external=True,
)

# ── model dims (BEVFormer-tiny style) ─────────────────────────
_dim_        = 256
_pos_dim_    = _dim_ // 2
_ffn_dim_    = _dim_ * 2
_num_levels_ = 1        # single-scale: C5 only
_num_cams_   = 6
_num_enc_    = 3        # encoder layers (BEVFormer-tiny: 3)
bev_h_       = 50
bev_w_       = 50
queue_length = 3        # temporal frames (BEVFormer-tiny: 3)

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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
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
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_cam=_num_cams_,
        num_feature_levels=_num_levels_,
        embed_dims=_dim_,
        num_det_classes=len(class_names),
        num_map_classes=len(map_classes),
        num_stage1_layers=_num_enc_,
        num_stage2_layers=_num_enc_,
        num_heads=8,
        ffn_dim=_ffn_dim_,
        dropout=0.1,
        # BEV encoder (BEVFormer-tiny style: 3 layers, TemporalSelfAttn + SpatialCrossAttn)
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=_num_enc_,
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
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            ),
        ),
        # Detection query decoder (6 layers)
        decoder=dict(
            type='DetectionTransformerDecoder',
            num_layers=6,
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1,
                    ),
                    dict(
                        type='CustomMSDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                    ),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            ),
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=5.0,
        ),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        task_loss_weight=dict(detection=1.0, mapping=1.0, planning=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range,
            ),
        ),
    ),
    test_cfg=dict(
        pts=dict(
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

# ── dataset & pipelines ───────────────────────────────────────
dataset_type = 'NuScenes3DDataset'
data_root    = 'data/nuscenes/'
# PKL files generated by:  bash scripts/prepare_data.sh  (from repo root)
ann_file_train = 'data/infos/nuscenes_mini_temporal_train.pkl'
ann_file_val   = 'data/infos/nuscenes_mini_temporal_val.pkl'
version        = 'v1.0-mini'

# Train pipeline — scale 1600x900 → 800x450 (BEVFormer-tiny recipe)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True, with_label_3d=True, with_attr_label=False,
    ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),   # 1600x900 -> 800x450
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img',
            'gt_ego_his_trajs', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
            'gt_ego_fut_cmd', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'sdc_planning', 'sdc_planning_mask', 'command',
        ],
    ),
]

# Test pipeline — same scale inside MultiScaleFlipAug3D
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),  # -> 800x450
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False,
            ),
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
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        point_cloud_range=point_cloud_range,
        version=version,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=1,
        version=version,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=1,
        version=version,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# ── optimiser / scheduler ─────────────────────────────────────
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
runner       = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config   = dict(interval=4, max_keep_ckpts=5)
evaluation          = dict(interval=4, pipeline=test_pipeline)
find_unused_parameters = True

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
