"""
UniDriveVLA Stage 2 — nuScenes mini (v1.0-mini)
Adds VLM (Qwen-VL) integration on top of Stage 1 perception model.
Stage 2 fine-tunes the VLM adapter while keeping the BEV backbone frozen.
"""

_base_ = ['./unidrivevla_mini_stage1.py']

# ---------------------------------------------------------------------------
# Stage 2 overrides
# ---------------------------------------------------------------------------
total_epochs = 15

# Load from stage 1 checkpoint (set via CLI --load-from or override here)
load_from = None

# VLM config (Qwen-VL 7B via HuggingFace)
vlm_config = dict(
    model_name_or_path='Qwen/Qwen-VL-Chat',
    torch_dtype='bfloat16',
    use_flash_attention_2=True,
    # LoRA adapter config (PEFT)
    lora_config=dict(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    ),
    # Projection from BEV embed_dims to VLM hidden size
    bev_proj=dict(
        in_channels=256,
        out_channels=4096,
        num_layers=2,
    ),
)

model = dict(
    type='UniDriveVLA',
    use_grid_mask=True,
    video_test_mode=True,
    # Freeze backbone and BEV encoder in stage 2
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bev_encoder=True,
    # VLM integration
    vlm_config=vlm_config,
    # planning_head inherits from stage1 _base_
    planning_head=dict(
        type='UnifiedPerceptionDecoder',
        bev_h=50,
        bev_w=50,
        num_cam=6,
        num_feature_levels=1,
        embed_dims=256,
        num_det_classes=10,
        num_map_classes=3,
        # Stage 2 adds VLM language conditioning
        use_vlm_features=True,
        vlm_embed_dims=4096,
        task_loss_weight=dict(
            detection=0.5,
            mapping=0.5,
            planning=1.0,
            vqa=1.0,
        ),
        loss_det_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_det_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_det_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_planning=dict(type='L1Loss', loss_weight=1.0),
        loss_vqa=dict(type='CrossEntropyLoss', loss_weight=1.0),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=50,
            col_num_embed=50,
        ),
    ),
)

# Stage 2 optimiser: lower LR, only train VLM adapter + projection
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    weight_decay=0.0,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'encoder': dict(lr_mult=0.0),
        },
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=3, max_keep_ckpts=5)
evaluation = dict(interval=3)

# DeepSpeed ZeRO-1 for VLM training
deepspeed_config = 'nuScenes/zero_configs/adam_zero1_bf16.json'
