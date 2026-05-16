# UniDriveVLA — Stage 2 config for nuScenes v1.0-mini
#
# Inherits all Stage 1 settings (BEVFormer-tiny style perception stack)
# and adds the Qwen3-VL-2B-Instruct VLM planning head with LoRA.
#
# Required env vars before running:
#   export VLM_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct
#   export OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
#   export STAGE1_CHECKPOINT=/path/to/stage1/checkpoint.pth
#
# Set by download_checkpoints.sh automatically.

_base_ = ['./unidrivevla_mini_stage1.py']

import os

vlm_pretrained_path = os.environ.get(
    'VLM_PRETRAINED_PATH', 'Qwen/Qwen3-VL-2B-Instruct'
)
occworld_vae_path = os.environ.get('OCCWORLD_VAE_PATH', None)
stage1_ckpt       = os.environ.get('STAGE1_CHECKPOINT', None)

load_from = stage1_ckpt  # resume perception weights from stage 1

# ── VLM planning head ─────────────────────────────────────────
model = dict(
    type='UniDriveVLA',
    planning_head=dict(
        type='QwenVL3APlanningHead',
        # Qwen3-VL-2B base model (set VLM_PRETRAINED_PATH env var)
        pretrained_path=vlm_pretrained_path,
        vlm_variant='2b',
        dtype='bfloat16',
        train_vlm=False,               # keep VLM frozen; only LoRA adapters train
        # Use eager attention (compatible with T4; flash-attn optional)
        attn_implementation='eager',
        inference_attn_impl='eager',
        # Action head
        action_dim=2,
        action_horizon=6,              # 3 s at 0.5 s intervals
        # LoRA (rank 64 as in original UniDriveVLA)
        lora_cfg=dict(
            r=64,
            lora_alpha=128,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            lora_dropout=0.05,
            bias='none',
        ),
        # OccWorld VAE for occupancy supervision
        occworld_vae_path=occworld_vae_path,
        # Loss weights
        occ_loss_weight=1.0,
        depth_loss_weight=0.2,
        collision_loss_weight=0.0,
        map_bound_loss_weight=0.0,
        vlm_grad_scale=1.0,
        # Perception decoder (same BEV dims as stage 1)
        unified_decoder_cfg=dict(
            type='UnifiedPerceptionDecoder',
            embed_dims=256,
            bev_h=50,
            bev_w=50,
            num_det_classes=10,
            num_map_classes=3,
            num_stage1_layers=3,
            num_stage2_layers=3,
        ),
    ),
)

# ── Stage 2 optimiser: lower LR, only VLM adapter + projection train ──
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),   # frozen
            'img_neck':     dict(lr_mult=0.0),   # frozen
            'encoder':      dict(lr_mult=0.0),   # frozen
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

total_epochs = 15
runner       = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=3, max_keep_ckpts=5)
evaluation        = dict(interval=3)

# DeepSpeed ZeRO-1 config (optional — for multi-GPU setups)
# deepspeed_config = 'zero_configs/adam_zero1_bf16.json'
