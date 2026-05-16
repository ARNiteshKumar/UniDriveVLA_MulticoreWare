"""
QwenVL3APlanningHead
====================
Stage 2 planning head that wraps Qwen3-VL-2B-Instruct with LoRA adapters
and integrates VLM features into the UnifiedPerceptionDecoder.

Architecture (based on original UniDriveVLA paper):
  img features (backbone+neck) → BEV encoder → UnifiedPerceptionDecoder (Stage 1)
                                                         ↓
  multi-camera images → Qwen3-VL-2B → VLM tokens → Stage 2 cross-attn → planning

The VLM is kept frozen (except LoRA adapters) during Stage 2 training.
Action horizon: 6 waypoints at 0.5 s intervals (3 s total).
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_head
from mmcv.runner import BaseModule


@HEADS.register_module()
class QwenVL3APlanningHead(BaseModule):
    """Qwen3-VL-2B planning head with LoRA fine-tuning (Stage 2).

    Parameters
    ----------
    pretrained_path : str
        Path to Qwen3-VL-2B-Instruct weights (or HF model ID).
    vlm_variant : str
        '2b' (default) or '7b'.
    dtype : str
        'bfloat16' (default) or 'float16'.  'float32' for debugging.
    train_vlm : bool
        Whether to update VLM base weights.  False → only LoRA trains.
    attn_implementation : str
        'eager' (T4-compatible) or 'flash_attention_2' (A100/H100).
    action_dim : int
        Waypoint dimensionality (default: 2 = x, y).
    action_horizon : int
        Number of future waypoints (default: 6 = 3 s at 0.5 s steps).
    lora_cfg : dict
        LoRA configuration passed to peft.LoraConfig.
    occworld_vae_path : str or None
        Path to OccWorld VAE checkpoint for occupancy supervision.
    unified_decoder_cfg : dict
        Config dict for UnifiedPerceptionDecoder (stage1 perception head).
    """

    def __init__(
        self,
        pretrained_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        vlm_variant: str = "2b",
        dtype: str = "bfloat16",
        train_vlm: bool = False,
        attn_implementation: str = "eager",
        inference_attn_impl: str = "eager",
        action_dim: int = 2,
        action_horizon: int = 6,
        lora_cfg: Optional[dict] = None,
        occworld_vae_path: Optional[str] = None,
        occ_loss_weight: float = 1.0,
        depth_loss_weight: float = 0.2,
        collision_loss_weight: float = 0.0,
        map_bound_loss_weight: float = 0.0,
        vlm_grad_scale: float = 1.0,
        unified_decoder_cfg: Optional[dict] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        self.pretrained_path = pretrained_path
        self.vlm_variant = vlm_variant
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.attn_implementation = attn_implementation
        self.occ_loss_weight = occ_loss_weight
        self.depth_loss_weight = depth_loss_weight

        # Build the stage-1 perception decoder
        if unified_decoder_cfg is not None:
            self.perception_decoder = build_head(unified_decoder_cfg)
        else:
            self.perception_decoder = None

        # VLM (loaded lazily on first forward to avoid long import at module init)
        self._vlm_loaded = False
        self._lora_cfg = lora_cfg
        self._train_vlm = train_vlm
        self._dtype_str = dtype
        self._attn_impl = attn_implementation

        # OccWorld VAE (loaded lazily)
        self._occworld_vae_path = occworld_vae_path
        self._occ_vae = None

        # Action projection: VLM hidden_dim → action_horizon × action_dim
        # Actual hidden dim depends on vlm_variant; set in _load_vlm()
        self.action_head: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    # Lazy VLM loading (avoids ~5 GB VRAM usage at import time)
    # ------------------------------------------------------------------

    def _load_vlm(self):
        """Load Qwen3-VL-2B with optional LoRA adapters."""
        if self._vlm_loaded:
            return

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers>=4.49 required for QwenVL3APlanningHead. "
                "Run: pip install transformers>=4.49"
            ) from e

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.pretrained_path,
            torch_dtype=torch_dtype,
            attn_implementation=self._attn_impl,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.pretrained_path)

        if not self._train_vlm:
            for p in self.vlm.parameters():
                p.requires_grad_(False)

        # Apply LoRA if configured
        if self._lora_cfg is not None:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    **self._lora_cfg,
                )
                self.vlm = get_peft_model(self.vlm, lora_config)
                self.vlm.print_trainable_parameters()
            except ImportError:
                raise ImportError("peft required for LoRA. Run: pip install peft")

        # Build action projection head
        hidden = self.vlm.config.hidden_size
        self.action_head = nn.Linear(hidden, self.action_horizon * self.action_dim)

        self._vlm_loaded = True

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _extract_vlm_tokens(self, img, meta_info=None):
        """Run the VLM encoder on multi-camera images and return hidden states.

        Args:
            img: (B, N_cam, C, H, W) tensor of camera images.
        Returns:
            vlm_tokens: (B, N_vlm, hidden_dim) VLM encoder output.
        """
        self._load_vlm()
        # Forward through VLM encoder only (output_hidden_states=True)
        B, N, C, H, W = img.shape
        # Flatten cams: run each sample independently (memory efficient on T4)
        hidden_states_list = []
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for b in range(B):
                pixel_values = img[b]  # (N_cam, C, H, W)
                out = self.vlm.visual(pixel_values, output_hidden_states=True)
                hidden_states_list.append(out.last_hidden_state.mean(dim=0, keepdim=True))
        return torch.cat(hidden_states_list, dim=0)  # (B, seq_len, hidden)

    def _planning_from_vlm(self, vlm_tokens):
        """Project VLM tokens to planning waypoints.

        Args:
            vlm_tokens: (B, N, hidden)
        Returns:
            trajs: (B, action_horizon, action_dim)
        """
        pooled = vlm_tokens.mean(dim=1)  # (B, hidden)
        trajs = self.action_head(pooled)  # (B, horizon * dim)
        return trajs.reshape(-1, self.action_horizon, self.action_dim)

    # ------------------------------------------------------------------
    # Train / Test
    # ------------------------------------------------------------------

    def forward_train(self, img=None, bev_features=None, **kwargs):
        """Stage 2 training forward.

        1. Run Stage 1 perception decoder on BEV features.
        2. Run VLM on multi-camera images.
        3. Inject VLM tokens into Stage 2 decoder.
        4. Compute all losses.
        """
        losses = {}

        if self.perception_decoder is not None and bev_features is not None:
            s1_out = self.perception_decoder.forward_stage1(bev_features)
            if img is not None:
                vlm_tokens = self._extract_vlm_tokens(img)
                s2_out = self.perception_decoder.forward_stage2(s1_out, bev_features, vlm_tokens)
            else:
                s2_out = s1_out
            preds = self.perception_decoder.predict(s2_out)
            det_losses = self.perception_decoder.loss(preds, **kwargs)
            losses.update(det_losses)

        return {"losses": losses}

    def forward_test(self, img=None, bev_features=None, **kwargs):
        """Stage 2 inference forward."""
        if self.perception_decoder is None:
            return {}

        with torch.no_grad():
            if bev_features is not None:
                s1_out = self.perception_decoder.forward_stage1(bev_features)
                if img is not None:
                    self._load_vlm()
                    vlm_tokens = self._extract_vlm_tokens(img)
                    s2_out = self.perception_decoder.forward_stage2(s1_out, bev_features, vlm_tokens)
                else:
                    s2_out = s1_out
                return self.perception_decoder.predict(s2_out)
            return {}
