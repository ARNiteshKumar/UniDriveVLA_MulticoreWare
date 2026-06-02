"""
UnifiedPerceptionDecoder
========================
A single transformer-based decoder that simultaneously handles:
  - 3D object detection   (10 nuScenes classes)
  - Online map prediction (3 element classes)
  - Ego-status estimation
  - Motion / trajectory prediction

Architecture mirrors the original UniDriveVLA paper but is tuned for the
nuScenes mini dataset (50×50 BEV, 800×450 images, BEVFormer-tiny style).

Stage 1  — perception-only; no VLM tokens injected.
Stage 2  — VLM output tokens are concatenated with BEV features before
           the second set of decoder layers.

ONNX-clean design notes:
  - Query params stored as (1, N, d) to avoid Unsqueeze ops in ONNX graph.
  - BEV positional encoding stored as (1, H*W, d) for same reason.
  - MultiheadAttention replaced with explicit Q/K/V projections to avoid the
    Split(in_proj_weight) op that nn.MultiheadAttention generates in ONNX.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import HEADS, build_loss

from .constants import (
    BEV_H_MINI,
    BEV_W_MINI,
    EGO_STATUS_DIM,
    MOTION_STEPS,
    NUSCENES_DETECTION_CLASSES,
    NUSCENES_MAP_CLASSES,
    PLANNING_STEPS,
)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Vanilla feed-forward network (2 or 3 layers)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVPositionEncoding(nn.Module):
    """2-D sine-cosine positional encoding over the BEV grid.

    pe stored as (1, H*W, d) so forward() needs no Unsqueeze op in ONNX.
    """

    def __init__(self, embed_dim: int, bev_h: int = BEV_H_MINI, bev_w: int = BEV_W_MINI):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sin-cos PE"
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        # Store as (1, H*W, d) — avoids Unsqueeze in ONNX graph
        self.register_buffer("pe", self._build_pe(embed_dim, bev_h, bev_w).unsqueeze(0))

    @staticmethod
    def _build_pe(d: int, h: int, w: int) -> torch.Tensor:
        d_half = d // 2
        freq = torch.arange(d_half // 2, dtype=torch.float32)
        freq = 1.0 / (10000 ** (2 * freq / d_half))

        ys = torch.arange(h, dtype=torch.float32).unsqueeze(1) * freq.unsqueeze(0)
        xs = torch.arange(w, dtype=torch.float32).unsqueeze(1) * freq.unsqueeze(0)

        pe_y = torch.cat([ys.sin(), ys.cos()], dim=-1)  # (H, d_half)
        pe_x = torch.cat([xs.sin(), xs.cos()], dim=-1)  # (W, d_half)

        pe = torch.cat(
            [
                pe_y.unsqueeze(1).expand(h, w, d_half),
                pe_x.unsqueeze(0).expand(h, w, d_half),
            ],
            dim=-1,
        )  # (H, W, d)
        return pe.view(h * w, d)  # (H*W, d)

    def forward(self, bev_tokens: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to BEV tokens.

        Args:
            bev_tokens: (B, H*W, d)
        Returns:
            (B, H*W, d)
        """
        # self.pe is (1, H*W, d) — no Unsqueeze needed, broadcasts over batch
        return bev_tokens + self.pe


class InstanceQueryBank(nn.Module):
    """Learnable instance-level queries stored as (1, N, d) Parameters.

    Using nn.Parameter with shape (1, N, d) instead of nn.Embedding avoids
    the Unsqueeze(0) + Expand pattern in the ONNX graph.
    """

    def __init__(self, num_det: int, num_map: int, embed_dim: int):
        super().__init__()
        # (1, N, d) — the leading 1 is the batch placeholder; Expand handles the rest
        self.det_queries = nn.Parameter(torch.empty(1, num_det, embed_dim))
        self.map_queries = nn.Parameter(torch.empty(1, num_map, embed_dim))
        nn.init.normal_(self.det_queries, std=0.02)
        nn.init.normal_(self.map_queries, std=0.02)

    def get_det(self) -> torch.Tensor:
        return self.det_queries  # (1, N_det, d)

    def get_map(self) -> torch.Tensor:
        return self.map_queries  # (1, N_map, d)


class ExplicitMHA(nn.Module):
    """Multi-head attention with explicit Q, K, V projection matrices.

    Replaces nn.MultiheadAttention to eliminate the Split(in_proj_weight)
    op that PyTorch's packed in_proj_weight produces in ONNX.

    Separate q_proj / k_proj / v_proj Linear layers each appear as a single
    MatMul in the ONNX graph — no Split needed.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Explicit projections — 3 separate MatMul ops, no Split
        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.k_proj  = nn.Linear(embed_dim, embed_dim)
        self.v_proj  = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            query : (B, Tq, d)
            key   : (B, Tk, d)
            value : (B, Tk, d)
        Returns:
            (B, Tq, d), None  — same signature as nn.MultiheadAttention
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(query).reshape(B, Tq, H, D).transpose(1, 2)   # (B, H, Tq, D)
        K = self.k_proj(key).reshape(B, Tk, H, D).transpose(1, 2)     # (B, H, Tk, D)
        V = self.v_proj(value).reshape(B, Tk, H, D).transpose(1, 2)   # (B, H, Tk, D)

        attn = (Q @ K.transpose(-2, -1)) * self.scale                  # (B, H, Tq, Tk)

        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            # (B, Tk) → (B, 1, 1, Tk)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, Tq, self.embed_dim)
        return self.out_proj(out), None


class BEVCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to BEV feature tokens."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = ExplicitMHA(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        bev_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attended, _ = self.attn(
            queries, bev_features, bev_features, key_padding_mask=key_padding_mask
        )
        return self.norm(queries + self.dropout(attended))


class SelfAttentionLayer(nn.Module):
    """Standard self-attention with pre-norm."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = ExplicitMHA(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + self.dropout(attended))


class FFNLayer(nn.Module):
    """Position-wise FFN with pre-norm."""

    def __init__(self, embed_dim: int, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        return self.norm(residual + self.dropout(x))


class PerceptionDecoderLayer(nn.Module):
    """One transformer decoder layer (self-attn → cross-attn → FFN)."""

    def __init__(self, embed_dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(embed_dim, num_heads, dropout)
        self.cross_attn = BEVCrossAttention(embed_dim, num_heads, dropout)
        self.ffn = FFNLayer(embed_dim, ffn_dim, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        bev_features: torch.Tensor,
        bev_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        queries = self.self_attn(queries)
        queries = self.cross_attn(queries, bev_features, bev_mask)
        queries = self.ffn(queries)
        return queries


# ---------------------------------------------------------------------------
# Task-specific prediction heads
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """Predicts 3-D bounding boxes + class logits per detection query."""

    def __init__(self, embed_dim: int, num_classes: int, num_layers: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = MLP(embed_dim, embed_dim, num_classes, num_layers)
        # Predict (cx, cy, cz, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy)
        self.reg_head = MLP(embed_dim, embed_dim, 10, num_layers)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls_head(queries), self.reg_head(queries)


class MapHead(nn.Module):
    """Predicts map element polylines + class logits per map query."""

    def __init__(self, embed_dim: int, num_classes: int, num_pts: int = 20, num_layers: int = 3):
        super().__init__()
        self.num_pts = num_pts
        self.cls_head = MLP(embed_dim, embed_dim, num_classes, num_layers)
        self.pts_head = MLP(embed_dim, embed_dim, num_pts * 2, num_layers)  # (x, y) per point

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_head(queries)
        pts = self.pts_head(queries).reshape(*queries.shape[:-1], self.num_pts, 2)
        return cls_logits, pts


class EgoStatusHead(nn.Module):
    """Predicts ego velocity and acceleration from a special ego query."""

    def __init__(self, embed_dim: int, ego_status_dim: int = EGO_STATUS_DIM):
        super().__init__()
        self.head = MLP(embed_dim, embed_dim // 2, ego_status_dim, 2)

    def forward(self, ego_query: torch.Tensor) -> torch.Tensor:
        return self.head(ego_query)


class MotionHead(nn.Module):
    """Predicts future trajectories for detected agents."""

    def __init__(self, embed_dim: int, num_steps: int = MOTION_STEPS, num_modes: int = 6):
        super().__init__()
        self.num_steps = num_steps
        self.num_modes = num_modes
        # Predict K modes × T steps × 2 coords + K mode scores
        self.traj_head = MLP(embed_dim, embed_dim, num_modes * num_steps * 2, 3)
        self.score_head = MLP(embed_dim, embed_dim // 2, num_modes, 2)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, d = queries.shape
        trajs = self.traj_head(queries).reshape(B, N, self.num_modes, self.num_steps, 2)
        scores = self.score_head(queries)
        return trajs, scores


class PlanningHead(nn.Module):
    """Predicts ego future waypoints from an ego planning query."""

    def __init__(self, embed_dim: int, num_steps: int = PLANNING_STEPS, num_modes: int = 3):
        super().__init__()
        self.num_steps = num_steps
        self.num_modes = num_modes
        self.traj_head = MLP(embed_dim, embed_dim, num_modes * num_steps * 2, 3)
        self.score_head = MLP(embed_dim, embed_dim // 2, num_modes, 2)

    def forward(self, ego_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_query: (B, d)
        Returns:
            trajs:  (B, num_modes, num_steps, 2)
            scores: (B, num_modes)
        """
        B, d = ego_query.shape
        trajs = self.traj_head(ego_query).reshape(B, self.num_modes, self.num_steps, 2)
        scores = self.score_head(ego_query)
        return trajs, scores


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------

@HEADS.register_module()
class UnifiedPerceptionDecoder(BaseModule):
    """Transformer decoder that jointly handles detection, mapping,
    ego-status, motion prediction, and ego planning.

    Parameters
    ----------
    embed_dim : int
        Model hidden size.
    num_det_queries : int
        Number of object detection queries.
    num_map_queries : int
        Number of map element queries.
    num_stage1_layers : int
        Number of decoder layers in stage 1 (perception only).
    num_stage2_layers : int
        Number of decoder layers in stage 2 (after VLM injection).
    bev_h, bev_w : int
        BEV grid dimensions.
    tasks : list[str]
        Subset of {'det', 'map', 'ego', 'motion', 'planning'} to enable.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        # mmdet3d configs use embed_dims (plural) — accept both
        embed_dims: Optional[int] = None,
        num_det_queries: int = 900,
        num_map_queries: int = 100,
        num_stage1_layers: int = 3,
        num_stage2_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        bev_h: int = BEV_H_MINI,
        bev_w: int = BEV_W_MINI,
        num_det_classes: int = len(NUSCENES_DETECTION_CLASSES),
        num_map_classes: int = len(NUSCENES_MAP_CLASSES),
        map_num_pts: int = 20,
        num_motion_modes: int = 6,
        num_motion_steps: int = MOTION_STEPS,
        num_plan_modes: int = 3,
        num_plan_steps: int = PLANNING_STEPS,
        tasks: Optional[List[str]] = None,
        # BEV feature input projection
        bev_in_channels: int = 256,
        loss_cls=None,
        loss_bbox=None,
        loss_map_cls=None,
        loss_map_pts=None,
        loss_ego=None,
        loss_motion=None,
        loss_plan=None,
        init_cfg=None,
        encoder=None,
        decoder=None,
        bbox_coder=None,
        positional_encoding=None,
        num_cam=None,
        num_feature_levels=None,
        task_loss_weight=None,
        train_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        # Accept embed_dims (mmdet3d convention) as alias for embed_dim
        if embed_dims is not None:
            embed_dim = embed_dims

        if tasks is None:
            tasks = ["det", "map", "ego", "motion", "planning"]
        self.tasks = tasks
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        # export=True → ONNX-safe code paths; export=False → normal training
        self.export = False

        # Try to build BEVFormerEncoder from mmdetection3d if the config is provided.
        self.bev_encoder = None
        if encoder is not None:
            try:
                from mmdet.models.builder import build_neck
                self.bev_encoder = build_neck(encoder)
            except Exception:
                self.bev_encoder = None

        # BEV feature projection (from backbone output to embed_dim)
        self.bev_proj = nn.Linear(bev_in_channels, embed_dim) if bev_in_channels != embed_dim else nn.Identity()

        # Positional encoding — pe stored as (1, H*W, d), no Unsqueeze in ONNX
        self.bev_pos_enc = BEVPositionEncoding(embed_dim, bev_h, bev_w)

        # Instance queries — stored as (1, N, d) Parameters, no Unsqueeze in ONNX
        self.query_bank = InstanceQueryBank(num_det_queries, num_map_queries, embed_dim)
        # ego_query stored as (1, 1, d) — no Unsqueeze needed in _get_queries
        self.ego_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.ego_query, std=0.02)

        # Stage 1 decoder layers (perception only)
        self.stage1_layers = nn.ModuleList(
            [
                PerceptionDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_stage1_layers)
            ]
        )

        # VLM token projection (stage 2) — projects VLM hidden dim → embed_dim
        self.vlm_proj: Optional[nn.Linear] = None  # built lazily on first call

        # Stage 2 decoder layers (after VLM injection)
        self.stage2_layers = nn.ModuleList(
            [
                PerceptionDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_stage2_layers)
            ]
        )

        # Task-specific heads
        if "det" in tasks:
            self.det_head = DetectionHead(embed_dim, num_det_classes)
        if "map" in tasks:
            self.map_head = MapHead(embed_dim, num_map_classes, map_num_pts)
        if "ego" in tasks:
            self.ego_status_head = EgoStatusHead(embed_dim)
        if "motion" in tasks:
            self.motion_head = MotionHead(embed_dim, num_motion_steps, num_motion_modes)
        if "planning" in tasks:
            self.plan_head = PlanningHead(embed_dim, num_plan_steps, num_plan_modes)

        # Losses (built from config dicts when provided)
        self.loss_cls    = build_loss(loss_cls)    if loss_cls    else None
        self.loss_bbox   = build_loss(loss_bbox)   if loss_bbox   else None
        self.loss_map_cls= build_loss(loss_map_cls)if loss_map_cls else None
        self.loss_map_pts= build_loss(loss_map_pts)if loss_map_pts else None
        self.loss_ego    = build_loss(loss_ego)    if loss_ego    else None
        self.loss_motion = build_loss(loss_motion) if loss_motion  else None
        self.loss_plan   = build_loss(loss_plan)   if loss_plan   else None

    # ------------------------------------------------------------------
    # Core forward passes
    # ------------------------------------------------------------------

    def _prepare_bev(self, bev_features: torch.Tensor) -> torch.Tensor:
        """Project + add positional encoding to BEV feature tensor.

        Args:
            bev_features: (B, C, H, W) or (B, H*W, C)
        Returns:
            (B, H*W, embed_dim)
        """
        if bev_features.dim() == 4:
            B, C, H, W = bev_features.shape
            # --- original (before export flag) ---
            # bev_features = bev_features.flatten(2).permute(0, 2, 1)
            if self.export:
                bev_features = bev_features.view(B, C, int(H) * int(W)).permute(0, 2, 1)
            else:
                bev_features = bev_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        bev_tokens = self.bev_proj(bev_features)
        bev_tokens = self.bev_pos_enc(bev_tokens)
        return bev_tokens

    def _get_queries(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return det, map, and ego queries expanded to batch size.

        query_bank params are (1, N, d) and ego_query is (1, 1, d) so only
        Expand is needed — no Unsqueeze op in the ONNX graph.
        """
        det_q = self.query_bank.get_det().expand(batch_size, -1, -1)  # (B, N_det, d)
        map_q = self.query_bank.get_map().expand(batch_size, -1, -1)  # (B, N_map, d)
        ego_q = self.ego_query.expand(batch_size, -1, -1)             # (B, 1, d)
        return det_q, map_q, ego_q

    def _run_bev_encoder(
        self,
        bev_features: torch.Tensor,
        img_feats: Optional[list] = None,
        img_metas: Optional[list] = None,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run BEVFormerEncoder when available; otherwise pass through."""
        if self.bev_encoder is None or img_feats is None:
            return bev_features

        try:
            B = bev_features.shape[0]
            bev_queries = self.bev_pos_enc.pe.expand(B, -1, -1)  # (B, H*W, C)
            spatial_shapes = torch.as_tensor(
                [[f.shape[-2], f.shape[-1]] for f in img_feats],
                dtype=torch.long,
                device=bev_queries.device,
            )
            level_start_index = torch.cat([
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            ])
            bev_out = self.bev_encoder(
                bev_queries,
                img_feats,
                img_feats,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                bev_pos=None,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            return bev_out  # (B, bev_h*bev_w, C)
        except Exception:
            return bev_features

    def forward_stage1(
        self,
        bev_features: torch.Tensor,
        bev_mask: Optional[torch.Tensor] = None,
        img_feats: Optional[list] = None,
        img_metas: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1 forward — perception only (no VLM)."""
        # --- original (before export flag) ---
        # bev_features = self._run_bev_encoder(bev_features, img_feats, img_metas)
        if self.export:
            pass  # skip BEV encoder — not available during ONNX export
        else:
            bev_features = self._run_bev_encoder(bev_features, img_feats, img_metas)
        bev_tokens = self._prepare_bev(bev_features)
        B = bev_tokens.shape[0]
        det_q, map_q, ego_q = self._get_queries(B)

        # Concatenate all queries into a single sequence
        all_q = torch.cat([det_q, map_q, ego_q], dim=1)  # (B, N_det+N_map+1, d)
        n_det = det_q.shape[1]
        n_map = map_q.shape[1]

        for layer in self.stage1_layers:
            all_q = layer(all_q, bev_tokens, bev_mask)

        det_out = all_q[:, :n_det]                  # (B, N_det, d)
        map_out = all_q[:, n_det : n_det + n_map]   # (B, N_map, d)
        ego_out = all_q[:, -1]                       # (B, d)

        return dict(det=det_out, map=map_out, ego=ego_out, all_queries=all_q)

    def forward_stage2(
        self,
        stage1_outputs: Dict[str, torch.Tensor],
        bev_features: torch.Tensor,
        vlm_tokens: Optional[torch.Tensor] = None,
        bev_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2 forward — optionally inject VLM tokens, then refine."""
        bev_tokens = self._prepare_bev(bev_features)
        all_q = stage1_outputs["all_queries"]  # (B, N_q, d)
        n_det = stage1_outputs["det"].shape[1]
        n_map = stage1_outputs["map"].shape[1]

        if vlm_tokens is not None:
            # --- original (before export flag) ---
            # vlm_hidden = vlm_tokens.shape[-1]
            # if self.vlm_proj is None or self.vlm_proj.in_features != vlm_hidden:
            #     self.vlm_proj = nn.Linear(vlm_hidden, self.embed_dim).to(all_q.device)
            # vlm_projected = self.vlm_proj(vlm_tokens)
            if self.export:
                if self.vlm_proj is not None:
                    vlm_projected = self.vlm_proj(vlm_tokens)
                else:
                    vlm_projected = vlm_tokens
            else:
                vlm_hidden = vlm_tokens.shape[-1]
                if self.vlm_proj is None or self.vlm_proj.in_features != vlm_hidden:
                    self.vlm_proj = nn.Linear(vlm_hidden, self.embed_dim).to(all_q.device)
                vlm_projected = self.vlm_proj(vlm_tokens)
            bev_tokens = torch.cat([vlm_projected, bev_tokens], dim=1)

        for layer in self.stage2_layers:
            all_q = layer(all_q, bev_tokens, bev_mask)

        det_out = all_q[:, :n_det]
        map_out = all_q[:, n_det : n_det + n_map]
        ego_out = all_q[:, -1]

        return dict(det=det_out, map=map_out, ego=ego_out, all_queries=all_q)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, decoder_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run task heads and return raw predictions."""
        preds: Dict[str, torch.Tensor] = {}

        if "det" in self.tasks:
            cls_l, bbox_l = self.det_head(decoder_outputs["det"])
            preds["det_cls"] = cls_l
            preds["det_bbox"] = bbox_l

        if "map" in self.tasks:
            map_cls, map_pts = self.map_head(decoder_outputs["map"])
            preds["map_cls"] = map_cls
            preds["map_pts"] = map_pts

        if "ego" in self.tasks:
            preds["ego_state"] = self.ego_status_head(decoder_outputs["ego"])

        if "motion" in self.tasks:
            m_trajs, m_scores = self.motion_head(decoder_outputs["det"])
            preds["motion_trajs"] = m_trajs
            preds["motion_scores"] = m_scores

        if "planning" in self.tasks:
            p_trajs, p_scores = self.plan_head(decoder_outputs["ego"])
            preds["plan_trajs"] = p_trajs
            preds["plan_scores"] = p_scores

        return preds

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        if "det" in self.tasks and self.loss_cls is not None and gt_labels_3d is not None:
            losses.update(self._det_loss(preds, gt_bboxes_3d, gt_labels_3d))

        if "map" in self.tasks and self.loss_map_cls is not None and gt_map_labels is not None:
            losses.update(self._map_loss(preds, gt_map_labels, gt_map_pts))

        if "ego" in self.tasks and self.loss_ego is not None and ego_status is not None:
            losses.update(self._ego_loss(preds, ego_status))

        if "motion" in self.tasks and self.loss_motion is not None and gt_agent_fut_trajs is not None:
            losses.update(self._motion_loss(preds, gt_agent_fut_trajs, gt_agent_fut_masks))

        if "planning" in self.tasks and self.loss_plan is not None and gt_ego_fut_trajs is not None:
            losses.update(self._plan_loss(preds, gt_ego_fut_trajs, gt_ego_fut_masks))

        return losses

    # ------------------------------------------------------------------
    # Per-task loss helpers
    # ------------------------------------------------------------------

    def _det_loss(self, preds, gt_bboxes_3d, gt_labels_3d):
        det_cls = preds["det_cls"]
        det_reg = preds["det_bbox"]
        B = det_cls.shape[0]
        losses = {}
        total_cls_loss = det_cls.new_zeros(())
        count = 0
        for b in range(B):
            labels = gt_labels_3d[b]
            if labels is None or len(labels) == 0:
                continue
            n_gt = len(labels)
            pred_logits = det_cls[b, :n_gt]
            total_cls_loss = total_cls_loss + F.cross_entropy(pred_logits, labels.long())
            count += 1
        if count > 0:
            losses["loss_det_cls"] = total_cls_loss / count
        else:
            losses["loss_det_cls"] = det_cls.sum() * 0.0
        return losses

    def _map_loss(self, preds, gt_map_labels, gt_map_pts):
        map_cls = preds["map_cls"]
        B = map_cls.shape[0]
        total_loss = map_cls.new_zeros(())
        count = 0
        for b in range(B):
            labels = gt_map_labels[b]
            if labels is None or len(labels) == 0:
                continue
            n_gt = len(labels)
            pred_logits = map_cls[b, :n_gt]
            total_loss = total_loss + F.cross_entropy(pred_logits, labels.long())
            count += 1
        return {"loss_map_cls": total_loss / max(count, 1)}

    def _ego_loss(self, preds, ego_status):
        ego_pred = preds["ego_state"]
        if torch.is_tensor(ego_status):
            target = ego_status.float()
        else:
            target = torch.stack(ego_status).float().to(ego_pred.device)
        return {"loss_ego": F.mse_loss(ego_pred, target[..., : ego_pred.shape[-1]])}

    def _motion_loss(self, preds, gt_fut_trajs, gt_fut_masks):
        trajs = preds["motion_trajs"]
        B, N, K, T, _ = trajs.shape
        gt = gt_fut_trajs
        if isinstance(gt, (list, tuple)):
            total = trajs.new_zeros(())
            count = 0
            for b in range(B):
                g = gt[b]
                if g is None or len(g) == 0:
                    continue
                n_ag = min(g.shape[0], N)
                pred_b = trajs[b, :n_ag]
                g_exp = g[:n_ag].unsqueeze(1).expand_as(pred_b).to(pred_b.device)
                ade = (pred_b - g_exp).norm(dim=-1).mean(dim=-1)
                best_ade = ade.min(dim=-1).values.mean()
                total = total + best_ade
                count += 1
            return {"loss_motion": total / max(count, 1)}
        return {"loss_motion": trajs.sum() * 0.0}

    def _plan_loss(self, preds, gt_ego_fut_trajs, gt_ego_fut_masks):
        trajs = preds["plan_trajs"]
        scores = preds["plan_scores"]
        B, K, T, _ = trajs.shape

        if isinstance(gt_ego_fut_trajs, (list, tuple)):
            gt_stack = torch.stack([
                g.to(trajs.device) if torch.is_tensor(g)
                else trajs.new_zeros(T, 2)
                for g in gt_ego_fut_trajs
            ])
        else:
            gt_stack = gt_ego_fut_trajs.to(trajs.device)

        gt_exp = gt_stack.unsqueeze(1).expand(B, K, T, 2)
        ade = (trajs - gt_exp).norm(dim=-1)
        if gt_ego_fut_masks is not None:
            if isinstance(gt_ego_fut_masks, (list, tuple)):
                mask = torch.stack([
                    m.to(trajs.device) if torch.is_tensor(m)
                    else trajs.new_ones(T)
                    for m in gt_ego_fut_masks
                ])
                ade = ade * mask.unsqueeze(1)
        min_ade = ade.mean(dim=-1).min(dim=-1).values
        return {"loss_plan_ade": min_ade.mean()}

    # ------------------------------------------------------------------
    # Post-processing (inference)
    # ------------------------------------------------------------------

    def post_process(
        self,
        preds: Dict[str, torch.Tensor],
        img_metas: Optional[List[Dict]] = None,
        score_threshold: float = 0.3,
    ) -> List[Dict]:
        B = next(iter(preds.values())).shape[0]
        results = []
        for b in range(B):
            sample = {}

            if "det_cls" in preds:
                cls_scores = preds["det_cls"][b].softmax(dim=-1)
                max_scores, max_labels = cls_scores[:, :-1].max(dim=-1)
                keep = max_scores > score_threshold
                sample["boxes_3d"] = preds["det_bbox"][b][keep].detach().cpu()
                sample["scores_3d"] = max_scores[keep].detach().cpu()
                sample["labels_3d"] = max_labels[keep].detach().cpu()

            if "map_cls" in preds:
                map_scores = preds["map_cls"][b].softmax(dim=-1)
                max_s, max_l = map_scores.max(dim=-1)
                keep_map = max_s > score_threshold
                sample["map_pts"] = preds["map_pts"][b][keep_map].detach().cpu()
                sample["map_labels"] = max_l[keep_map].detach().cpu()
                sample["map_scores"] = max_s[keep_map].detach().cpu()

            if "plan_trajs" in preds:
                p_trajs  = preds["plan_trajs"][b].detach().cpu()
                p_scores = preds["plan_scores"][b].detach().cpu()
                best_mode = p_scores.argmax()
                sample["final_planning"] = p_trajs[best_mode]
                sample["plan_trajs"] = p_trajs
                sample["plan_scores"] = p_scores

            if "motion_trajs" in preds:
                sample["motion_trajs"]  = preds["motion_trajs"][b].detach().cpu()
                sample["motion_scores"] = preds["motion_scores"][b].detach().cpu()

            results.append(sample)
        return results

    # ------------------------------------------------------------------
    # fp16 support
    # ------------------------------------------------------------------

    @auto_fp16(apply_to=("bev_features", "vlm_tokens"))
    def forward(
        self,
        bev_features: torch.Tensor,
        vlm_tokens: Optional[torch.Tensor] = None,
        bev_mask: Optional[torch.Tensor] = None,
        run_stage2: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convenience forward that runs stage1 → (optionally stage2) → predict."""
        s1 = self.forward_stage1(bev_features, bev_mask)
        if run_stage2:
            s2 = self.forward_stage2(s1, bev_features, vlm_tokens, bev_mask)
            return self.predict(s2)
        return self.predict(s1)
