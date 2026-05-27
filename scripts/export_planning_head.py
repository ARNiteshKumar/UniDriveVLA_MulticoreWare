#!/usr/bin/env python3
"""
export_planning_head.py — Export the QwenVL3APlanningHead block standalone.

Exports the planning head component from unidrivevla_b2d_stage1_unified_2b config:

    model = dict(
        type='UniDriveVLA',
        planning_head=dict(
            type='QwenVL3APlanningHead',
            pretrained_path=vlm_pretrained_path,
            action_dim=2,
            action_horizon=6,
            vlm_fusion_cfg=dict(type='direct'),
            unified_decoder_cfg=...
        )
    )

Since Qwen3-VL-2B weights are not in this repo, the VLM visual encoder is
replaced with a stub linear projection of the same hidden dimension (2048).
This exports the full planning head architecture with random weights.

Requires ONLY:  torch  onnx  onnxruntime
    pip install onnx onnxruntime
    python scripts/export_planning_head.py --output-dir exports/
"""

import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── mock mmcv / mmdet ─────────────────────────────────────────────────────────
def _make_mmcv_mock():
    class _BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()

    mmcv   = types.ModuleType("mmcv")
    runner = types.ModuleType("mmcv.runner")
    runner.BaseModule  = _BaseModule
    runner.auto_fp16   = lambda **kw: (lambda f: f)
    mmcv.runner        = runner
    sys.modules["mmcv"]        = mmcv
    sys.modules["mmcv.runner"] = runner

    mmdet   = types.ModuleType("mmdet")
    builder = types.ModuleType("mmdet.models.builder")
    builder.HEADS      = type("Reg", (), {"register_module": lambda s, **k: (lambda c: c)})()
    builder.build_loss = lambda cfg: None
    mmdet.models         = types.ModuleType("mmdet.models")
    mmdet.models.builder = builder
    sys.modules["mmdet"]                = mmdet
    sys.modules["mmdet.models"]         = mmdet.models
    sys.modules["mmdet.models.builder"] = builder

_make_mmcv_mock()

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── load UnifiedPerceptionDecoder directly (no __init__.py cascade) ───────────
import importlib.util as _ilu

_cspec = _ilu.spec_from_file_location(
    "constants",
    REPO_ROOT / "nuScenes/projects/mmdet3d_plugin/unidrivevla/dense_heads/constants.py",
)
_cmod = _ilu.module_from_spec(_cspec)
sys.modules["constants"] = _cmod
_cspec.loader.exec_module(_cmod)

_dec_src = (
    REPO_ROOT / "nuScenes/projects/mmdet3d_plugin/unidrivevla/dense_heads/unified_perception_decoder.py"
).read_text().replace("from .constants import", "from constants import")
_dec_mod = types.ModuleType("unified_perception_decoder")
exec(compile(_dec_src, "unified_perception_decoder.py", "exec"), _dec_mod.__dict__)
UnifiedPerceptionDecoder = _dec_mod.UnifiedPerceptionDecoder


# ── VLM stub — replaces Qwen3-VL-2B visual encoder ───────────────────────────

class VLMVisualStub(nn.Module):
    """
    Stub for Qwen3-VL-2B visual encoder.

    In the real model:
        6 camera images → Qwen3-VL-2B visual encoder → vlm_tokens (B, seq, 2048)

    Here we use a single Linear projection to simulate the same output shape
    so the rest of the planning head exports cleanly.

    hidden_dim = 2048  (Qwen3-VL-2B hidden size, from config vlm_variant='2b')
    """
    def __init__(self, in_channels: int = 256, hidden_dim: int = 2048, seq_len: int = 196):
        super().__init__()
        self.proj   = nn.Linear(in_channels, hidden_dim)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, bev_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_tokens: (B, 2500, 256)  BEV tokens from neck
        Returns:
            vlm_tokens: (B, seq_len, 2048)  simulated VLM hidden states
        """
        # pool BEV tokens to seq_len, then project to hidden_dim
        B, N, C = bev_tokens.shape
        x = bev_tokens[:, :self.seq_len, :]          # (B, 196, 256)
        return self.proj(x)                          # (B, 196, 2048)


# ── VLM fusion — type='direct' from config vlm_fusion_cfg ────────────────────

class DirectVLMFusion(nn.Module):
    """
    vlm_fusion_cfg = dict(type='direct')
    Directly projects VLM tokens to match decoder embed_dim (256).
    """
    def __init__(self, vlm_hidden: int = 2048, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(vlm_hidden, embed_dim)

    def forward(self, vlm_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(vlm_tokens)   # (B, seq, 256)


# ── action head — projects VLM tokens → waypoints ────────────────────────────

class ActionHead(nn.Module):
    """
    Projects pooled VLM hidden states to planning waypoints.
    action_dim=2, action_horizon=6  → 12 output values = 6 (x,y) waypoints
    """
    def __init__(self, hidden_dim: int = 2048, action_horizon: int = 6, action_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, action_horizon * action_dim)
        self.action_horizon = action_horizon
        self.action_dim     = action_dim

    def forward(self, vlm_tokens: torch.Tensor) -> torch.Tensor:
        pooled = vlm_tokens.mean(dim=1)                        # (B, 2048)
        out    = self.fc(pooled)                               # (B, 12)
        return out.reshape(-1, self.action_horizon, self.action_dim)  # (B, 6, 2)


# ── full planning head export wrapper ─────────────────────────────────────────

class PlanningHeadExportWrapper(nn.Module):
    """
    Standalone export of the QwenVL3APlanningHead block.

    Matches the config from unidrivevla_b2d_stage1_unified_2b_no_cotraining.py:
        planning_head=dict(
            type='QwenVL3APlanningHead',
            action_dim=2,
            action_horizon=6,
            vlm_variant='2b',
            vlm_fusion_cfg=dict(type='direct'),
            unified_decoder_cfg=dict(type='UnifiedPerceptionDecoder', ...)
        )

    Input
    -----
    bev_tokens : (B, 2500, 256)   BEV tokens from backbone + neck

    Outputs
    -------
    det_cls     (B, 900, 10)     detection class logits
    det_bbox    (B, 900, 10)     box params
    map_cls     (B, 100,   3)    map class logits
    map_pts     (B, 100,  20, 2) map polyline waypoints
    plan_trajs  (B,   3,   6, 2) planning trajectories
    plan_scores (B,   3)         planning mode scores
    vlm_plan    (B,   6,   2)    direct VLM → waypoints (action_head output)
    """

    def __init__(self):
        super().__init__()

        # VLM stub (replaces Qwen3-VL-2B visual encoder)
        self.vlm_stub   = VLMVisualStub(in_channels=256, hidden_dim=2048, seq_len=196)

        # vlm_fusion_cfg=dict(type='direct')
        self.vlm_fusion = DirectVLMFusion(vlm_hidden=2048, embed_dim=256)

        # UnifiedPerceptionDecoder (from our dense_heads/)
        self.decoder    = UnifiedPerceptionDecoder(
            embed_dim=256, bev_h=50, bev_w=50, bev_in_channels=256,
            num_det_queries=900, num_map_queries=100,
            num_stage1_layers=3, num_stage2_layers=3,
            num_heads=8, ffn_dim=1024, dropout=0.1,
        )

        # action_dim=2, action_horizon=6
        self.action_head = ActionHead(hidden_dim=2048, action_horizon=6, action_dim=2)

    def forward(self, bev_tokens: torch.Tensor):
        # ── Stage 1: perception decoder on BEV tokens ──────────────────────
        s1_out = self.decoder.forward_stage1(bev_tokens)

        # ── VLM stub: BEV → simulated Qwen3-VL-2B hidden states ────────────
        vlm_tokens = self.vlm_stub(bev_tokens)           # (B, 196, 2048)

        # ── vlm_fusion_cfg type='direct': project VLM → embed_dim ──────────
        vlm_fused  = self.vlm_fusion(vlm_tokens)         # (B, 196, 256)

        # ── Stage 2: inject fused VLM tokens into decoder ──────────────────
        s2_out = self.decoder.forward_stage2(s1_out, bev_tokens, vlm_fused)

        # ── Decode predictions ──────────────────────────────────────────────
        preds = self.decoder.predict(s2_out)

        # ── Action head: VLM tokens → direct waypoint prediction ───────────
        vlm_plan = self.action_head(vlm_tokens)           # (B, 6, 2)

        return (
            preds["det_cls"],
            preds["det_bbox"],
            preds["map_cls"],
            preds["map_pts"],
            preds["plan_trajs"],
            preds["plan_scores"],
            vlm_plan,
        )


# ── export helpers ────────────────────────────────────────────────────────────

OUT_NAMES = [
    "det_cls", "det_bbox",
    "map_cls", "map_pts",
    "plan_trajs", "plan_scores",
    "vlm_plan",
]

def export_onnx(model, dummy, path):
    print("  Exporting ONNX …")
    t = time.time()
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["bev_tokens"],
        output_names=OUT_NAMES,
        dynamic_axes={
            "bev_tokens": {0: "batch"},
            **{n: {0: "batch"} for n in OUT_NAMES},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )
    mb = path.stat().st_size / 1e6
    print(f"  Done {time.time()-t:.1f}s  |  {mb:.1f} MB")
    return mb


def verify_onnx(path, dummy, ref_outs):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping verify")
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"bev_tokens": dummy.numpy()})
    import numpy as np
    ok = True
    for name, r, o in zip(OUT_NAMES, ref_outs, ort_outs):
        diff = abs(r.detach().numpy() - o).max()
        sym  = "✓" if diff < 1e-4 else "✗"
        print(f"  {sym}  {name:<14} shape={list(r.shape)}  max_diff={diff:.2e}")
        if diff >= 1e-4:
            ok = False
    print("  ONNX verified ✓" if ok else "  WARNING: some outputs differ")


def export_torchscript(model, dummy, path):
    print("  Exporting TorchScript …")
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)
    traced.save(str(path))
    mb = path.stat().st_size / 1e6
    print(f"  Done {time.time()-t:.1f}s  |  {mb:.1f} MB")
    return mb


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="exports", type=Path)
    p.add_argument("--bev-h",     type=int, default=50)
    p.add_argument("--bev-w",     type=int, default=50)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  UniDriveVLA — QwenVL3APlanningHead Export")
    print("=" * 60)
    print("  Config source: unidrivevla_b2d_stage1_unified_2b_no_cotraining.py")
    print("  VLM: Qwen3-VL-2B stub (same hidden_dim=2048, random weights)")
    print("  vlm_fusion_cfg: type='direct'")
    print("  action_dim=2  action_horizon=6")

    print("\n[1/4] Building planning head …")
    model   = PlanningHeadExportWrapper().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")
    print(f"    VLM stub (proj 256→2048) : "
          f"{sum(p.numel() for p in model.vlm_stub.parameters())/1e6:.1f} M")
    print(f"    VLM fusion (2048→256)    : "
          f"{sum(p.numel() for p in model.vlm_fusion.parameters())/1e6:.1f} M")
    print(f"    UnifiedPerceptionDecoder : "
          f"{sum(p.numel() for p in model.decoder.parameters())/1e6:.1f} M")
    print(f"    Action head (2048→12)    : "
          f"{sum(p.numel() for p in model.action_head.parameters())/1e6:.3f} M")

    print("\n[2/4] Reference forward pass …")
    # input: BEV tokens (B, bev_h*bev_w, 256)
    dummy = torch.zeros(1, args.bev_h * args.bev_w, 256)
    print(f"  Input shape: {list(dummy.shape)}  (1 batch, {args.bev_h}×{args.bev_w} BEV tokens, 256-dim)")
    t0 = time.time()
    with torch.no_grad():
        ref = model(dummy)
    print(f"  Done {time.time()-t0:.1f}s")
    print("  Output shapes:")
    for name, out in zip(OUT_NAMES, ref):
        print(f"    {name:<14}: {list(out.shape)}")

    print("\n[3/4] ONNX export …")
    onnx_path = args.output_dir / "planning_head.onnx"
    onnx_mb   = export_onnx(model, dummy, onnx_path)
    print("  Verifying …")
    verify_onnx(onnx_path, dummy, ref)

    print("\n[4/4] TorchScript export …")
    ts_path = args.output_dir / "planning_head.pt"
    ts_mb   = export_torchscript(model, dummy, ts_path)

    # metadata
    info = {
        "component": "QwenVL3APlanningHead",
        "config_source": "unidrivevla_b2d_stage1_unified_2b_no_cotraining.py",
        "vlm": "Qwen3-VL-2B (stub — same hidden_dim=2048, random weights)",
        "vlm_fusion_cfg": "type='direct'",
        "action_dim": 2,
        "action_horizon": 6,
        "input": {"name": "bev_tokens", "shape": [1, 2500, 256], "dtype": "float32",
                  "description": "(batch, bev_h*bev_w, embed_dim) BEV tokens from neck"},
        "outputs": {n: {"shape": list(r.shape)} for n, r in zip(OUT_NAMES, ref)},
        "parameters_M": round(n_params, 1),
        "file_sizes_MB": {"onnx": round(onnx_mb, 1), "torchscript": round(ts_mb, 1)},
    }
    info_path = args.output_dir / "planning_head_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"\n  Metadata → {info_path}")

    print("\n" + "=" * 60)
    print("  Export complete.")
    print(f"  {onnx_path}")
    print(f"  {ts_path}")
    print(f"  {info_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
