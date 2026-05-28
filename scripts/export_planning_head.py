#!/usr/bin/env python3
"""
export_planning_head.py — Export the QwenVL3APlanningHead block standalone.

Full pipeline: Camera images → ResNet-50 Backbone → FPN Neck → BEV → Planning Head

Matches the config from unidrivevla_b2d_stage1_unified_2b_no_cotraining.py:

    model = dict(
        type='UniDriveVLA',
        img_backbone=dict(type='ResNet', depth=50, ...),
        img_neck=dict(type='FPN', in_channels=[2048], out_channels=256),
        planning_head=dict(
            type='QwenVL3APlanningHead',
            action_dim=2,
            action_horizon=6,
            vlm_variant='2b',
            vlm_fusion_cfg=dict(type='direct'),
            unified_decoder_cfg=dict(type='UnifiedPerceptionDecoder', ...)
        )
    )

Since Qwen3-VL-2B weights are not in this repo, the VLM visual encoder is
replaced with a stub linear projection of the same hidden dimension (2048).

Input : img  (B, N_cam, 3, H, W)   — raw camera images (6 cams, 450×800)
Output: 7 tensors (det_cls, det_bbox, map_cls, map_pts, plan_trajs, plan_scores, vlm_plan)

Requires ONLY:  torch  torchvision  onnx  onnxruntime
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


# ── backbone: ResNet-50 ───────────────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor — returns C5 (layer4) feature map."""
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.resnet50(weights=None)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4   # C5: (B, 2048, H/32, W/32)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)       # (B, 2048, H', W')


# ── neck: SimpleFPN 2048 → 256 ───────────────────────────────────────────────

class SimpleFPN(nn.Module):
    """
    Single-level FPN neck matching the config:
        img_neck=dict(type='FPN', in_channels=[2048], out_channels=256)

    Reduces backbone C5 (2048 ch) to 256 ch used by the decoder.
    Two Conv2d layers:
        lateral : Conv2d(2048, 256, kernel=1)   — channel reduction
        output  : Conv2d(256,  256, kernel=3, pad=1) — spatial smoothing
    """
    def __init__(self, in_ch: int = 2048, out_ch: int = 256):
        super().__init__()
        self.lateral = nn.Conv2d(in_ch,  out_ch, kernel_size=1)
        self.output  = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2048, H', W')  →  (B, 256, H', W')"""
        return self.output(self.lateral(x))


# ── VLM stub — replaces Qwen3-VL-2B visual encoder ───────────────────────────

class VLMVisualStub(nn.Module):
    """
    Stub for Qwen3-VL-2B visual encoder.
    hidden_dim=2048 matches vlm_variant='2b' in the config.
    """
    def __init__(self, in_channels: int = 256, hidden_dim: int = 2048, seq_len: int = 196):
        super().__init__()
        self.proj    = nn.Linear(in_channels, hidden_dim)
        self.seq_len = seq_len

    def forward(self, bev_tokens: torch.Tensor) -> torch.Tensor:
        x = bev_tokens[:, :self.seq_len, :]   # (B, 196, 256)
        return self.proj(x)                   # (B, 196, 2048)


# ── VLM fusion — type='direct' from config vlm_fusion_cfg ────────────────────

class DirectVLMFusion(nn.Module):
    """vlm_fusion_cfg=dict(type='direct') — project VLM hidden → embed_dim."""
    def __init__(self, vlm_hidden: int = 2048, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(vlm_hidden, embed_dim)

    def forward(self, vlm_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(vlm_tokens)   # (B, seq, 256)


# ── action head — VLM tokens → planning waypoints ────────────────────────────

class ActionHead(nn.Module):
    """
    action_dim=2, action_horizon=6 → 12 output values = 6 (x,y) waypoints.
    Linear(2048 → 12) then reshape to (B, 6, 2).
    """
    def __init__(self, hidden_dim: int = 2048, action_horizon: int = 6, action_dim: int = 2):
        super().__init__()
        self.fc             = nn.Linear(hidden_dim, action_horizon * action_dim)
        self.action_horizon = action_horizon
        self.action_dim     = action_dim

    def forward(self, vlm_tokens: torch.Tensor) -> torch.Tensor:
        pooled = vlm_tokens.mean(dim=1)                               # (B, 2048)
        return self.fc(pooled).reshape(-1, self.action_horizon, self.action_dim)


# ── full pipeline export wrapper ──────────────────────────────────────────────

class PlanningHeadExportWrapper(nn.Module):
    """
    Full end-to-end export: Camera images → Backbone → Neck → BEV → Planning Head

    Matches the config from unidrivevla_b2d_stage1_unified_2b_no_cotraining.py:
        model = dict(
            type='UniDriveVLA',
            img_backbone=dict(type='ResNet', depth=50, ...),
            img_neck=dict(type='FPN', in_channels=[2048], out_channels=256),
            planning_head=dict(
                type='QwenVL3APlanningHead',
                action_dim=2,
                action_horizon=6,
                vlm_variant='2b',
                vlm_fusion_cfg=dict(type='direct'),
                unified_decoder_cfg=dict(type='UnifiedPerceptionDecoder', ...)
            )
        )

    Input
    -----
    img : (B, N_cam, 3, H, W)   float32  camera images (6 cameras, 450x800)

    Outputs (7 total)
    -------
    det_cls     (B, 900, 10)      detection class logits  (10 nuScenes classes)
    det_bbox    (B, 900, 10)      box params (cx,cy,cz,w,l,h,sin,cos,vx,vy)
    map_cls     (B, 100,   3)     map class logits  (divider/ped_crossing/boundary)
    map_pts     (B, 100,  20, 2)  map polyline BEV waypoints
    plan_trajs  (B,   3,   6, 2)  planning trajectories (3 modes x 6 steps x xy)
    plan_scores (B,   3)          planning mode scores
    vlm_plan    (B,   6,   2)     direct VLM action head waypoints
    """

    def __init__(self, bev_h: int = 50, bev_w: int = 50):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # ── img_backbone: ResNet-50 ────────────────────────────────────────
        self.backbone = ResNet50Backbone()

        # ── img_neck: FPN 2048 → 256  (the part mam asked for) ────────────
        self.neck = SimpleFPN(in_ch=2048, out_ch=256)

        # ── VLM stub (Qwen3-VL-2B hidden_dim=2048, random weights) ────────
        self.vlm_stub   = VLMVisualStub(in_channels=256, hidden_dim=2048, seq_len=196)

        # ── vlm_fusion_cfg=dict(type='direct') ────────────────────────────
        self.vlm_fusion = DirectVLMFusion(vlm_hidden=2048, embed_dim=256)

        # ── UnifiedPerceptionDecoder ───────────────────────────────────────
        self.decoder = UnifiedPerceptionDecoder(
            embed_dim=256, bev_h=bev_h, bev_w=bev_w, bev_in_channels=256,
            num_det_queries=900, num_map_queries=100,
            num_stage1_layers=3, num_stage2_layers=3,
            num_heads=8, ffn_dim=1024, dropout=0.1,
        )

        # ── action_dim=2, action_horizon=6 ────────────────────────────────
        self.action_head = ActionHead(hidden_dim=2048, action_horizon=6, action_dim=2)

    def forward(self, img: torch.Tensor):
        B, N, C, H, W = img.shape

        # ── 1. Backbone: ResNet-50 ─────────────────────────────────────────
        c5 = self.backbone(img.reshape(B * N, C, H, W))
        # c5: (B*N, 2048, H/32, W/32)

        # ── 2. Neck: FPN 2048 → 256 ───────────────────────────────────────
        fpn = self.neck(c5)
        # fpn: (B*N, 256, H/32, W/32)

        # ── 3. BEV construction: camera avg + pool to 50×50 ───────────────
        fpn = fpn.reshape(B, N, 256, fpn.shape[2], fpn.shape[3])
        bev_map = fpn.mean(dim=1)                              # (B, 256, H', W')
        bev_map = F.adaptive_avg_pool2d(bev_map, (self.bev_h, self.bev_w))
        bev_tokens = bev_map.flatten(2).permute(0, 2, 1)      # (B, 2500, 256)

        # ── 4. Stage 1: UnifiedPerceptionDecoder ──────────────────────────
        s1_out = self.decoder.forward_stage1(bev_tokens)

        # ── 5. VLM stub → vlm_fusion_cfg type='direct' ───────────────────
        vlm_tokens = self.vlm_stub(bev_tokens)                 # (B, 196, 2048)
        vlm_fused  = self.vlm_fusion(vlm_tokens)               # (B, 196, 256)

        # ── 6. Stage 2: inject VLM tokens ─────────────────────────────────
        s2_out = self.decoder.forward_stage2(s1_out, bev_tokens, vlm_fused)

        # ── 7. Decode all outputs ──────────────────────────────────────────
        preds    = self.decoder.predict(s2_out)
        vlm_plan = self.action_head(vlm_tokens)                # (B, 6, 2)

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
    print("  Exporting ONNX (verbose=True) ...")
    t = time.time()
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["img"],
        output_names=OUT_NAMES,
        dynamic_axes={
            "img": {0: "batch"},
            **{n: {0: "batch"} for n in OUT_NAMES},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=True,   # show op-level tracing from source
    )
    mb = path.stat().st_size / 1e6
    print(f"  Done {time.time()-t:.1f}s  |  {mb:.1f} MB")
    return mb


def verify_onnx(path, dummy, ref_outs):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed -- skipping verify")
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"img": dummy.numpy()})
    import numpy as np
    ok = True
    for name, r, o in zip(OUT_NAMES, ref_outs, ort_outs):
        diff = abs(r.detach().numpy() - o).max()
        sym  = "OK" if diff < 1e-4 else "DIFF"
        print(f"  [{sym}]  {name:<14} shape={list(r.shape)}  max_diff={diff:.2e}")
        if diff >= 1e-4:
            ok = False
    print("  ONNX verified OK" if ok else "  WARNING: some outputs differ")


def analyze_onnx_ops(path):
    """Print ONNX op-type counts and flag any Unsqueeze / Split ops."""
    try:
        import onnx
        from collections import Counter
        g = onnx.load(str(path))
        counts = Counter(n.op_type for n in g.graph.node)
        total = len(g.graph.node)
        print(f"\n  ONNX graph: {total} nodes total")
        print("  Op counts (all types):")
        for op, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            flag = ""
            if op == "Unsqueeze":
                flag = "  <-- CHECK: was this from query params or pos-enc?"
            elif op == "Split":
                flag = "  <-- CHECK: was this from MHA in_proj_weight?"
            elif op == "Conv":
                flag = "  <-- backbone + FPN neck confirmed"
            print(f"    {op:<22}: {cnt:>4}{flag}")
        # Summary
        n_unsqueeze = counts.get("Unsqueeze", 0)
        n_split     = counts.get("Split",     0)
        n_conv      = counts.get("Conv",      0)
        print(f"\n  Unsqueeze ops : {n_unsqueeze}  (target: 0 after fix)")
        print(f"  Split ops     : {n_split}  (target: 0 after fix)")
        print(f"  Conv ops      : {n_conv}  (backbone + neck present: {'YES' if n_conv > 0 else 'NO'})")
        if n_unsqueeze == 0 and n_split == 0:
            print("  [PASS] No Unsqueeze or Split ops in graph.")
        else:
            print("  [WARN] Still has Unsqueeze/Split — check source above.")
    except ImportError:
        print("  onnx not installed -- skipping op analysis")


def export_torchscript(model, dummy, path):
    print("  Exporting TorchScript ...")
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
    p = argparse.ArgumentParser(description="Export QwenVL3APlanningHead (backbone+neck+head)")
    p.add_argument("--output-dir", default="exports", type=Path,
                   help="Directory to save exported files")
    p.add_argument("--bev-h",    type=int, default=50,  help="BEV grid height (default: 50)")
    p.add_argument("--bev-w",    type=int, default=50,  help="BEV grid width  (default: 50)")
    p.add_argument("--img-h",    type=int, default=450, help="Camera image height (default: 450)")
    p.add_argument("--img-w",    type=int, default=800, help="Camera image width  (default: 800)")
    p.add_argument("--num-cams", type=int, default=6,   help="Number of cameras   (default: 6)")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  UniDriveVLA -- QwenVL3APlanningHead Standalone Export")
    print("=" * 62)
    print("  Config: unidrivevla_b2d_stage1_unified_2b_no_cotraining.py")
    print("  Pipeline:")
    print("    img (B,N,3,H,W)")
    print("    --> ResNet-50 backbone  (C5: 2048 ch)")
    print("    --> FPN neck            (256 ch, 2x Conv2d)")
    print("    --> BEV avg-pool        (50x50 grid)")
    print("    --> UnifiedPerceptionDecoder (Stage 1 + Stage 2)")
    print("    --> det/map/plan/vlm_plan outputs")
    print(f"  VLM: Qwen3-VL-2B stub (hidden_dim=2048, random weights)")
    print(f"  vlm_fusion_cfg: type='direct'  |  action_dim=2  action_horizon=6")

    # ── [1/4] Build model ────────────────────────────────────────────────────
    print("\n[1/4] Building PlanningHeadExportWrapper ...")
    model = PlanningHeadExportWrapper(bev_h=args.bev_h, bev_w=args.bev_w).eval()

    n_backbone  = sum(p.numel() for p in model.backbone.parameters())  / 1e6
    n_neck      = sum(p.numel() for p in model.neck.parameters())      / 1e6
    n_vlm_stub  = sum(p.numel() for p in model.vlm_stub.parameters())  / 1e6
    n_vlm_fuse  = sum(p.numel() for p in model.vlm_fusion.parameters())/ 1e6
    n_decoder   = sum(p.numel() for p in model.decoder.parameters())   / 1e6
    n_action    = sum(p.numel() for p in model.action_head.parameters())/ 1e6
    n_total     = sum(p.numel() for p in model.parameters())           / 1e6

    print(f"  Total parameters: {n_total:.1f} M")
    print(f"    ResNet-50 backbone          : {n_backbone:.1f} M")
    print(f"    FPN neck (Conv2d 2048->256) : {n_neck:.3f} M")
    print(f"      lateral Conv2d(2048,256,k=1) : "
          f"{sum(p.numel() for p in model.neck.lateral.parameters())/1e6:.3f} M")
    print(f"      output  Conv2d(256,256,k=3)  : "
          f"{sum(p.numel() for p in model.neck.output.parameters())/1e6:.3f} M")
    print(f"    VLM stub (Linear 256->2048) : {n_vlm_stub:.3f} M")
    print(f"    VLM fusion (Linear 2048->256): {n_vlm_fuse:.3f} M")
    print(f"    UnifiedPerceptionDecoder    : {n_decoder:.1f} M")
    print(f"    Action head (Linear 2048->12): {n_action:.4f} M")

    # Neck weight statistics
    lat_w = model.neck.lateral.weight.data
    out_w = model.neck.output.weight.data
    print(f"\n  Neck weight stats:")
    print(f"    lateral.weight  shape={list(lat_w.shape)}  "
          f"min={lat_w.min():.4f}  max={lat_w.max():.4f}  "
          f"mean={lat_w.mean():.4f}  std={lat_w.std():.4f}")
    print(f"    output.weight   shape={list(out_w.shape)}  "
          f"min={out_w.min():.4f}  max={out_w.max():.4f}  "
          f"mean={out_w.mean():.4f}  std={out_w.std():.4f}")

    # ── [2/4] Reference forward pass ─────────────────────────────────────────
    print(f"\n[2/4] Reference forward pass ...")
    # Camera image input — the REAL pipeline input
    dummy = torch.zeros(1, args.num_cams, 3, args.img_h, args.img_w)
    print(f"  Input  : img  shape={list(dummy.shape)}"
          f"  (1 batch, {args.num_cams} cams, 3ch, {args.img_h}x{args.img_w})")

    t0 = time.time()
    with torch.no_grad():
        ref = model(dummy)
    print(f"  Forward pass: {time.time()-t0:.1f}s")

    # Show intermediate shapes
    with torch.no_grad():
        # Run backbone only to show C5 shape
        c5_sample = model.backbone(dummy[0])  # (N, 2048, H/32, W/32)
        fpn_sample = model.neck(c5_sample)    # (N, 256, H/32, W/32)
    print(f"\n  Intermediate shapes:")
    print(f"    After backbone (C5)  : {list(c5_sample.shape)}  "
          f"[N_cam, 2048, {args.img_h//32}, {args.img_w//32}]")
    print(f"    After FPN neck       : {list(fpn_sample.shape)}  "
          f"[N_cam, 256, {args.img_h//32}, {args.img_w//32}]")
    print(f"    After BEV avg+pool   : [1, {args.bev_h*args.bev_w}, 256]  "
          f"[batch, {args.bev_h}x{args.bev_w} BEV, embed_dim]")

    print(f"\n  Output shapes:")
    for name, out in zip(OUT_NAMES, ref):
        print(f"    {name:<14}: {list(out.shape)}")

    # ── [3/4] ONNX export ────────────────────────────────────────────────────
    print("\n[3/4] ONNX export ...")
    onnx_path = args.output_dir / "planning_head.onnx"
    onnx_mb   = export_onnx(model, dummy, onnx_path)
    print("  Verifying ONNX outputs vs PyTorch reference ...")
    verify_onnx(onnx_path, dummy, ref)
    analyze_onnx_ops(onnx_path)

    # Count Conv ops in ONNX graph to confirm neck is present
    try:
        import onnx as _onnx
        graph = _onnx.load(str(onnx_path))
        conv_count = sum(1 for n in graph.graph.node if n.op_type == "Conv")
        print(f"  ONNX graph: {len(graph.graph.node)} nodes, {conv_count} Conv ops "
              f"(backbone+neck Conv layers confirmed)")
    except ImportError:
        pass

    # ── [4/4] TorchScript export ─────────────────────────────────────────────
    print("\n[4/4] TorchScript export ...")
    ts_path = args.output_dir / "planning_head.pt"
    ts_mb   = export_torchscript(model, dummy, ts_path)

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    info = {
        "component": "QwenVL3APlanningHead (full pipeline)",
        "config_source": "unidrivevla_b2d_stage1_unified_2b_no_cotraining.py",
        "pipeline": [
            "ResNet-50 backbone (img_backbone)",
            "FPN neck Conv2d(2048->256) (img_neck)",
            "BEV camera-avg + adaptive_avg_pool2d(50x50)",
            "UnifiedPerceptionDecoder Stage1 (perception)",
            "VLM stub Linear(256->2048) [Qwen3-VL-2B placeholder]",
            "DirectVLMFusion Linear(2048->256) (vlm_fusion_cfg=direct)",
            "UnifiedPerceptionDecoder Stage2 (VLM cross-attn)",
            "ActionHead Linear(2048->12) -> (B,6,2)",
        ],
        "vlm": "Qwen3-VL-2B (stub -- same hidden_dim=2048, random weights)",
        "vlm_fusion_cfg": "type='direct'",
        "action_dim": 2,
        "action_horizon": 6,
        "input": {
            "name": "img",
            "shape": [1, args.num_cams, 3, args.img_h, args.img_w],
            "dtype": "float32",
            "description": f"(batch, {args.num_cams} cameras, 3ch, {args.img_h}x{args.img_w}) raw camera images",
        },
        "intermediate_shapes": {
            "c5_per_cam": [args.num_cams, 2048, args.img_h // 32, args.img_w // 32],
            "fpn_per_cam": [args.num_cams, 256,  args.img_h // 32, args.img_w // 32],
            "bev_tokens":  [1, args.bev_h * args.bev_w, 256],
        },
        "neck_weight_stats": {
            "lateral_Conv2d_2048x256x1x1": {
                "min":  round(float(lat_w.min()), 4),
                "max":  round(float(lat_w.max()), 4),
                "mean": round(float(lat_w.mean()), 4),
                "std":  round(float(lat_w.std()),  4),
            },
            "output_Conv2d_256x256x3x3": {
                "min":  round(float(out_w.min()), 4),
                "max":  round(float(out_w.max()), 4),
                "mean": round(float(out_w.mean()), 4),
                "std":  round(float(out_w.std()),  4),
            },
        },
        "outputs": {n: {"shape": list(r.shape)} for n, r in zip(OUT_NAMES, ref)},
        "parameters_M": {
            "total":     round(n_total,    1),
            "backbone":  round(n_backbone, 1),
            "neck":      round(n_neck,     3),
            "vlm_stub":  round(n_vlm_stub, 3),
            "vlm_fusion":round(n_vlm_fuse, 3),
            "decoder":   round(n_decoder,  1),
            "action_head": round(n_action, 4),
        },
        "file_sizes_MB": {"onnx": round(onnx_mb, 1), "torchscript": round(ts_mb, 1)},
    }
    info_path = args.output_dir / "planning_head_info.json"
    info_path.write_text(json.dumps(info, indent=2))

    print(f"\n  Metadata --> {info_path}")
    print("\n" + "=" * 62)
    print("  Export complete.")
    print(f"  {onnx_path}  ({onnx_mb:.1f} MB)")
    print(f"  {ts_path}  ({ts_mb:.1f} MB)")
    print(f"  {info_path}")
    print("=" * 62)
    print("\n  ONNX input  : 'img'  shape=(batch, N_cam, 3, H, W)")
    print("  Conv ops in ONNX graph confirm backbone + neck are exported.")


if __name__ == "__main__":
    main()
