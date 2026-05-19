#!/usr/bin/env python3
"""
export_model.py — Export UniDriveVLA perception pipeline to ONNX / TorchScript.

Exports:
  backbone (ResNet-50) + neck (FPN) + UnifiedPerceptionDecoder
  → unidrivevla_perception.onnx   (runs on CPU via ONNX Runtime)
  → unidrivevla_perception.pt     (TorchScript, needs only PyTorch)
  → export_info.json              (I/O shapes, param count, file sizes)

Does NOT require CUDA, CARLA, vllm, or mmdet3d at inference time.
Runs on CPU (Intel i7, etc.) in ~5-10 minutes.

Usage
-----
  # With a real checkpoint
  python scripts/export_model.py \\
      --checkpoint checkpoints/stage1/latest.pth \\
      --output-dir exports/

  # Without a checkpoint — random weights (pipeline smoke test)
  python scripts/export_model.py --random-weights --output-dir exports/

  # Export only ONNX or only TorchScript
  python scripts/export_model.py --checkpoint ... --format onnx
  python scripts/export_model.py --checkpoint ... --format torchscript

  # Custom image size / cameras (must match your dataset config)
  python scripts/export_model.py --checkpoint ... \\
      --img-h 450 --img-w 800 --num-cams 6
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── repo paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nuScenes"))
sys.path.insert(0, str(REPO_ROOT / "nuScenes" / "projects"))
sys.path.insert(0, str(REPO_ROOT / "nuScenes" / "projects" / "mmdet3d_plugin"))


# ── wrapper module ────────────────────────────────────────────────────────────

class PerceptionExportWrapper(nn.Module):
    """
    Self-contained wrapper for ONNX / TorchScript tracing.

    Input
    -----
    img : (B, N_cam, 3, H, W)  float32, ImageNet-normalised

    Outputs  (in order)
    -------------------
    det_cls    : (B, 900, 10)      detection class logits
    det_bbox   : (B, 900, 10)      detection box params
                                   (cx,cy,cz,log_w,log_l,log_h,sin_yaw,cos_yaw,vx,vy)
    map_cls    : (B, 100, 3)       map element class logits
    map_pts    : (B, 100, 20, 2)   map polyline BEV waypoints
    plan_trajs : (B, 3, 6, 2)      planning trajectories  (3 modes × 6 steps × xy)
    plan_scores: (B, 3)            planning mode scores
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        bev_h: int = 50,
        bev_w: int = 50,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.decoder = decoder
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, img: torch.Tensor):
        B, N, C, H, W = img.shape

        # ── backbone + neck ──────────────────────────────────────────────────
        img_flat = img.reshape(B * N, C, H, W)
        bb_feats = self.backbone(img_flat)          # tuple of feature maps
        neck_feats = self.neck(bb_feats)            # list of FPN levels

        # ── pseudo-BEV (camera-average + adaptive pool) ───────────────────────
        # Uses finest FPN level; geometrically-correct BEV needs BEVFormerEncoder
        # (requires mmdet3d at runtime — handled by the full training stack).
        feat = neck_feats[0]                        # (B*N, C', H', W')
        feat = feat.reshape(B, N, feat.shape[1],
                            feat.shape[2], feat.shape[3])
        feat = feat.mean(dim=1)                     # (B, C', H', W')
        feat = F.adaptive_avg_pool2d(
            feat, (self.bev_h, self.bev_w))         # (B, C', bev_h, bev_w)
        bev = feat.flatten(2).permute(0, 2, 1)     # (B, bev_h*bev_w, C')

        # ── perception decoder ───────────────────────────────────────────────
        s1_out = self.decoder.forward_stage1(bev)
        preds = self.decoder.predict(s1_out)

        return (
            preds["det_cls"],
            preds["det_bbox"],
            preds["map_cls"],
            preds["map_pts"],
            preds["plan_trajs"],
            preds["plan_scores"],
        )


# ── model building ────────────────────────────────────────────────────────────

def _build_backbone_neck():
    """Build ResNet-50 + FPN matching the stage1 config."""
    try:
        from mmdet.models.builder import build_backbone, build_neck
    except ImportError:
        raise ImportError(
            "mmdet not found. Run:  bash scripts/setup_env.sh\n"
            "Or install manually:   pip install mmdet==2.28.2"
        )

    backbone = build_backbone(dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    ))

    neck = build_neck(dict(
        type="FPN",
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=1,
        relu_before_extra_convs=True,
    ))

    backbone.init_weights()
    return backbone, neck


def _build_decoder():
    """Build UnifiedPerceptionDecoder (our local plugin)."""
    try:
        from unidrivevla.dense_heads.unified_perception_decoder import (
            UnifiedPerceptionDecoder,
        )
    except ImportError:
        # Try alternate import path
        sys.path.insert(0, str(REPO_ROOT / "nuScenes" / "projects"
                                / "mmdet3d_plugin"))
        from unidrivevla.dense_heads.unified_perception_decoder import (
            UnifiedPerceptionDecoder,
        )

    return UnifiedPerceptionDecoder(
        embed_dim=256,
        bev_h=50,
        bev_w=50,
        bev_in_channels=256,
        num_det_queries=900,
        num_map_queries=100,
        num_stage1_layers=3,
        num_stage2_layers=3,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
    )


def build_model(checkpoint_path=None):
    """Assemble full model; optionally load checkpoint weights."""
    print("Building backbone (ResNet-50) ...")
    backbone, neck = _build_backbone_neck()

    print("Building UnifiedPerceptionDecoder ...")
    decoder = _build_decoder()

    model = PerceptionExportWrapper(backbone, neck, decoder)
    model.eval()

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Accept several common checkpoint formats
        state = (
            ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt.get("model_state_dict")
            or ckpt
        )

        # Strip module prefix added by DataParallel / DistributedDataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] {len(missing)} missing keys (expected for partial load)")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys (ignored)")
        print("  Checkpoint loaded.")
    else:
        print("  No checkpoint — using random (initialised) weights.")

    return model


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, dummy_input, out_path, opset=14):
    """Export to ONNX with dynamic batch axis."""
    print(f"\nExporting ONNX  →  {out_path}")
    t0 = time.time()

    output_names = [
        "det_cls", "det_bbox",
        "map_cls", "map_pts",
        "plan_trajs", "plan_scores",
    ]
    dynamic_axes = {
        "img": {0: "batch"},
        "det_cls":     {0: "batch"},
        "det_bbox":    {0: "batch"},
        "map_cls":     {0: "batch"},
        "map_pts":     {0: "batch"},
        "plan_trajs":  {0: "batch"},
        "plan_scores": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["img"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )

    elapsed = time.time() - t0
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Done in {elapsed:.1f}s  |  file size: {size_mb:.1f} MB")
    return size_mb


def verify_onnx(onnx_path, dummy_input, torch_outputs):
    """Run ONNX Runtime and compare against PyTorch outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed — skipping verification.")
        print("         Install with:  pip install onnxruntime")
        return False

    print(f"\nVerifying ONNX output ...")
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])

    ort_inputs = {"img": dummy_input.numpy()}
    ort_outs = sess.run(None, ort_inputs)

    all_close = True
    names = ["det_cls", "det_bbox", "map_cls", "map_pts", "plan_trajs", "plan_scores"]
    for name, pt_out, ort_out in zip(names, torch_outputs, ort_outs):
        pt_arr = pt_out.detach().numpy()
        max_diff = abs(pt_arr - ort_out).max()
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {status}  {name:14s}  shape={list(pt_out.shape)}  "
              f"max_diff={max_diff:.2e}")
        if max_diff >= 1e-4:
            all_close = False

    if all_close:
        print("  All outputs match within tolerance. ONNX export verified.")
    else:
        print("  WARNING: some outputs differ — check opset or model ops.")
    return all_close


# ── TorchScript export ────────────────────────────────────────────────────────

def export_torchscript(model, dummy_input, out_path):
    """Trace model and save as TorchScript."""
    print(f"\nExporting TorchScript  →  {out_path}")
    t0 = time.time()

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input, strict=False)

    traced.save(str(out_path))
    elapsed = time.time() - t0
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Done in {elapsed:.1f}s  |  file size: {size_mb:.1f} MB")
    return size_mb


# ── metadata ──────────────────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_info(out_dir, args, dummy_input, torch_outputs, onnx_mb, ts_mb):
    info = {
        "model": "UniDriveVLA perception pipeline (ResNet-50 + FPN + UnifiedPerceptionDecoder)",
        "checkpoint": str(args.checkpoint) if args.checkpoint else "random weights",
        "export_format": args.format,
        "opset": 14,
        "inputs": {
            "img": {
                "shape": list(dummy_input.shape),
                "description": "(batch, num_cameras, 3, img_h, img_w) float32",
                "normalisation": "ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]",
            }
        },
        "outputs": {
            "det_cls":     {"shape": list(torch_outputs[0].shape), "description": "detection class logits"},
            "det_bbox":    {"shape": list(torch_outputs[1].shape), "description": "detection box params (cx,cy,cz,log_w,log_l,log_h,sin_yaw,cos_yaw,vx,vy)"},
            "map_cls":     {"shape": list(torch_outputs[2].shape), "description": "map element class logits"},
            "map_pts":     {"shape": list(torch_outputs[3].shape), "description": "map polyline BEV waypoints"},
            "plan_trajs":  {"shape": list(torch_outputs[4].shape), "description": "ego planning trajectories (3 modes x 6 steps x xy)"},
            "plan_scores": {"shape": list(torch_outputs[5].shape), "description": "planning mode scores"},
        },
        "config": {
            "backbone": "ResNet-50",
            "neck": "FPN (1 level, 256 ch)",
            "bev_grid": "50 x 50",
            "num_cameras": args.num_cams,
            "image_size": f"{args.img_h} x {args.img_w}",
            "det_queries": 900,
            "map_queries": 100,
            "plan_modes": 3,
            "plan_steps": 6,
        },
        "file_sizes_mb": {},
    }

    if onnx_mb is not None:
        info["file_sizes_mb"]["onnx"] = round(onnx_mb, 1)
    if ts_mb is not None:
        info["file_sizes_mb"]["torchscript"] = round(ts_mb, 1)

    out = out_dir / "export_info.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nExport info saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to .pth checkpoint. Omit to use random weights.")
    p.add_argument("--random-weights", action="store_true",
                   help="Force random weights (ignore checkpoint).")
    p.add_argument("--output-dir", type=Path, default=Path("exports"),
                   help="Directory to write exported files (default: exports/).")
    p.add_argument("--format", choices=["onnx", "torchscript", "both"],
                   default="both", help="Export format (default: both).")
    p.add_argument("--img-h", type=int, default=450,
                   help="Input image height (default: 450 for nuScenes mini).")
    p.add_argument("--img-w", type=int, default=800,
                   help="Input image width  (default: 800 for nuScenes mini).")
    p.add_argument("--num-cams", type=int, default=6,
                   help="Number of cameras  (default: 6 for nuScenes).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Dummy batch size used for tracing (default: 1).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.random_weights:
        args.checkpoint = None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  UniDriveVLA — Model Export")
    print("=" * 60)

    # ── build model ───────────────────────────────────────────────
    ckpt = None if args.random_weights else args.checkpoint
    model = build_model(ckpt)
    model.eval()

    total_params, _ = count_params(model)
    print(f"\n  Parameters: {total_params / 1e6:.1f} M")

    # ── dummy input ───────────────────────────────────────────────
    dummy = torch.zeros(
        args.batch_size, args.num_cams, 3, args.img_h, args.img_w,
        dtype=torch.float32,
    )
    print(f"  Dummy input shape: {list(dummy.shape)}")

    # ── torch forward (reference) ─────────────────────────────────
    print("\nRunning reference forward pass on CPU ...")
    t0 = time.time()
    with torch.no_grad():
        torch_outputs = model(dummy)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n  Output shapes:")
    names = ["det_cls", "det_bbox", "map_cls", "map_pts", "plan_trajs", "plan_scores"]
    for name, out in zip(names, torch_outputs):
        print(f"    {name:14s}: {list(out.shape)}")

    # ── exports ───────────────────────────────────────────────────
    onnx_mb = ts_mb = None

    if args.format in ("onnx", "both"):
        onnx_path = args.output_dir / "unidrivevla_perception.onnx"
        onnx_mb = export_onnx(model, dummy, onnx_path)
        verify_onnx(onnx_path, dummy, torch_outputs)

    if args.format in ("torchscript", "both"):
        ts_path = args.output_dir / "unidrivevla_perception.pt"
        ts_mb = export_torchscript(model, dummy, ts_path)

    # ── metadata ──────────────────────────────────────────────────
    save_info(args.output_dir, args, dummy, torch_outputs, onnx_mb, ts_mb)

    print("\n" + "=" * 60)
    print("  Export complete.")
    print(f"  Output directory: {args.output_dir.resolve()}")
    if args.format in ("onnx", "both"):
        print(f"  ONNX model  : {args.output_dir / 'unidrivevla_perception.onnx'}")
    if args.format in ("torchscript", "both"):
        print(f"  TorchScript : {args.output_dir / 'unidrivevla_perception.pt'}")
    print(f"  Info JSON   : {args.output_dir / 'export_info.json'}")
    print("=" * 60)
    print("\nTo run inference with the ONNX model:")
    print("  python scripts/verify_export.py \\")
    print(f"      --model {args.output_dir / 'unidrivevla_perception.onnx'}")


if __name__ == "__main__":
    main()
