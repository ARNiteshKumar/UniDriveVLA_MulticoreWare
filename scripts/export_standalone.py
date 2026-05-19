#!/usr/bin/env python3
"""
export_standalone.py — Generate ONNX + TorchScript exports.

Requires ONLY:  torch  torchvision  onnx  onnxruntime
No mmdet, mmcv, CUDA, or CARLA needed.

    pip install torchvision onnx onnxruntime
    python scripts/export_standalone.py --output-dir exports/
"""

import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── mock mmcv so our plugin module can be imported without it ─────────────────
def _make_mmcv_mock():
    # BaseModule must accept init_cfg= and ignore it
    class _BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()

    mmcv = types.ModuleType("mmcv")
    runner = types.ModuleType("mmcv.runner")
    runner.BaseModule = _BaseModule
    runner.auto_fp16 = lambda **kw: (lambda f: f)   # no-op decorator
    mmcv.runner = runner
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.runner"] = runner
    # stub mmdet.models.builder
    mmdet = types.ModuleType("mmdet")
    builder = types.ModuleType("mmdet.models.builder")
    builder.HEADS = type("Reg", (), {"register_module": lambda s, **k: (lambda c: c)})()
    builder.build_loss = lambda cfg: None
    mmdet.models = types.ModuleType("mmdet.models")
    mmdet.models.builder = builder
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.models"] = mmdet.models
    sys.modules["mmdet.models.builder"] = builder

_make_mmcv_mock()

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load UnifiedPerceptionDecoder directly from its source file,
# bypassing the plugin __init__.py which would trigger mmdet imports.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "unified_perception_decoder",
    REPO_ROOT / "nuScenes" / "projects" / "mmdet3d_plugin"
             / "unidrivevla" / "dense_heads" / "unified_perception_decoder.py",
)
# Also need to make the constants sub-module importable
_cspec = _ilu.spec_from_file_location(
    "constants",
    REPO_ROOT / "nuScenes" / "projects" / "mmdet3d_plugin"
             / "unidrivevla" / "dense_heads" / "constants.py",
)
_cmod = _ilu.module_from_spec(_cspec)
sys.modules["constants"] = _cmod
_cspec.loader.exec_module(_cmod)

# Patch the decoder module's import of .constants to find our loaded version
_dec_src = (
    REPO_ROOT / "nuScenes" / "projects" / "mmdet3d_plugin"
    / "unidrivevla" / "dense_heads" / "unified_perception_decoder.py"
).read_text()
_dec_src = _dec_src.replace("from .constants import", "from constants import")
_dec_mod = types.ModuleType("unified_perception_decoder")
exec(compile(_dec_src, "unified_perception_decoder.py", "exec"), _dec_mod.__dict__)
UnifiedPerceptionDecoder = _dec_mod.UnifiedPerceptionDecoder


# ── backbone: torchvision ResNet-50 ──────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor — returns layer4 feature map."""

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.resnet50(weights=None)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4   # output: (B, 2048, H/32, W/32)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return (x,)   # tuple so FPN receives iterable


# ── neck: simple 1-level FPN 2048 → 256 ──────────────────────────────────────

class SimpleFPN(nn.Module):
    def __init__(self, in_ch: int = 2048, out_ch: int = 256):
        super().__init__()
        self.lateral = nn.Conv2d(in_ch,  out_ch, 1)
        self.output  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, feats):
        x = feats[-1]
        return [self.output(self.lateral(x))]


# ── export wrapper ────────────────────────────────────────────────────────────

class PerceptionExportWrapper(nn.Module):
    """
    End-to-end perception forward pass.

    Input
    -----
    img : (B, N_cam, 3, H, W)  float32  ImageNet-normalised

    Outputs
    -------
    det_cls    (B, 900, 10)      detection class logits
    det_bbox   (B, 900, 10)      box params (cx cy cz log_w log_l log_h sin cos vx vy)
    map_cls    (B, 100,   3)     map element class logits
    map_pts    (B, 100,  20, 2)  map polyline BEV waypoints
    plan_trajs (B,   3,   6, 2)  planning trajectories  (3 modes × 6 steps × xy)
    plan_scores(B,   3)          planning mode scores
    """

    def __init__(self, backbone, neck, decoder, bev_h=50, bev_w=50):
        super().__init__()
        self.backbone = backbone
        self.neck     = neck
        self.decoder  = decoder
        self.bev_h    = bev_h
        self.bev_w    = bev_w

    def forward(self, img: torch.Tensor):
        B, N, C, H, W = img.shape

        # backbone + neck (per-camera)
        feats = self.neck(self.backbone(img.reshape(B * N, C, H, W)))

        # pseudo-BEV: camera-average + adaptive pool to (bev_h, bev_w)
        f = feats[0].reshape(B, N, feats[0].shape[1],
                             feats[0].shape[2], feats[0].shape[3])
        f = f.mean(1)                                      # (B, C', H', W')
        f = F.adaptive_avg_pool2d(f, (self.bev_h, self.bev_w))
        bev = f.flatten(2).permute(0, 2, 1)               # (B, 2500, C')

        preds = self.decoder.predict(self.decoder.forward_stage1(bev))
        return (
            preds["det_cls"],
            preds["det_bbox"],
            preds["map_cls"],
            preds["map_pts"],
            preds["plan_trajs"],
            preds["plan_scores"],
        )


# ── build ─────────────────────────────────────────────────────────────────────

def build_model(checkpoint=None):
    backbone = ResNet50Backbone()
    neck     = SimpleFPN(2048, 256)
    decoder  = UnifiedPerceptionDecoder(
        embed_dim=256, bev_h=50, bev_w=50, bev_in_channels=256,
        num_det_queries=900, num_map_queries=100,
        num_stage1_layers=3, num_stage2_layers=3,
        num_heads=8, ffn_dim=1024, dropout=0.1,
    )
    model = PerceptionExportWrapper(backbone, neck, decoder).eval()

    if checkpoint:
        ckpt  = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        miss, unex = model.load_state_dict(state, strict=False)
        print(f"  Checkpoint loaded  ({len(miss)} missing, {len(unex)} unexpected keys)")
    else:
        print("  Random (initialised) weights")
    return model


# ── export helpers ────────────────────────────────────────────────────────────

OUT_NAMES = ["det_cls","det_bbox","map_cls","map_pts","plan_trajs","plan_scores"]

def export_onnx(model, dummy, path):
    print(f"  Exporting ONNX …")
    t = time.time()
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["img"],
        output_names=OUT_NAMES,
        dynamic_axes={"img": {0: "batch"}, **{n: {0: "batch"} for n in OUT_NAMES}},
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
    ort_outs = sess.run(None, {"img": dummy.numpy()})
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
    print(f"  Exporting TorchScript …")
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)
    traced.save(str(path))
    mb = path.stat().st_size / 1e6
    print(f"  Done {time.time()-t:.1f}s  |  {mb:.1f} MB")
    return mb


def save_info(out_dir, ref_outs, dummy, onnx_mb, ts_mb):
    info = {
        "model": "UniDriveVLA — ResNet-50 + FPN + UnifiedPerceptionDecoder",
        "weights": "random initialisation (no checkpoint)",
        "export_format": "onnx + torchscript",
        "onnx_opset": 14,
        "input": {"name": "img",
                  "shape": list(dummy.shape),
                  "dtype": "float32",
                  "description": "(batch, num_cameras, 3, H, W) ImageNet-normalised"},
        "outputs": {
            n: {"shape": list(r.shape), "dtype": "float32"}
            for n, r in zip(OUT_NAMES, ref_outs)
        },
        "architecture": {
            "backbone":  "ResNet-50 (torchvision)",
            "neck":      "FPN 2048→256, 1 level",
            "bev_grid":  "50×50",
            "num_cameras": dummy.shape[1],
            "image_size": f"{dummy.shape[3]}×{dummy.shape[4]}",
            "det_queries": 900,
            "map_queries": 100,
            "plan_modes": 3,
            "plan_steps": 6,
        },
        "total_parameters_M": round(
            sum(p.numel() for p in __import__('gc').get_objects()
                if isinstance(p, torch.Tensor) and p.requires_grad) / 1e6, 1),
        "file_sizes_MB": {"onnx": round(onnx_mb,1), "torchscript": round(ts_mb,1)},
    }
    # count params properly
    info["total_parameters_M"] = "see export_info.json"
    p_out = out_dir / "export_info.json"
    p_out.write_text(json.dumps(info, indent=2))
    print(f"  Metadata saved → {p_out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="exports", type=Path)
    p.add_argument("--img-h",    type=int, default=450)
    p.add_argument("--img-w",    type=int, default=800)
    p.add_argument("--num-cams", type=int, default=6)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print("  UniDriveVLA — Standalone Model Export")
    print("=" * 58)

    print("\n[1/4] Building model …")
    model = build_model(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")

    print("\n[2/4] Reference forward pass (CPU) …")
    dummy = torch.zeros(1, args.num_cams, 3, args.img_h, args.img_w)
    t0 = time.time()
    with torch.no_grad():
        ref = model(dummy)
    print(f"  Done {time.time()-t0:.1f}s")
    print("  Output shapes:")
    for name, out in zip(OUT_NAMES, ref):
        print(f"    {name:<14}: {list(out.shape)}")

    print("\n[3/4] ONNX export …")
    onnx_path = args.output_dir / "unidrivevla_perception.onnx"
    onnx_mb   = export_onnx(model, dummy, onnx_path)
    print("  Verifying …")
    verify_onnx(onnx_path, dummy, ref)

    print("\n[4/4] TorchScript export …")
    ts_path = args.output_dir / "unidrivevla_perception.pt"
    ts_mb   = export_torchscript(model, dummy, ts_path)

    save_info(args.output_dir, ref, dummy, onnx_mb, ts_mb)

    print("\n" + "=" * 58)
    print("  Export complete.")
    print(f"  {onnx_path}")
    print(f"  {ts_path}")
    print(f"  {args.output_dir / 'export_info.json'}")
    print("=" * 58)


if __name__ == "__main__":
    main()
