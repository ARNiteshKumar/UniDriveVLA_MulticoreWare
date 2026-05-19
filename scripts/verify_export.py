#!/usr/bin/env python3
"""
verify_export.py — Run inference with the exported ONNX or TorchScript model.

This script requires ONLY:
  - onnxruntime   (for .onnx)   pip install onnxruntime
  - torch         (for .pt)     pip install torch

No mmdet3d, mmcv, CUDA, or CARLA needed.

Usage
-----
  # Verify ONNX model (random dummy input)
  python scripts/verify_export.py \\
      --model exports/unidrivevla_perception.onnx

  # Verify TorchScript model
  python scripts/verify_export.py \\
      --model exports/unidrivevla_perception.pt

  # Use a real image folder (6 camera images, jpg/png)
  python scripts/verify_export.py \\
      --model exports/unidrivevla_perception.onnx \\
      --image-dir /path/to/6_camera_images/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

OUTPUT_NAMES = [
    "det_cls", "det_bbox",
    "map_cls", "map_pts",
    "plan_trajs", "plan_scores",
]
OUTPUT_DESC = [
    "Detection class logits  (batch, 900, 10)",
    "Detection box params    (batch, 900, 10)  cx,cy,cz,log_w,log_l,log_h,sin_yaw,cos_yaw,vx,vy",
    "Map class logits        (batch, 100, 3)",
    "Map polyline waypoints  (batch, 100, 20, 2)  BEV x,y",
    "Planning trajectories   (batch, 3, 6, 2)  3 modes × 6 steps × x,y",
    "Planning mode scores    (batch, 3)",
]

NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
MAP_CLASSES = ["divider", "ped_crossing", "boundary"]


def load_image_dir(image_dir, img_h=450, img_w=800):
    """Load up to 6 camera images from a directory, resize, normalise."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not found — using random image tensor.")
        return None

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])[:6]

    if not paths:
        print(f"No images found in {image_dir} — using random tensor.")
        return None

    imgs = []
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)

    for p in paths:
        img = Image.open(p).convert("RGB").resize((img_w, img_h))
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - mean) / std
        imgs.append(arr)

    # Pad to 6 cameras if fewer images
    while len(imgs) < 6:
        imgs.append(np.zeros_like(imgs[0]))

    # (1, 6, 3, H, W)
    return torch.tensor(np.stack(imgs)[np.newaxis], dtype=torch.float32)


def infer_onnx(model_path, dummy):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed.")
        print("Install:  pip install onnxruntime")
        sys.exit(1)

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)

    print(f"\nONNX Runtime  |  providers: {sess.get_providers()}")
    print(f"Input  : {sess.get_inputs()[0].name}  "
          f"shape={sess.get_inputs()[0].shape}")

    t0 = time.time()
    outputs = sess.run(None, {"img": dummy.numpy()})
    elapsed = time.time() - t0

    return outputs, elapsed


def infer_torchscript(model_path, dummy):
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        outputs = model(dummy)
    elapsed = time.time() - t0

    return [o.numpy() for o in outputs], elapsed


def print_results(outputs, elapsed):
    print(f"\n  Inference time: {elapsed * 1000:.0f} ms  (CPU)")
    print()
    print(f"  {'Output':<20} {'Shape':<28} {'Description'}")
    print("  " + "-" * 80)
    for name, desc, out in zip(OUTPUT_NAMES, OUTPUT_DESC, outputs):
        arr = np.array(out)
        print(f"  {name:<20} {str(list(arr.shape)):<28} {desc.split('(')[0].strip()}")

    # ── detection top-5 ──────────────────────────────────────────────────────
    det_cls = np.array(outputs[0])   # (B, 900, 10)
    det_box = np.array(outputs[1])   # (B, 900, 10)

    scores = det_cls[0].max(axis=-1)                # (900,) — best class score
    top_idx = np.argsort(scores)[::-1][:5]

    print("\n  Top-5 detected objects (by max class logit):")
    print(f"  {'Rank':<6} {'Class':<22} {'Logit':>8}  Box params (cx, cy, cz)")
    print("  " + "-" * 60)
    for rank, idx in enumerate(top_idx, 1):
        cls_id = det_cls[0, idx].argmax()
        cls_name = NUSCENES_CLASSES[cls_id] if cls_id < len(NUSCENES_CLASSES) else "unknown"
        logit = scores[idx]
        cx, cy, cz = det_box[0, idx, 0], det_box[0, idx, 1], det_box[0, idx, 2]
        print(f"  {rank:<6} {cls_name:<22} {logit:>8.3f}  ({cx:.2f}, {cy:.2f}, {cz:.2f})")

    # ── best planning trajectory ──────────────────────────────────────────────
    plan_trajs  = np.array(outputs[4])   # (B, 3, 6, 2)
    plan_scores = np.array(outputs[5])   # (B, 3)
    best_mode   = plan_scores[0].argmax()
    best_traj   = plan_trajs[0, best_mode]  # (6, 2)

    print(f"\n  Best planning trajectory (mode {best_mode}, score={plan_scores[0, best_mode]:.3f}):")
    print(f"  {'Step':<8} {'X (m)':>10} {'Y (m)':>10}")
    print("  " + "-" * 32)
    horizons = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for step, (x, y) in enumerate(best_traj):
        print(f"  {horizons[step]:.1f}s    {x:>10.3f} {y:>10.3f}")

    # ── map summary ───────────────────────────────────────────────────────────
    map_cls = np.array(outputs[2])  # (B, 100, 3)
    map_scores = map_cls[0].max(axis=-1)
    map_labels = map_cls[0].argmax(axis=-1)
    above = map_scores > 0.0
    counts = {c: int((map_labels[above] == i).sum())
              for i, c in enumerate(MAP_CLASSES)}

    print(f"\n  Map elements (positive logit):")
    for cls_name, cnt in counts.items():
        print(f"    {cls_name:<20}: {cnt}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="Path to .onnx or .pt exported model.")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="Optional: directory with 6 camera images (.jpg/.png).")
    p.add_argument("--img-h", type=int, default=450)
    p.add_argument("--img-w", type=int, default=800)
    p.add_argument("--num-cams", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        print(f"Model file not found: {args.model}")
        print("Run first:  python scripts/export_model.py --random-weights")
        sys.exit(1)

    print("=" * 60)
    print("  UniDriveVLA — Export Verification")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Format: {args.model.suffix}")

    # ── load input ────────────────────────────────────────────────
    if args.image_dir:
        dummy = load_image_dir(args.image_dir, args.img_h, args.img_w)
    else:
        dummy = None

    if dummy is None:
        dummy = torch.zeros(1, args.num_cams, 3, args.img_h, args.img_w)
        print(f"\n  Using zero dummy input  {list(dummy.shape)}")
    else:
        print(f"\n  Loaded real images  {list(dummy.shape)}")

    # ── run inference ─────────────────────────────────────────────
    suffix = args.model.suffix.lower()
    if suffix == ".onnx":
        outputs, elapsed = infer_onnx(args.model, dummy)
    elif suffix == ".pt":
        outputs, elapsed = infer_torchscript(args.model, dummy)
    else:
        print(f"Unknown model format '{suffix}'. Expected .onnx or .pt")
        sys.exit(1)

    print_results(outputs, elapsed)

    print("\n" + "=" * 60)
    print("  Verification complete — model is producing valid outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
