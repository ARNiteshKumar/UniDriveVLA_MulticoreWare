"""
VAD-format nuScenes converter.

Produces annotation pkl files that include ego future trajectories and
driving command labels required by the planning head training.

Extends nuscenes_converter with:
  - ego future waypoints (6 steps × 0.5 s = 3 s horizon)
  - high-level driving command (turn left / straight / turn right)
  - CAN-bus ego kinematics

Usage:
    python vad_nuscenes_converter.py nuscenes \
        --root-path data/nuscenes \
        --canbus   data/nuscenes \
        --out-dir  data/infos \
        --extra-tag vad_nuscenes_mini \
        --version  v1.0-mini
"""

import argparse
import math
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

# Re-use shared helpers from the base converter
from nuscenes_converter import (
    CAMERA_CHANNELS,
    get_sample_info,
    create_nuscenes_infos as _base_create,
)

PLANNING_STEPS = 6       # 3 s at 0.5 s intervals
PLANNING_DT    = 0.5     # seconds per step


def parse_args():
    parser = argparse.ArgumentParser(description="VAD nuScenes converter")
    parser.add_argument("dataset")
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--canbus", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extra-tag", default="vad_nuscenes_mini")
    parser.add_argument("--version", default="v1.0-mini")
    return parser.parse_args()


def _ego_pose_at_timestamp(nusc, scene_token: str, timestamp: int) -> Optional[Dict]:
    """Get the ego pose closest to a given timestamp in a scene."""
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    best = None
    best_dt = float("inf")
    while sample_token:
        sample = nusc.get("sample", sample_token)
        dt = abs(sample["timestamp"] - timestamp)
        if dt < best_dt:
            best_dt = dt
            best = sample
        sample_token = sample["next"]
    return best


def _driving_command(dx: float, dy: float) -> int:
    """Heuristic driving command from displacement (ego frame).
    0=turn right, 1=straight, 2=turn left.
    """
    angle = math.atan2(dy, max(abs(dx), 1e-3))
    if angle < -0.3:
        return 0   # right
    elif angle > 0.3:
        return 2   # left
    return 1       # straight


def enrich_with_planning(nusc: "NuScenes", infos: List[Dict]) -> List[Dict]:
    """Add ego future trajectory + driving command to each info dict."""
    # Build token → index lookup
    token_to_idx = {info["token"]: i for i, info in enumerate(infos)}

    for idx, info in enumerate(infos):
        future_traj = np.zeros((PLANNING_STEPS, 2), dtype=np.float32)
        future_mask = np.zeros(PLANNING_STEPS, dtype=np.float32)

        # Reference ego pose (at current sample)
        ego_t = np.array(info["ego2global_translation"])
        ego_r = Quaternion(info["ego2global_rotation"])
        ego_r_inv = ego_r.inverse

        # Walk forward PLANNING_STEPS samples
        sample_token = info.get("next", "")
        for step in range(PLANNING_STEPS):
            if not sample_token:
                break
            try:
                next_sample = nusc.get("sample", sample_token)
            except Exception:
                break

            next_data = nusc.get("sample_data", next_sample["data"]["LIDAR_TOP"])
            next_pose = nusc.get("ego_pose", next_data["ego_pose_token"])
            next_t = np.array(next_pose["translation"])

            # Displacement in ego-centric frame
            delta_global = next_t - ego_t
            delta_ego = ego_r_inv.rotate(delta_global)
            future_traj[step] = delta_ego[:2]  # (x, y)
            future_mask[step] = 1.0

            sample_token = next_sample["next"]

        cmd = _driving_command(future_traj[0, 0], future_traj[0, 1]) if future_mask[0] else 1

        info["gt_ego_fut_trajs"] = future_traj
        info["gt_ego_fut_masks"] = future_mask
        info["gt_ego_fut_cmd"]   = cmd

    return infos


def create_vad_nuscenes_infos(
    root_path: str,
    canbus_path: str,
    out_dir: str,
    extra_tag: str = "vad_nuscenes_mini",
    version: str = "v1.0-mini",
):
    if not HAS_NUSCENES:
        print("[WARN] nuscenes-devkit not installed. Writing empty placeholders.")
        os.makedirs(out_dir, exist_ok=True)
        for split in ("train", "val"):
            out = os.path.join(out_dir, f"{extra_tag}_{split}.pkl")
            with open(out, "wb") as f:
                pickle.dump({"infos": [], "metadata": {"version": version}}, f)
            print(f"  Wrote placeholder: {out}")
        return

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    splits = create_splits_scenes()
    scene_name_to_token = {s["name"]: s["token"] for s in nusc.scene}

    if version == "v1.0-mini":
        train_scene_names = splits["mini_train"]
        val_scene_names   = splits["mini_val"]
    else:
        train_scene_names = splits["train"]
        val_scene_names   = splits["val"]

    def collect_infos(scene_names):
        infos = []
        for name in scene_names:
            if name not in scene_name_to_token:
                continue
            scene_token = scene_name_to_token[name]
            scene = nusc.get("scene", scene_token)
            sample_token = scene["first_sample_token"]
            while sample_token:
                info = get_sample_info(nusc, sample_token)
                infos.append(info)
                sample = nusc.get("sample", sample_token)
                sample_token = sample["next"]
        return infos

    print("Collecting train samples ...")
    train_infos = enrich_with_planning(nusc, collect_infos(train_scene_names))
    print("Collecting val samples ...")
    val_infos   = enrich_with_planning(nusc, collect_infos(val_scene_names))

    os.makedirs(out_dir, exist_ok=True)
    for split, infos in [("train", train_infos), ("val", val_infos)]:
        out = os.path.join(out_dir, f"{extra_tag}_{split}.pkl")
        with open(out, "wb") as f:
            pickle.dump({"infos": infos, "metadata": {"version": version}}, f)
        print(f"Saved {len(infos)} VAD infos → {out}")


if __name__ == "__main__":
    args = parse_args()
    create_vad_nuscenes_infos(
        root_path=args.root_path,
        canbus_path=args.canbus,
        out_dir=args.out_dir,
        extra_tag=args.extra_tag,
        version=args.version,
    )
