"""
nuScenes → mmdet3d info file converter.

Supports both v1.0 (full) and v1.0-mini datasets.  Produces
nuscenes_infos_{split}.pkl files consumed by NuScenesMiniDataset.

Usage:
    python nuscenes_converter.py nuscenes \
        --root-path data/nuscenes \
        --canbus   data/nuscenes \
        --out-dir  data/infos \
        --extra-tag nuscenes_mini \
        --version  v1.0-mini
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False
    print("[WARN] nuscenes-devkit not installed; converter will be a no-op.")


def parse_args():
    parser = argparse.ArgumentParser(description="nuScenes to mmdet3d format converter")
    parser.add_argument("dataset", choices=["nuscenes"], help="Dataset name")
    parser.add_argument("--root-path", required=True, help="nuScenes dataset root")
    parser.add_argument("--canbus", required=True, help="CAN bus root (same as root usually)")
    parser.add_argument("--out-dir", required=True, help="Output directory for pkl files")
    parser.add_argument("--extra-tag", default="nuscenes", help="Prefix for output files")
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        choices=["v1.0", "v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="Dataset version",
    )
    return parser.parse_args()


def get_available_scenes(nusc: "NuScenes") -> List[Dict]:
    available = []
    for scene in nusc.scene:
        token = scene["token"]
        scene_rec = nusc.get("scene", token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        has_lidarseg = any(
            nusc.get("sample_data", sd["token"])["fileformat"] == "pcd"
            for sd in nusc.get("sample", sample_rec["token"])["data"].values()
            if nusc.get("sample_data", sd["token"])["channel"] == "LIDAR_TOP"
        ) if False else True  # simplification
        if has_lidarseg:
            available.append(scene)
    return available


def _get_sensor2lidar(nusc, cam_data, lidar_token):
    """Compute cam-to-lidar extrinsics."""
    cam_cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    cam_R = Quaternion(cam_cs["rotation"]).rotation_matrix
    cam_t = np.array(cam_cs["translation"])
    lid_R = Quaternion(lidar_cs["rotation"]).rotation_matrix
    lid_t = np.array(lidar_cs["translation"])

    sensor2ego_R = cam_R
    sensor2ego_t = cam_t
    ego2lidar_R = lid_R.T
    ego2lidar_t = -lid_R.T @ lid_t

    sensor2lidar_R = ego2lidar_R @ sensor2ego_R
    sensor2lidar_t = ego2lidar_R @ sensor2ego_t + ego2lidar_t
    return sensor2lidar_R, sensor2lidar_t


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
    """Compute sensor-to-global transformation matrix."""
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec.get("camera_intrinsic", []))

    l2e_r = Quaternion(cs_rec["rotation"]).rotation_matrix
    l2e_t_sensor = np.array(cs_rec["translation"])

    sensor2ego = np.eye(4)
    sensor2ego[:3, :3] = l2e_r
    sensor2ego[:3, 3] = l2e_t_sensor

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_rec["rotation"]).rotation_matrix
    ego2global[:3, 3] = np.array(pose_rec["translation"])

    sensor2global = ego2global @ sensor2ego

    return dict(
        data_path=sd_rec["filename"],
        type=sd_rec["channel"],
        sample_data_token=sd_rec["token"],
        sensor2ego_translation=cs_rec["translation"],
        sensor2ego_rotation=cs_rec["rotation"],
        ego2global_translation=pose_rec["translation"],
        ego2global_rotation=pose_rec["rotation"],
        timestamp=sd_rec["timestamp"],
        sensor2lidar_rotation=l2e_r.T.tolist(),
        sensor2lidar_translation=(-l2e_r.T @ l2e_t_sensor).tolist(),
        cam_intrinsic=camera_intrinsic.tolist() if len(camera_intrinsic) > 0 else [],
    )


CAMERA_CHANNELS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def get_sample_info(nusc, sample_token: str) -> Dict:
    """Extract all relevant info for one sample."""
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    l2e_r_mat = Quaternion(lidar_cs["rotation"]).rotation_matrix
    l2e_t = np.array(lidar_cs["translation"])
    e2g_r_mat = Quaternion(lidar_pose["rotation"]).rotation_matrix
    e2g_t = np.array(lidar_pose["translation"])

    cams = {}
    for cam_ch in CAMERA_CHANNELS:
        if cam_ch not in sample["data"]:
            continue
        cam_token = sample["data"][cam_ch]
        cams[cam_ch] = obtain_sensor2top(
            nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat
        )

    # 3D annotations
    anns = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cat = ann["category_name"]
        velocity = nusc.box_velocity(ann_token)
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)

        anns.append(
            dict(
                category_name=cat,
                translation=ann["translation"],
                size=ann["size"],
                rotation=ann["rotation"],
                velocity=velocity[:2].tolist(),
                token=ann_token,
                num_lidar_pts=ann["num_lidar_pts"],
                num_radar_pts=ann["num_radar_pts"],
            )
        )

    return dict(
        token=sample_token,
        timestamp=sample["timestamp"],
        scene_token=sample["scene_token"],
        prev=sample["prev"],
        next=sample["next"],
        cams=cams,
        lidar_path=lidar_data["filename"],
        ego2global_translation=lidar_pose["translation"],
        ego2global_rotation=lidar_pose["rotation"],
        anns=anns,
    )


def create_nuscenes_infos(
    root_path: str,
    out_dir: str,
    extra_tag: str = "nuscenes_mini",
    version: str = "v1.0-mini",
    canbus_path: Optional[str] = None,
):
    if not HAS_NUSCENES:
        print("[WARN] nuscenes-devkit not available. Writing empty placeholder pkls.")
        os.makedirs(out_dir, exist_ok=True)
        for split in ("train", "val"):
            out = os.path.join(out_dir, f"{extra_tag}_{split}.pkl")
            with open(out, "wb") as f:
                pickle.dump({"infos": [], "metadata": {"version": version}}, f)
            print(f"  Wrote placeholder: {out}")
        return

    print(f"Loading nuScenes {version} from {root_path} ...")
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    splits = create_splits_scenes()
    if version == "v1.0-mini":
        train_scenes = splits["mini_train"]
        val_scenes = splits["mini_val"]
    else:
        train_scenes = splits["train"]
        val_scenes = splits["val"]

    scene_name_to_token = {s["name"]: s["token"] for s in nusc.scene}

    def get_infos_for_scenes(scene_names):
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

    print("Processing train split ...")
    train_infos = get_infos_for_scenes(train_scenes)
    print(f"  {len(train_infos)} samples")

    print("Processing val split ...")
    val_infos = get_infos_for_scenes(val_scenes)
    print(f"  {len(val_infos)} samples")

    os.makedirs(out_dir, exist_ok=True)

    for split, infos in [("train", train_infos), ("val", val_infos)]:
        out_path = os.path.join(out_dir, f"{extra_tag}_{split}.pkl")
        data = {"infos": infos, "metadata": {"version": version}}
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(infos)} infos to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    create_nuscenes_infos(
        root_path=args.root_path,
        out_dir=args.out_dir,
        extra_tag=args.extra_tag,
        version=args.version,
        canbus_path=args.canbus,
    )
