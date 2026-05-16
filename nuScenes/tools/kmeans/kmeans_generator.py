"""
K-means anchor generator for UniDriveVLA instance query banks.

Reads a nuScenes temporal info PKL file and computes cluster centres for
four task types:
  det    — 3-D bounding-box (x, y) centres in BEV space
  map    — polyline midpoints (x, y) in BEV space
  motion — agent future-trajectory endpoints (x, y) at each step
  plan   — ego future-trajectory endpoints (x, y) at each step

Saves each cluster-centre array as a NumPy .npy file that can be loaded by
the instance query bank inside UnifiedPerceptionDecoder.

Usage
-----
python tools/kmeans/kmeans_generator.py \\
    --info-path data/infos/nuscenes_mini_temporal_train.pkl \\
    --task      det \\
    --num-clusters 900 \\
    --output    data/kmeans/kmeans_det_900_mini.npy
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: str) -> List[dict]:
    """Load a nuScenes temporal info PKL file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Accept both list-of-infos and {'infos': [...]} layouts
    if isinstance(data, dict):
        data = data.get("infos", data.get("data_list", list(data.values())[0]))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected PKL format: {type(data)}")
    return data


def _kmeans_fit(points: np.ndarray, k: int, max_iter: int = 300, tol: float = 1e-4,
                random_state: int = 42) -> np.ndarray:
    """
    Pure-NumPy K-means (Lloyd's algorithm).
    Falls back to scipy if available for speed/stability.

    Parameters
    ----------
    points : (N, D) float32 array
    k      : number of clusters
    Returns cluster centres of shape (k, D).
    """
    if len(points) == 0:
        raise ValueError("No data points provided.")
    if len(points) < k:
        # Pad centres by repeating if fewer samples than clusters
        centres = np.tile(points, (k // len(points) + 1, 1))[:k]
        return centres.astype(np.float32)

    try:
        from scipy.cluster.vq import kmeans2  # type: ignore
        centres, _ = kmeans2(points.astype(np.float64), k, iter=max_iter,
                             minit="points", seed=random_state)
        return centres.astype(np.float32)
    except ImportError:
        pass

    # Fallback: pure NumPy K-means++
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(points))
    centres = [points[idx]]
    for _ in range(1, k):
        dists = np.min(np.stack([np.sum((points - c) ** 2, axis=-1) for c in centres]), axis=0)
        probs = dists / dists.sum()
        centres.append(points[rng.choice(len(points), p=probs)])
    centres = np.array(centres, dtype=np.float64)

    for _ in range(max_iter):
        dists = np.stack([np.sum((points - c) ** 2, axis=-1) for c in centres])
        labels = np.argmin(dists, axis=0)
        new_centres = np.array([
            points[labels == j].mean(axis=0) if (labels == j).any() else centres[j]
            for j in range(k)
        ])
        if np.max(np.abs(new_centres - centres)) < tol:
            break
        centres = new_centres

    return centres.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _extract_det(infos: List[dict]) -> np.ndarray:
    """Extract ground-truth box (x, y) centres in BEV for detection."""
    pts = []
    for info in infos:
        boxes = info.get("gt_boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        boxes = np.array(boxes)
        if boxes.ndim == 2 and boxes.shape[1] >= 2:
            pts.append(boxes[:, :2])
    if not pts:
        print("[WARN] No GT boxes found — returning zero anchors.")
        return np.zeros((1, 2), dtype=np.float32)
    return np.concatenate(pts, axis=0).astype(np.float32)


def _extract_map(infos: List[dict]) -> np.ndarray:
    """Extract polyline midpoints (x, y) for map element anchors."""
    pts = []
    for info in infos:
        gt_map = info.get("gt_map_pts", info.get("map_pts", None))
        if gt_map is None:
            continue
        for poly in gt_map:
            poly = np.array(poly)
            if poly.ndim == 2 and poly.shape[1] >= 2:
                # Use midpoint of each polyline
                mid = poly[len(poly) // 2, :2]
                pts.append(mid)
    if not pts:
        print("[WARN] No map polylines found — returning zero anchors.")
        return np.zeros((1, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def _extract_motion(infos: List[dict]) -> np.ndarray:
    """Extract agent future-trajectory endpoints (x, y) for motion anchors."""
    pts = []
    for info in infos:
        fut = info.get("gt_agent_fut_trajs", None)
        if fut is None:
            continue
        fut = np.array(fut)
        # fut shape: (N_agents, T, 2) or (N_agents, T*2)
        if fut.ndim == 3 and fut.shape[-1] >= 2:
            # Take each future step across all agents
            pts.append(fut[:, :, :2].reshape(-1, 2))
        elif fut.ndim == 2 and fut.shape[-1] % 2 == 0:
            pts.append(fut.reshape(-1, 2))
    if not pts:
        print("[WARN] No agent future trajectories found — returning zero anchors.")
        return np.zeros((1, 2), dtype=np.float32)
    return np.concatenate(pts, axis=0).astype(np.float32)


def _extract_plan(infos: List[dict]) -> np.ndarray:
    """Extract ego future-trajectory points (x, y) for planning anchors."""
    pts = []
    for info in infos:
        traj = info.get("gt_ego_fut_trajs", info.get("sdc_planning", None))
        if traj is None:
            continue
        traj = np.array(traj)
        # traj shape: (T, 2) or (T*2,)
        if traj.ndim == 2 and traj.shape[-1] >= 2:
            pts.append(traj[:, :2])
        elif traj.ndim == 1 and traj.shape[0] % 2 == 0:
            pts.append(traj.reshape(-1, 2))
    if not pts:
        print("[WARN] No ego future trajectories found — returning zero anchors.")
        return np.zeros((1, 2), dtype=np.float32)
    return np.concatenate(pts, axis=0).astype(np.float32)


EXTRACTOR_MAP = {
    "det":    _extract_det,
    "map":    _extract_map,
    "motion": _extract_motion,
    "plan":   _extract_plan,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-means anchor generator for UniDriveVLA query banks",
    )
    parser.add_argument(
        "--info-path",
        required=True,
        help="Path to nuScenes temporal info PKL file (e.g. nuscenes_mini_temporal_train.pkl)",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(EXTRACTOR_MAP.keys()),
        help="Task type: det | map | motion | plan",
    )
    parser.add_argument(
        "--num-clusters", "-k",
        type=int,
        required=True,
        help="Number of k-means cluster centres",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for .npy anchor file",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum k-means iterations (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for k-means initialisation (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[kmeans_generator] task={args.task}, k={args.num_clusters}")
    print(f"  info-path : {args.info_path}")
    print(f"  output    : {args.output}")

    # 1. Load info PKL
    print("Loading PKL ...")
    infos = _load_pkl(args.info_path)
    print(f"  Loaded {len(infos)} samples.")

    # 2. Extract relevant data points
    extractor = EXTRACTOR_MAP[args.task]
    points = extractor(infos)
    print(f"  Extracted {len(points)} data points of shape {points.shape}")

    # 3. Run k-means
    k = min(args.num_clusters, len(points))
    if k < args.num_clusters:
        print(f"  [WARN] Fewer points ({len(points)}) than clusters ({args.num_clusters}). "
              f"Using k={k}.")
    print(f"  Running k-means with k={k}, max_iter={args.max_iter} ...")
    centres = _kmeans_fit(points, k=k, max_iter=args.max_iter, random_state=args.seed)
    print(f"  Cluster centres shape: {centres.shape}")

    # 4. Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, centres)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
