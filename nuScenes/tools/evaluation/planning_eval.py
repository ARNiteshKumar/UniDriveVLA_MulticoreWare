"""
Planning Evaluation — L2 displacement and collision rate.

Computes per-timestep and average L2 error between predicted ego
trajectories and ground-truth future waypoints, plus a naive collision
indicator.

Usage:
    python tools/evaluation/planning_eval.py \
        --result-path work_dirs/mini_stage1/results.pkl \
        --gt-path     data/infos/nuscenes_infos_mini_val.pkl \
        --output-dir  work_dirs/mini_stage1/eval
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


PLANNING_STEPS = 6   # 3 s at 0.5 s resolution
COLLISION_RADIUS = 1.5  # metres — simplified ego footprint radius


def parse_args():
    parser = argparse.ArgumentParser(description="Planning evaluation (L2 + collision)")
    parser.add_argument("--result-path", required=True, help="Prediction pkl/json from test.py")
    parser.add_argument("--gt-path", required=True, help="Ground-truth pkl (infos)")
    parser.add_argument("--output-dir", default="work_dirs/eval")
    return parser.parse_args()


def load_results(path: str) -> List[Dict]:
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path) as f:
        return json.load(f)


def load_gt(gt_path: str) -> List[Dict]:
    with open(gt_path, "rb") as f:
        data = pickle.load(f)
    return data.get("infos", data) if isinstance(data, dict) else data


def compute_l2(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """L2 per time-step.

    Args:
        pred: (T, 2)
        gt:   (T, 2)
        mask: (T,) binary mask; if None all steps count
    Returns:
        (T,) array of L2 errors
    """
    diff = pred - gt
    l2 = np.linalg.norm(diff, axis=-1)
    if mask is not None:
        l2 = l2 * mask
    return l2


def is_collision(pred: np.ndarray, gt_agents: List[np.ndarray], radius: float = COLLISION_RADIUS) -> np.ndarray:
    """Naive point-in-radius collision check at each step.

    Args:
        pred:       (T, 2) ego predicted positions
        gt_agents:  list of (T, 2) agent future positions
        radius:     collision threshold in metres
    Returns:
        (T,) bool array
    """
    T = pred.shape[0]
    collisions = np.zeros(T, dtype=bool)
    for agent_traj in gt_agents:
        if agent_traj.shape[0] < T:
            continue
        dist = np.linalg.norm(pred - agent_traj[:T], axis=-1)
        collisions |= dist < radius
    return collisions


def evaluate_planning(results: List[Dict], infos: List[Dict]) -> Dict:
    """Compute planning metrics over all samples.

    Returns dict with per-step L2 and average collision rate.
    """
    n = min(len(results), len(infos))
    per_step_l2 = [[] for _ in range(PLANNING_STEPS)]
    per_step_col = [[] for _ in range(PLANNING_STEPS)]

    for i in range(n):
        res = results[i]
        info = infos[i]

        # Extract predicted trajectory
        img_bbox = res.get("img_bbox", res)
        pred_traj = img_bbox.get("final_planning", None)
        if pred_traj is None:
            continue
        if not isinstance(pred_traj, np.ndarray):
            pred_traj = np.array(pred_traj)
        pred_traj = pred_traj.reshape(-1, 2)[:PLANNING_STEPS]

        # Ground-truth future trajectory from info
        gt_traj = info.get("gt_ego_fut_trajs", None)
        gt_mask = info.get("gt_ego_fut_masks", None)
        if gt_traj is None:
            continue
        if not isinstance(gt_traj, np.ndarray):
            gt_traj = np.array(gt_traj)
        gt_traj = gt_traj.reshape(-1, 2)[:PLANNING_STEPS]

        # Pad pred if shorter than GT
        if pred_traj.shape[0] < PLANNING_STEPS:
            pad = PLANNING_STEPS - pred_traj.shape[0]
            pred_traj = np.concatenate([pred_traj, np.zeros((pad, 2))], axis=0)

        mask = gt_mask[:PLANNING_STEPS] if gt_mask is not None else np.ones(PLANNING_STEPS)

        l2_steps = compute_l2(pred_traj, gt_traj, mask)
        for t in range(PLANNING_STEPS):
            if mask[t] > 0:
                per_step_l2[t].append(l2_steps[t])

        # Collision (dummy: no agent trajectories in mini info)
        gt_agents = []
        for ann in info.get("anns", []):
            fut = ann.get("fut_traj", None)
            if fut is not None:
                gt_agents.append(np.array(fut).reshape(-1, 2))
        col = is_collision(pred_traj, gt_agents)
        for t in range(PLANNING_STEPS):
            if mask[t] > 0:
                per_step_col[t].append(float(col[t]))

    metrics: Dict = {}
    for t in range(PLANNING_STEPS):
        ts = (t + 1) * 0.5
        l2 = float(np.mean(per_step_l2[t])) if per_step_l2[t] else float("nan")
        col = float(np.mean(per_step_col[t])) if per_step_col[t] else float("nan")
        metrics[f"l2_{ts:.1f}s"] = l2
        metrics[f"col_{ts:.1f}s"] = col

    # Aggregate over 1s / 2s / 3s
    for horizon, steps in [("1s", 2), ("2s", 4), ("3s", 6)]:
        valid_l2 = [v for t, v in enumerate(
            metrics.get(f"l2_{(t+1)*0.5:.1f}s", float("nan")) for t in range(steps)
        ) if not np.isnan(v)]
        metrics[f"avg_l2_{horizon}"] = float(np.mean(valid_l2)) if valid_l2 else float("nan")

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  UniDriveVLA — Planning Evaluation")
    print("=" * 60)

    results = load_results(args.result_path)
    infos = load_gt(args.gt_path)

    print(f"  Predictions : {len(results)}")
    print(f"  GT samples  : {len(infos)}")

    metrics = evaluate_planning(results, infos)

    print("\nPer-step results:")
    for k, v in sorted(metrics.items()):
        tag = f"  {k:<20s}: "
        val = f"{v:.4f}" if not np.isnan(v) else "N/A"
        print(tag + val)

    out_path = os.path.join(args.output_dir, "planning_eval.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
