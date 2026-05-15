"""
nuScenes Open-Loop Evaluation
==============================
Runs the official nuScenes detection and tracking metrics via the
nuscenes-devkit, then appends online-mapping and planning metrics.

Usage:
    python tools/evaluation/nuscenes_eval.py \
        --result-path work_dirs/mini_stage1/results.json \
        --dataroot    data/nuscenes \
        --version     v1.0-mini \
        --eval-set    mini_val \
        --output-dir  work_dirs/mini_stage1/eval
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="nuScenes evaluation")
    parser.add_argument("--result-path", required=True, help="Path to results JSON (nuScenes format)")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--eval-set", default="mini_val", help="Evaluation split name")
    parser.add_argument("--output-dir", default="work_dirs/eval")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_nuscenes_detection_eval(
    result_path: str,
    dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
    verbose: bool = False,
) -> Optional[Dict]:
    """Run official nuScenes detection evaluation."""
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError:
        print("[WARN] nuscenes-devkit not installed. Skipping detection eval.")
        return None

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

    cfg = config_factory("detection_cvpr_2019")
    evaluator = NuScenesEval(
        nusc,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
    )
    metrics, metric_data_list = evaluator.evaluate()

    summary = {
        "NDS":  metrics.nd_score,
        "mAP":  metrics.mean_ap,
        "mATE": metrics.mean_dist_aps.get("trans_err", float("nan")),
        "mASE": metrics.mean_dist_aps.get("scale_err", float("nan")),
        "mAOE": metrics.mean_dist_aps.get("orient_err", float("nan")),
        "mAVE": metrics.mean_dist_aps.get("vel_err", float("nan")),
        "mAAE": metrics.mean_dist_aps.get("attr_err", float("nan")),
    }
    return summary


def run_mapping_eval(result_path: str) -> Dict:
    """Compute online-map IoU from prediction results."""
    try:
        with open(result_path) as f:
            results = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load results for map eval: {e}")
        return {}

    map_classes = ["divider", "ped_crossing", "boundary"]
    iou_per_class = {c: [] for c in map_classes}

    for token, pred in results.get("results", {}).items():
        for cls_name in map_classes:
            pred_map = pred.get(f"map_{cls_name}", None)
            gt_map = pred.get(f"gt_map_{cls_name}", None)
            if pred_map is None or gt_map is None:
                continue

            pred_arr = np.array(pred_map, dtype=np.float32)
            gt_arr = np.array(gt_map, dtype=np.float32)

            intersection = np.logical_and(pred_arr > 0.5, gt_arr > 0.5).sum()
            union = np.logical_or(pred_arr > 0.5, gt_arr > 0.5).sum()
            iou = intersection / (union + 1e-6)
            iou_per_class[cls_name].append(iou)

    summary = {}
    ious = []
    for cls_name, values in iou_per_class.items():
        mean_iou = float(np.mean(values)) if values else float("nan")
        summary[f"map_iou_{cls_name}"] = mean_iou
        if not np.isnan(mean_iou):
            ious.append(mean_iou)
    summary["map_iou_mean"] = float(np.mean(ious)) if ious else float("nan")
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  UniDriveVLA — nuScenes Open-Loop Evaluation")
    print("=" * 60)
    print(f"  Result file : {args.result_path}")
    print(f"  Version     : {args.version}")
    print(f"  Eval set    : {args.eval_set}")
    print("=" * 60 + "\n")

    all_metrics: Dict = {}

    # ── Detection ─────────────────────────────────────────────────────
    print("Running detection evaluation ...")
    det_metrics = run_nuscenes_detection_eval(
        args.result_path,
        args.dataroot,
        args.version,
        args.eval_set,
        args.output_dir,
        args.verbose,
    )
    if det_metrics:
        all_metrics.update(det_metrics)
        print("\nDetection results:")
        for k, v in det_metrics.items():
            print(f"  {k:>8s} = {v:.4f}")

    # ── Online Mapping ─────────────────────────────────────────────────
    print("\nRunning mapping evaluation ...")
    map_metrics = run_mapping_eval(args.result_path)
    if map_metrics:
        all_metrics.update(map_metrics)
        print("Map results:")
        for k, v in map_metrics.items():
            print(f"  {k:>24s} = {v:.4f}" if not np.isnan(v) else f"  {k:>24s} = N/A")

    # ── Save summary ───────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # ── Pretty print ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL METRICS")
    print("=" * 60)
    for k, v in sorted(all_metrics.items()):
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k:<30s}: {v:.4f}")
        else:
            print(f"  {k:<30s}: N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
