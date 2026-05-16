"""DriveBench evaluation — per-corruption accuracy, mPC, rPC.

mPC  = mean Performance under Corruption
rPC  = relative Performance under Corruption (vs clean)

Usage:
    python vqa_evaluation/DriveBench/eval_drivebench.py \
        --pred_path results/drivebench/preds.json \
        --data_root data/DriveBench \
        --output    results/drivebench/scores.json
"""
import argparse, json, os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output",    default="drivebench_scores.json")
    return p.parse_args()


def exact_match(pred, gt):
    return str(pred).lower().strip() == str(gt).lower().strip()


def main():
    args = parse_args()
    with open(args.pred_path) as f:
        preds = json.load(f)

    # Group by corruption type
    by_corruption = defaultdict(list)
    for item in preds:
        ct = item.get("corruption_type", "clean")
        by_corruption[ct].append(item)

    per_corruption = {}
    for ct, items in by_corruption.items():
        correct = sum(exact_match(it["prediction"], it["gt_answer"]) for it in items)
        per_corruption[ct] = correct / len(items) * 100 if items else 0.0

    clean_acc  = per_corruption.get("clean", 0.0)
    corrupt_accs = [v for k, v in per_corruption.items() if k != "clean"]

    mPC = sum(corrupt_accs) / len(corrupt_accs) if corrupt_accs else 0.0
    rPC = mPC / clean_acc if clean_acc > 0 else 0.0

    results = {
        "clean_accuracy": clean_acc,
        "mPC":            mPC,
        "rPC":            rPC,
        "per_corruption": per_corruption,
        "num_corruption_types": len(corrupt_accs),
        "total_samples":  len(preds),
    }

    print("\n" + "="*50)
    print("  DriveBench Evaluation Results")
    print("="*50)
    print(f"  Clean Accuracy : {clean_acc:.2f}%")
    print(f"  mPC            : {mPC:.2f}%")
    print(f"  rPC            : {rPC:.4f}")
    print(f"\n  Per-corruption breakdown:")
    for ct, acc in sorted(per_corruption.items()):
        print(f"    {ct:<30}: {acc:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved → {args.output}")


if __name__ == "__main__":
    main()
