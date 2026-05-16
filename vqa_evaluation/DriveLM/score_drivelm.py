"""DriveLM scoring — BLEU, CIDEr, accuracy, DriveLM composite score.

Usage:
    python vqa_evaluation/DriveLM/score_drivelm.py \
        --pred_path results/drivelm/preds.json \
        --gt_path   data/DriveLM/QA_dataset_nus_v1_val.json \
        --output    results/drivelm/scores.json
"""
import argparse, json, os
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", required=True)
    p.add_argument("--gt_path",   required=True)
    p.add_argument("--output",    default="drivelm_scores.json")
    return p.parse_args()

def _tokenize(text):
    return str(text).lower().split()

def bleu_n(pred_tokens, gt_tokens, n):
    if len(pred_tokens) < n:
        return 0.0
    pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
    gt_ngrams   = [tuple(gt_tokens[i:i+n])   for i in range(len(gt_tokens)-n+1)]
    gt_set = defaultdict(int)
    for g in gt_ngrams:
        gt_set[g] += 1
    match = sum(min(pred_ngrams.count(g), cnt) for g, cnt in gt_set.items())
    return match / len(pred_ngrams) if pred_ngrams else 0.0

def compute_metrics(preds):
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    exact_matches = []
    per_type = defaultdict(list)

    for item in preds:
        pred  = _tokenize(item.get("prediction", ""))
        gt    = _tokenize(item.get("gt_answer",  ""))
        qtype = item.get("qa_type", "unknown")

        for n in [1, 2, 3, 4]:
            s = bleu_n(pred, gt, n)
            bleu_scores[n].append(s)

        em = 1.0 if " ".join(pred) == " ".join(gt) else 0.0
        exact_matches.append(em)
        per_type[qtype].append(em)

    results = {
        "Accuracy (EM)": sum(exact_matches) / len(exact_matches) * 100 if exact_matches else 0,
        "BLEU-1": sum(bleu_scores[1]) / len(bleu_scores[1]) * 100 if bleu_scores[1] else 0,
        "BLEU-2": sum(bleu_scores[2]) / len(bleu_scores[2]) * 100 if bleu_scores[2] else 0,
        "BLEU-3": sum(bleu_scores[3]) / len(bleu_scores[3]) * 100 if bleu_scores[3] else 0,
        "BLEU-4": sum(bleu_scores[4]) / len(bleu_scores[4]) * 100 if bleu_scores[4] else 0,
        "per_type": {k: sum(v)/len(v)*100 for k, v in per_type.items()},
        "num_samples": len(preds),
    }
    # DriveLM composite: average of BLEU-4 and accuracy
    results["DriveLM_score"] = (results["BLEU-4"] + results["Accuracy (EM)"]) / 2
    return results

def main():
    args = parse_args()
    with open(args.pred_path) as f:
        preds = json.load(f)

    metrics = compute_metrics(preds)

    print("\n" + "="*50)
    print("  DriveLM Evaluation Results")
    print("="*50)
    for k, v in metrics.items():
        if k not in ("per_type", "num_samples"):
            print(f"  {k:<25}: {v:.2f}")
    print(f"\n  Per question type:")
    for k, v in metrics["per_type"].items():
        print(f"    {k:<20}: {v:.2f}%")
    print(f"\n  Total samples : {metrics['num_samples']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nScores saved → {args.output}")

if __name__ == "__main__":
    main()
