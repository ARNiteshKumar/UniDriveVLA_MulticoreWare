"""DriveBench robustness metric helpers."""

import re
from typing import Dict, List


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _accuracy_for_split(
    predictions: List[Dict],
    gt: Dict[str, str],
) -> float:
    correct = 0
    total = 0
    for pred in predictions:
        qid = str(pred.get("id", ""))
        if qid not in gt:
            continue
        if _normalize(pred.get("answer", "")) == _normalize(gt[qid]):
            correct += 1
        total += 1
    return correct / max(total, 1)


def compute_per_corruption_accuracy(
    predictions: Dict[str, List[Dict]],
    gt_lookup: Dict[str, Dict[str, str]],
) -> Dict[str, float]:
    """Accuracy for each (corruption, severity) split."""
    results = {}
    for key, preds in predictions.items():
        gt = gt_lookup.get(key, {})
        results[key] = _accuracy_for_split(preds, gt)
    return results


def compute_mpc(per_corruption: Dict[str, float]) -> float:
    """Mean Performance under Corruptions (excluding clean)."""
    values = [v for k, v in per_corruption.items() if k != "clean"]
    return sum(values) / len(values) if values else 0.0


def compute_rr(per_corruption: Dict[str, float]) -> float:
    """Relative Robustness = mPC / clean_accuracy."""
    clean = per_corruption.get("clean", None)
    if clean is None or clean == 0:
        return 0.0
    mpc = compute_mpc(per_corruption)
    return mpc / clean
