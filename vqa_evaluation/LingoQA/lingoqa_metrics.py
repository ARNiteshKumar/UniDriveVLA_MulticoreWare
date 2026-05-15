"""LingoQA metric computation helpers."""

import re
from collections import Counter
from typing import Dict, List


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def _ngram_overlap(pred: str, ref: str, n: int = 4) -> float:
    p_tok = _tokenize(pred)
    r_tok = _tokenize(ref)
    if len(p_tok) < n or len(r_tok) < n:
        return float(p_tok == r_tok)
    p_ng = Counter(tuple(p_tok[i : i + n]) for i in range(len(p_tok) - n + 1))
    r_ng = Counter(tuple(r_tok[i : i + n]) for i in range(len(r_tok) - n + 1))
    overlap = sum(min(cnt, r_ng[ng]) for ng, cnt in p_ng.items())
    return overlap / max(len(p_tok) - n + 1, 1)


def compute_lingoqa_score(predictions: List[str], references: List[str]) -> float:
    """Official LingoQA score: weighted combination of token-level F1 and BLEU-4.

    Approximate implementation; the full official scorer can be found at
    https://github.com/wayveai/LingoQA/blob/main/lingoqa/metrics.py
    """
    assert len(predictions) == len(references)

    f1_scores, bleu4_scores = [], []
    for pred, ref in zip(predictions, references):
        p_tok = set(_tokenize(pred))
        r_tok = set(_tokenize(ref))
        if not p_tok and not r_tok:
            f1_scores.append(1.0)
            bleu4_scores.append(1.0)
            continue
        precision = len(p_tok & r_tok) / max(len(p_tok), 1)
        recall    = len(p_tok & r_tok) / max(len(r_tok), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        bleu4 = _ngram_overlap(pred, ref, n=4)
        f1_scores.append(f1)
        bleu4_scores.append(bleu4)

    avg_f1    = sum(f1_scores)    / len(f1_scores)
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)

    # Official weighting: 0.5 × token-F1 + 0.5 × BLEU-4
    return 0.5 * avg_f1 + 0.5 * avg_bleu4


def compute_per_type_accuracy(
    predictions: List[str],
    references: List[str],
    question_types: List[str],
) -> Dict[str, float]:
    """Exact-match accuracy broken down by question type."""
    from collections import defaultdict
    type_correct: Dict[str, int] = defaultdict(int)
    type_total:   Dict[str, int] = defaultdict(int)

    for pred, ref, qtype in zip(predictions, references, question_types):
        match = int(_normalize(pred) == _normalize(ref))
        type_correct[qtype] += match
        type_total[qtype]   += 1

    return {
        qt: type_correct[qt] / max(type_total[qt], 1)
        for qt in type_total
    }
