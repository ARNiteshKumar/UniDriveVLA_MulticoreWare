"""
DriveLM metric computation helpers.

BLEU implemented from scratch (no NLTK dependency required).
CIDEr uses a TF-IDF based consensus score.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Exact-match accuracy after normalisation."""
    assert len(predictions) == len(references)
    correct = sum(
        _normalize(p) == _normalize(r)
        for p, r in zip(predictions, references)
    )
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> Tuple[int, int]:
    pred_ngrams = _ngram_counts(pred_tokens, n)
    ref_ngrams  = _ngram_counts(ref_tokens, n)
    clipped = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in pred_ngrams.items())
    total = max(1, len(pred_tokens) - n + 1)
    return clipped, total


def _brevity_penalty(pred_len: float, ref_len: float) -> float:
    if pred_len >= ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / max(pred_len, 1))


def compute_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> List[float]:
    """Corpus-level BLEU-1 through BLEU-{max_n}."""
    assert len(predictions) == len(references)

    clipped_total = [0] * max_n
    pred_total    = [0] * max_n
    pred_len_sum = 0
    ref_len_sum  = 0

    for pred, ref in zip(predictions, references):
        p_tok = _tokenize(pred)
        r_tok = _tokenize(ref)
        pred_len_sum += len(p_tok)
        ref_len_sum  += len(r_tok)
        for n in range(1, max_n + 1):
            c, t = _clipped_precision(p_tok, r_tok, n)
            clipped_total[n - 1] += c
            pred_total[n - 1]    += t

    bp = _brevity_penalty(pred_len_sum, ref_len_sum)

    scores = []
    for n in range(1, max_n + 1):
        prec = clipped_total[n - 1] / max(pred_total[n - 1], 1)
        if prec == 0:
            scores.append(0.0)
        else:
            log_geo_mean = sum(
                math.log(clipped_total[i] / max(pred_total[i], 1))
                for i in range(n)
            ) / n
            scores.append(bp * math.exp(log_geo_mean))

    return scores


# ---------------------------------------------------------------------------
# CIDEr (simplified, without stemming)
# ---------------------------------------------------------------------------

def _tfidf_weights(ngrams_list: List[Counter], n: int) -> Dict:
    """Compute IDF weights for all n-grams in the corpus."""
    df: Counter = Counter()
    for ngrams in ngrams_list:
        df.update(set(ngrams.keys()))
    idf = {ng: math.log(len(ngrams_list) / (cnt + 1.0)) + 1.0 for ng, cnt in df.items()}
    return idf


def compute_cider(predictions: List[str], references: List[str], n: int = 4) -> float:
    """Corpus-level CIDEr-D (simplified, single reference)."""
    assert len(predictions) == len(references)

    pred_ngrams = [
        _ngram_counts(_tokenize(p), n) for p in predictions
    ]
    ref_ngrams  = [
        _ngram_counts(_tokenize(r), n) for r in references
    ]

    idf = _tfidf_weights(ref_ngrams, n)

    scores = []
    for p_ng, r_ng in zip(pred_ngrams, ref_ngrams):
        all_keys = set(p_ng) | set(r_ng)
        pred_vec = {k: p_ng.get(k, 0) * idf.get(k, 1.0) for k in all_keys}
        ref_vec  = {k: r_ng.get(k, 0) * idf.get(k, 1.0) for k in all_keys}

        dot   = sum(pred_vec[k] * ref_vec[k] for k in all_keys)
        n_p   = math.sqrt(sum(v ** 2 for v in pred_vec.values()) + 1e-9)
        n_r   = math.sqrt(sum(v ** 2 for v in ref_vec.values())  + 1e-9)
        scores.append(dot / (n_p * n_r))

    return float(sum(scores) / len(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# DriveLM official score
# ---------------------------------------------------------------------------

def compute_drivelm_score(metrics: Dict) -> float:
    """Weighted combination used in the DriveLM leaderboard.

    Official formula (approximate, from the DriveLM paper):
        score = 0.4 * BLEU-4 + 0.4 * CIDEr / 10 + 0.2 * Accuracy
    """
    bleu4  = metrics.get("bleu_4", 0.0)
    cider  = metrics.get("cider", 0.0)
    acc    = metrics.get("accuracy", 0.0)
    return 0.4 * bleu4 + 0.4 * (cider / 10.0) + 0.2 * acc
