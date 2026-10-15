"""Evaluation metrics (numpy only; one function each, all unit-tested).

Everything here is pure: lists/arrays in, numbers/dicts out. No I/O, no models.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


# ---- classification -------------------------------------------------------

def accuracy(y_true: list, y_pred: list) -> float:
    """Fraction of exact matches. Items with y_pred=None count as wrong."""
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
    """Counts[i][j] = # items with true=labels[i], pred=labels[j]."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def per_class_f1(y_true: list, y_pred: list, labels: list) -> dict[str, float]:
    """Per-class F1 (precision/recall harmonic mean), 0 when undefined."""
    out = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[lab] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return out


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    f1 = per_class_f1(y_true, y_pred, labels)
    return sum(f1.values()) / len(labels) if labels else 0.0


# ---- advice / refusal -----------------------------------------------------

def advice_recall(is_advice_true: list[bool], is_advice_pred: list[bool]) -> float:
    """Of the true-advice items, fraction correctly flagged (caught refusals)."""
    pos = [p for t, p in zip(is_advice_true, is_advice_pred) if t]
    return sum(pos) / len(pos) if pos else 0.0


def false_refusal_rate(is_advice_true: list[bool], is_advice_pred: list[bool]) -> float:
    """Of the non-advice items, fraction wrongly refused."""
    neg = [p for t, p in zip(is_advice_true, is_advice_pred) if not t]
    return sum(neg) / len(neg) if neg else 0.0


# ---- calibration ----------------------------------------------------------

def coverage_accuracy_curve(
    correct: list[bool], confidence: list[float], thresholds: list[float]
) -> list[dict]:
    """At each threshold, coverage (fraction answered) and accuracy on answered."""
    correct = np.asarray(correct, dtype=bool)
    conf = np.asarray(confidence, dtype=float)
    rows = []
    n = len(correct)
    for th in thresholds:
        answered = conf >= th
        cov = answered.mean() if n else 0.0
        acc = correct[answered].mean() if answered.any() else float("nan")
        rows.append({"threshold": round(th, 4), "coverage": float(cov), "accuracy": float(acc)})
    return rows


def expected_calibration_error(
    correct: list[bool], confidence: list[float], n_bins: int = 10
) -> float:
    """ECE: weighted gap between confidence and accuracy across equal-width bins."""
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    n = len(correct)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def reliability_table(
    correct: list[bool], confidence: list[float], n_bins: int = 10
) -> list[dict]:
    """Per-bin mean confidence, accuracy, and count (for a reliability diagram)."""
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if mask.any() else float("nan"),
            "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
        })
    return rows


# ---- ranking --------------------------------------------------------------

def recall_at_k(ranked: list[list], relevant: list[set], k: int) -> float:
    """Mean over queries of (relevant items found in top-k) / (total relevant)."""
    vals = []
    for r, rel in zip(ranked, relevant):
        if not rel:
            continue
        vals.append(len(set(r[:k]) & rel) / len(rel))
    return float(np.mean(vals)) if vals else 0.0


def mrr(ranked: list[list], relevant: list[set]) -> float:
    """Mean reciprocal rank of the first relevant item per query."""
    vals = []
    for r, rel in zip(ranked, relevant):
        rr = 0.0
        for i, item in enumerate(r, 1):
            if item in rel:
                rr = 1.0 / i
                break
        vals.append(rr)
    return float(np.mean(vals)) if vals else 0.0


# ---- agreement / uncertainty ---------------------------------------------

def cohen_kappa(a: list, b: list) -> float:
    """Inter-annotator agreement corrected for chance."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(list(ca) + list(cb)))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Paired test for A vs B accuracy. Returns discordant counts + two-sided p."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(((~a) & b).sum())   # A wrong, B right
    b10 = int((a & (~b)).sum())   # A right, B wrong
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0}
    # exact two-sided binomial p (p=0.5) without scipy
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)
    return {"b01": b01, "b10": b10, "p_value": float(p)}


def valid_output_rate(valid_flags: list[bool]) -> float:
    """Fraction of decisions that parsed to an allowed answer (not 'invalid')."""
    return sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
