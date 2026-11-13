"""Ranking and classification metrics. All computed on real predictions only."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


def _top_k_idx(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="mergesort")[:k]


def precision_at_k(y_true, scores, k: int) -> float:
    y = np.asarray(y_true)
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(y))
    return float(y[_top_k_idx(scores, k)].mean())


def lift_at_k(y_true, scores, k: int) -> float:
    base = float(np.mean(y_true))
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / base


def ndcg_at_k(y_true, scores, k: int) -> float:
    y = np.asarray(y_true).reshape(1, -1)
    s = np.asarray(scores).reshape(1, -1)
    if y.shape[1] < 2:
        return float("nan")
    return float(ndcg_score(y, s, k=min(k, y.shape[1])))


def evaluate_scores(y_true, scores, ks=(10, 50)) -> dict[str, float]:
    y = np.asarray(y_true)
    out: dict[str, float] = {"n": int(len(y)), "positive_rate": float(y.mean())}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["average_precision"] = float(average_precision_score(y, scores))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(y, scores, k)
        out[f"lift_at_{k}"] = lift_at_k(y, scores, k)
        out[f"ndcg_at_{k}"] = ndcg_at_k(y, scores, k)
    return out


def bootstrap_diff(
    y_true, scores_a, scores_b, metric: Callable = roc_auc_score, n_boot: int = 200, seed: int = 42
) -> dict[str, float]:
    """Bootstrap the metric difference (a - b) on the same validation rows."""
    y = np.asarray(y_true); a = np.asarray(scores_a); b = np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    if not diffs:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_used": 0}
    d = np.array(diffs)
    return {
        "mean": float(d.mean()),
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "n_boot_used": int(len(d)),
    }
