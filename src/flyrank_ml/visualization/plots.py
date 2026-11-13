"""Charts. They contain only aggregate metrics and pseudonymous counts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils.io import ensure_dir  # noqa: E402


def plot_model_vs_baseline(model_metrics: dict, baseline_metrics: dict, out_path: str | Path, keys=None) -> Path:
    keys = keys or ["roc_auc", "average_precision", "precision_at_10", "precision_at_50", "ndcg_at_10"]
    keys = [k for k in keys if k in model_metrics and k in baseline_metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(keys))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [baseline_metrics[k] for k in keys], w, label="Baseline", color="#8a9a8f")
    ax.bar([i + w / 2 for i in x], [model_metrics[k] for k in keys], w, label="Model", color="#1f4e5f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Score (same validation split)")
    ax.set_title("Model vs baseline on the same held-out pages and period")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_action_counts(recs: pd.DataFrame, out_path: str | Path) -> Path:
    counts = recs["action"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#1f4e5f")
    ax.set_xlabel("Pages")
    ax.set_title("Recommended actions (validation pages)")
    fig.tight_layout()
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
