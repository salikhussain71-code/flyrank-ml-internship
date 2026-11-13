"""End-to-end pipeline: quality -> split -> features -> labels -> baseline -> model -> eval -> recs.

`data_source` must be "flyrank_real" for results to count as FlyRank research results.
Anything else is recorded as synthetic/demo and can never be used for the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data.quality import clean_daily, quality_report
from .evaluation.leakage import (
    assert_disjoint_groups, assert_features_before_cutoff, assert_no_temporal_overlap, flag_suspicious_features,
)
from .evaluation.metrics import bootstrap_diff, evaluate_scores
from .evaluation.splits import derive_cutoffs, group_time_split
from .features.builder import FEATURE_COLUMNS, build_features
from .labels.builder import make_future_label
from .models.baseline import BASELINE_DESCRIPTION, baseline_score
from .models.train import predict_scores, train_model
from .recommendations.engine import build_recommendations
from .utils.io import ensure_dir, save_json, set_seed
from .visualization.plots import plot_action_counts, plot_model_vs_baseline

DEFAULTS = {
    "window_days": 28, "horizon_days": 28, "decline_threshold": 0.30, "min_recent_clicks": 10,
    "valid_fraction": 0.3, "seed": 42, "top_k": 10, "algorithm": "hist_gradient_boosting", "model_params": {},
}


def run_pipeline(daily: pd.DataFrame, out_dir: str | Path, data_source: str, cfg: dict[str, Any] | None = None,
                 page_col: str = "page_id", date_col: str = "date") -> dict[str, Any]:
    c = {**DEFAULTS, **(cfg or {})}
    set_seed(c["seed"])
    out = ensure_dir(out_dir)
    metric_cols = ["clicks", "impressions"]

    quality = quality_report(daily, page_col, date_col, metric_cols)
    clean, exclusions = clean_daily(daily, page_col, date_col, metric_cols)
    save_json({"quality": quality, "exclusions": exclusions}, out / "data_quality" / "quality.json")

    cutoffs = derive_cutoffs(clean[date_col].max(), c["horizon_days"])
    needed_start = cutoffs["train_cutoff"] - pd.Timedelta(days=2 * c["window_days"] - 1)
    if clean[date_col].min() > needed_start:
        raise ValueError(
            f"Not enough history: need data from {needed_start.date()} but earliest is {clean[date_col].min().date()}."
        )
    assert_no_temporal_overlap(cutoffs["train_cutoff"] + pd.Timedelta(days=c["horizon_days"]), cutoffs["valid_cutoff"] + pd.Timedelta(days=1))

    train_daily, valid_daily = group_time_split(clean, page_col, c["valid_fraction"], c["seed"])
    assert_disjoint_groups(train_daily[page_col].unique(), valid_daily[page_col].unique())

    def make(daily_part: pd.DataFrame, cutoff):
        f = build_features(daily_part, cutoff, page_col, date_col, "clicks", "impressions", c["window_days"])
        lab = make_future_label(daily_part, f, cutoff, c["horizon_days"], c["decline_threshold"],
                                c["min_recent_clicks"], page_col, date_col, "clicks")
        assert_features_before_cutoff(daily_part, lab, cutoff, date_col)
        return lab

    train = make(train_daily, cutoffs["train_cutoff"])
    valid = make(valid_daily, cutoffs["valid_cutoff"])
    if len(train) == 0 or len(valid) == 0:
        raise ValueError("Empty train or validation set after eligibility filter.")

    suspicious = flag_suspicious_features(train[FEATURE_COLUMNS], train["label_decline"])
    if suspicious:
        raise AssertionError(f"Suspicious near-perfect feature/label correlation: {suspicious}")

    model = train_model(train[FEATURE_COLUMNS], train["label_decline"], c["algorithm"], c["model_params"], c["seed"])
    model_scores = predict_scores(model, valid[FEATURE_COLUMNS])
    base_scores = baseline_score(valid).to_numpy()
    y = valid["label_decline"].to_numpy()
    model_m = evaluate_scores(y, model_scores)
    base_m = evaluate_scores(y, base_scores)
    diff = bootstrap_diff(y, model_scores, base_scores, seed=c["seed"])

    recs = build_recommendations(valid, model_scores, page_col, c["top_k"])
    ensure_dir(out / "tables")
    recs.to_csv(out / "tables" / "ranked_recommendations.csv", index=False)
    plot_model_vs_baseline(model_m, base_m, out / "figures" / "model_vs_baseline.png")
    plot_action_counts(recs, out / "figures" / "action_counts.png")

    results = {
        "data_source": data_source,
        "is_flyrank_real_data": data_source == "flyrank_real",
        "label": "Does a page lose >= {:.0%} of clicks in the next {} days vs its last {} days?".format(
            c["decline_threshold"], c["horizon_days"], c["window_days"]),
        "config": c,
        "cutoffs": {k: str(v.date()) for k, v in cutoffs.items()},
        "n_train": int(len(train)), "n_valid": int(len(valid)),
        "features": FEATURE_COLUMNS,
        "baseline_description": BASELINE_DESCRIPTION,
        "model_metrics": model_m, "baseline_metrics": base_m,
        "roc_auc_diff_bootstrap_model_minus_baseline": diff,
        "leakage_checks": {"temporal_overlap": "passed", "group_overlap": "passed",
                           "feature_cutoff": "passed", "suspicious_features": "none"},
        "exclusions": exclusions,
    }
    save_json(results, out / "metrics" / "results.json")
    return results
