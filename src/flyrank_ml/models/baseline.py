"""Transparent baseline action score (no training)."""
from __future__ import annotations

import pandas as pd

BASELINE_DESCRIPTION = (
    "Review-priority score = 0.5 * percentile(falling clicks) + 0.3 * percentile(recent clicks) "
    "+ 0.2 * percentile(recent impressions). Pages that are losing clicks, and that matter "
    "because they have volume, rank first."
)


def baseline_score(features: pd.DataFrame) -> pd.Series:
    """Score in [0, 1]; higher = review sooner. Uses only point-in-time features."""
    required = ["clicks_delta_pct", "clicks_recent", "impressions_recent"]
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns for baseline: {missing}")
    falling = (-features["clicks_delta_pct"]).rank(pct=True)
    volume = features["clicks_recent"].rank(pct=True)
    visibility = features["impressions_recent"].rank(pct=True)
    score = 0.5 * falling + 0.3 * volume + 0.2 * visibility
    return score.rename("baseline_score")
