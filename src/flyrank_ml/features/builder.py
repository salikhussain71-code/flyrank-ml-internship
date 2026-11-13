"""Point-in-time feature engineering.

Every feature uses ONLY rows dated on or before the cutoff (the decision date).
Two equal-length windows are compared: the recent window (ending at the cutoff)
and the prior window (immediately before it).
"""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "clicks_recent", "clicks_prior", "impressions_recent", "impressions_prior",
    "ctr_recent", "ctr_prior", "clicks_delta_pct", "impressions_delta_pct",
    "active_days_recent",
]


def _window_agg(df: pd.DataFrame, page_col: str, date_col: str, clicks: str, impr: str, suffix: str) -> pd.DataFrame:
    g = df.groupby(page_col).agg(
        **{
            f"clicks_{suffix}": (clicks, "sum"),
            f"impressions_{suffix}": (impr, "sum"),
            f"active_days_{suffix}": (date_col, "nunique"),
        }
    )
    return g


def build_features(
    daily: pd.DataFrame,
    cutoff,
    page_col: str = "page_id",
    date_col: str = "date",
    clicks_col: str = "clicks",
    impressions_col: str = "impressions",
    window_days: int = 28,
) -> pd.DataFrame:
    """Return one row per page with features computed as of `cutoff` (inclusive)."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    d = daily[[page_col, date_col, clicks_col, impressions_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    cutoff = pd.Timestamp(cutoff)
    d = d[d[date_col] <= cutoff]  # hard point-in-time filter: nothing after the cutoff is used
    recent_start = cutoff - pd.Timedelta(days=window_days - 1)
    prior_start = recent_start - pd.Timedelta(days=window_days)
    recent = d[d[date_col] >= recent_start]
    prior = d[(d[date_col] >= prior_start) & (d[date_col] < recent_start)]
    r = _window_agg(recent, page_col, date_col, clicks_col, impressions_col, "recent")
    p = _window_agg(prior, page_col, date_col, clicks_col, impressions_col, "prior")
    f = r.join(p, how="outer").fillna(0.0)
    f["ctr_recent"] = f["clicks_recent"] / f["impressions_recent"].where(f["impressions_recent"] > 0)
    f["ctr_prior"] = f["clicks_prior"] / f["impressions_prior"].where(f["impressions_prior"] > 0)
    f[["ctr_recent", "ctr_prior"]] = f[["ctr_recent", "ctr_prior"]].fillna(0.0)
    f["clicks_delta_pct"] = (f["clicks_recent"] - f["clicks_prior"]) / (f["clicks_prior"] + 1.0)
    f["impressions_delta_pct"] = (f["impressions_recent"] - f["impressions_prior"]) / (f["impressions_prior"] + 1.0)
    f = f.drop(columns=["active_days_prior"])
    f.index.name = page_col
    f = f.reset_index()
    f["feature_cutoff"] = cutoff
    return f
