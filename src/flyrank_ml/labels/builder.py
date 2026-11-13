"""Future-window label: does a page's clicks fall materially in the NEXT window?

Label uses only rows strictly AFTER the cutoff. Features use only rows ON OR BEFORE
the cutoff, so features and labels never share a row.
"""
from __future__ import annotations

import pandas as pd


def make_future_label(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    cutoff,
    horizon_days: int = 28,
    decline_threshold: float = 0.30,
    min_recent_clicks: int = 10,
    page_col: str = "page_id",
    date_col: str = "date",
    clicks_col: str = "clicks",
) -> pd.DataFrame:
    """Add future_clicks, future_change_pct and binary label_decline to eligible pages.

    Eligible pages have at least `min_recent_clicks` clicks in the recent window, so a
    "decline" is measurable. Pages with no future rows are treated as 0 future clicks
    (assumption: the warehouse omits page-days with no impressions).
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if not 0 < decline_threshold < 1:
        raise ValueError("decline_threshold must be between 0 and 1")
    cutoff = pd.Timestamp(cutoff)
    d = daily[[page_col, date_col, clicks_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    end = cutoff + pd.Timedelta(days=horizon_days)
    fut = d[(d[date_col] > cutoff) & (d[date_col] <= end)]
    fut_clicks = fut.groupby(page_col)[clicks_col].sum().rename("future_clicks")
    out = features.merge(fut_clicks, how="left", left_on=page_col, right_index=True)
    out["future_clicks"] = out["future_clicks"].fillna(0.0)
    out = out[out["clicks_recent"] >= min_recent_clicks].copy()
    out["future_change_pct"] = (out["future_clicks"] - out["clicks_recent"]) / (out["clicks_recent"] + 1.0)
    out["label_decline"] = (out["future_change_pct"] <= -decline_threshold).astype(int)
    out["label_window_start"] = cutoff + pd.Timedelta(days=1)
    out["label_window_end"] = end
    return out.reset_index(drop=True)
