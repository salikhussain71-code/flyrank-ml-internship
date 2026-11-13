"""Data-quality checks: missing values, duplicates, ranges, coverage."""
from __future__ import annotations

from typing import Any

import pandas as pd


def check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def quality_report(df: pd.DataFrame, page_col: str, date_col: str, metric_cols: list[str]) -> dict[str, Any]:
    """Aggregate-only report. Contains no row-level or identifying data."""
    check_required_columns(df, [page_col, date_col, *metric_cols])
    dates = pd.to_datetime(df[date_col], errors="coerce")
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_pages": int(df[page_col].nunique()),
        "date_min": None if dates.isna().all() else str(dates.min().date()),
        "date_max": None if dates.isna().all() else str(dates.max().date()),
        "n_unparseable_dates": int(dates.isna().sum()),
        "n_duplicate_page_date_rows": int(df.duplicated(subset=[page_col, date_col]).sum()),
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "negative_metric_values": {c: int((pd.to_numeric(df[c], errors="coerce") < 0).sum()) for c in metric_cols},
        "n_distinct_dates": int(dates.dropna().nunique()),
    }
    if "clicks" in metric_cols and "impressions" in metric_cols:
        report["rows_clicks_exceed_impressions"] = int((df["clicks"] > df["impressions"]).sum())
    return report


def clean_daily(df: pd.DataFrame, page_col: str, date_col: str, metric_cols: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unusable rows and collapse duplicate page-date rows. Returns (clean_df, exclusions)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    exclusions: dict[str, int] = {}
    bad_date = out[date_col].isna()
    exclusions["unparseable_date"] = int(bad_date.sum())
    out = out[~bad_date]
    bad_page = out[page_col].isna()
    exclusions["missing_page_id"] = int(bad_page.sum())
    out = out[~bad_page]
    for c in metric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad_metric = out[metric_cols].isna().any(axis=1) | (out[metric_cols] < 0).any(axis=1)
    exclusions["missing_or_negative_metric"] = int(bad_metric.sum())
    out = out[~bad_metric]
    n_before = len(out)
    out = out.groupby([page_col, date_col], as_index=False)[metric_cols].sum()
    exclusions["duplicate_rows_collapsed"] = int(n_before - len(out))
    return out, exclusions
