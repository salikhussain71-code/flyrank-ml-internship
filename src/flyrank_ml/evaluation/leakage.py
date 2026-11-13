"""Leakage guards. These raise errors rather than warn."""
from __future__ import annotations

import pandas as pd


def assert_no_temporal_overlap(train_label_end, valid_label_start) -> None:
    """Training labels must be fully observed before the validation label window begins."""
    if pd.Timestamp(train_label_end) >= pd.Timestamp(valid_label_start):
        raise AssertionError(
            f"Temporal leakage: train label window ends {pd.Timestamp(train_label_end).date()} "
            f"but validation label window starts {pd.Timestamp(valid_label_start).date()}."
        )


def assert_disjoint_groups(train_ids, valid_ids) -> None:
    overlap = set(train_ids) & set(valid_ids)
    if overlap:
        raise AssertionError(f"Group leakage: {len(overlap)} pages appear in both train and validation.")


def assert_features_before_cutoff(daily: pd.DataFrame, features: pd.DataFrame, cutoff, date_col: str = "date") -> None:
    """Every feature row must be stamped with the cutoff and the cutoff must precede the label window."""
    cutoff = pd.Timestamp(cutoff)
    if "feature_cutoff" in features and (pd.to_datetime(features["feature_cutoff"]) != cutoff).any():
        raise AssertionError("Feature rows carry a cutoff different from the declared decision date.")
    if "label_window_start" in features and (pd.to_datetime(features["label_window_start"]) <= cutoff).any():
        raise AssertionError("Label window starts on or before the feature cutoff.")


def flag_suspicious_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> list[str]:
    """Features with near-perfect absolute correlation to the label are likely leaks."""
    flagged = []
    for c in X.columns:
        col = pd.to_numeric(X[c], errors="coerce")
        if col.nunique(dropna=True) < 2:
            continue
        corr = col.corr(pd.Series(y, index=X.index).astype(float))
        if pd.notna(corr) and abs(corr) >= threshold:
            flagged.append(c)
    return flagged
