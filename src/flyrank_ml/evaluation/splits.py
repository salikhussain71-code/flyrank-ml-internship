"""Group + time-aware split: validation pages are never seen in training and validation
labels come from a later period than training labels."""
from __future__ import annotations

import hashlib

import pandas as pd


def _bucket(value, seed: int) -> float:
    h = hashlib.md5(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_groups(page_ids: pd.Series, valid_fraction: float = 0.3, seed: int = 42) -> pd.Series:
    """Deterministic page-level assignment: 'train' or 'valid'."""
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    return page_ids.map(lambda v: "valid" if _bucket(v, seed) < valid_fraction else "train")


def derive_cutoffs(max_date, horizon_days: int) -> dict[str, pd.Timestamp]:
    """valid_cutoff leaves one full horizon of future data; train_cutoff is one horizon earlier."""
    max_date = pd.Timestamp(max_date)
    valid_cutoff = max_date - pd.Timedelta(days=horizon_days)
    train_cutoff = valid_cutoff - pd.Timedelta(days=horizon_days)
    return {"train_cutoff": train_cutoff, "valid_cutoff": valid_cutoff}


def group_time_split(
    daily: pd.DataFrame, page_col: str = "page_id", valid_fraction: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the daily table by page (group) into train-pages and valid-pages."""
    ids = daily[page_col].drop_duplicates()
    groups = assign_groups(ids, valid_fraction, seed)
    valid_pages = set(ids[groups == "valid"])
    mask = daily[page_col].isin(valid_pages)
    return daily[~mask].copy(), daily[mask].copy()
