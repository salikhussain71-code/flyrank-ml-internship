"""Tiny SYNTHETIC datasets for unit tests only (never FlyRank data)."""
import pandas as pd


def tiny_daily():
    rows = []
    for page, (a, b) in {"p1": (10, 2), "p2": (5, 5), "p3": (1, 10)}.items():
        for i, d in enumerate(pd.date_range("2025-01-01", periods=60)):
            clicks = a if i < 30 else b
            rows.append((page, d, clicks, clicks * 10 + 5))
    return pd.DataFrame(rows, columns=["page_id", "date", "clicks", "impressions"])
