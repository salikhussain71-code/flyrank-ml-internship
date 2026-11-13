"""SYNTHETIC data generator. For unit tests and clearly-labelled demos ONLY.

Output of this module must NEVER be presented as FlyRank data or FlyRank research results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_daily(n_pages: int = 300, n_days: int = 200, seed: int = 0, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for i in range(n_pages):
        base_impr = rng.lognormal(mean=5.0, sigma=1.0)
        ctr = rng.uniform(0.01, 0.08)
        trend = rng.choice([-0.004, 0.0, 0.003], p=[0.35, 0.4, 0.25])
        for t, d in enumerate(dates):
            mult = max(0.05, 1 + trend * t)
            impr = rng.poisson(base_impr * mult)
            if impr == 0:
                continue
            clicks = rng.binomial(impr, ctr)
            rows.append((f"synthetic-page-{i}", d, int(clicks), int(impr)))
    return pd.DataFrame(rows, columns=["page_id", "date", "clicks", "impressions"])
