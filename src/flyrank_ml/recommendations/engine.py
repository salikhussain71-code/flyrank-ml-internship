"""Ranked content recommendations with reason codes.

Scores are model/baseline estimates of RISK OF DECLINE. They are decision-support
signals, not confirmed business outcomes, and imply no causal effect of any action.
"""
from __future__ import annotations

import pandas as pd

from ..utils.io import anonymize_ids

REASON_DESCRIPTIONS = {
    "DECLINING_CLICKS": "Clicks fell by 20% or more versus the prior window.",
    "FALLING_IMPRESSIONS": "Impressions fell by 20% or more versus the prior window.",
    "VISIBLE_LOW_CTR": "High impressions (top quartile) but click-through rate below the median.",
    "GROWING_CLICKS": "Clicks grew by 20% or more versus the prior window.",
    "LOW_ACTIVITY": "Fewer than 7 active days in the recent window; evidence is thin.",
}

DISCLAIMER = "Model score is a decision-support estimate, not a confirmed outcome or causal effect."


def _reasons(row: pd.Series, impr_q75: float, ctr_median: float) -> list[str]:
    codes = []
    if row["clicks_delta_pct"] <= -0.20:
        codes.append("DECLINING_CLICKS")
    if row["impressions_delta_pct"] <= -0.20:
        codes.append("FALLING_IMPRESSIONS")
    if row["impressions_recent"] >= impr_q75 and row["ctr_recent"] < ctr_median:
        codes.append("VISIBLE_LOW_CTR")
    if row["clicks_delta_pct"] >= 0.20:
        codes.append("GROWING_CLICKS")
    if row["active_days_recent"] < 7:
        codes.append("LOW_ACTIVITY")
    return codes


def _action(codes: list[str], rank: int, top_k: int) -> str:
    if "LOW_ACTIVITY" in codes:
        return "MONITOR"
    if "GROWING_CLICKS" in codes and "DECLINING_CLICKS" not in codes:
        return "PROTECT"
    if "VISIBLE_LOW_CTR" in codes and "DECLINING_CLICKS" not in codes:
        return "IMPROVE_METADATA_REVIEW"
    if "DECLINING_CLICKS" in codes and rank <= top_k:
        return "REFRESH_REVIEW"
    return "MONITOR"


def build_recommendations(
    features: pd.DataFrame,
    scores,
    page_col: str = "page_id",
    top_k: int = 10,
    salt: str = "flyrank",
) -> pd.DataFrame:
    """Return a ranked table (rank 1 = highest decline-risk score) using anonymised page IDs."""
    if len(features) != len(scores):
        raise ValueError("features and scores must have the same length")
    df = features.reset_index(drop=True).copy()
    df["score"] = list(scores)
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = df.index + 1
    impr_q75 = df["impressions_recent"].quantile(0.75)
    ctr_median = df["ctr_recent"].median()
    codes = df.apply(lambda r: _reasons(r, impr_q75, ctr_median), axis=1)
    df["reason_codes"] = codes.map(lambda c: ";".join(c) if c else "NONE")
    df["action"] = [_action(c, r, top_k) for c, r in zip(codes, df["rank"])]
    df["explanation"] = codes.map(
        lambda c: " ".join(REASON_DESCRIPTIONS[x] for x in c) if c else "No rule-based reason triggered; ranked on model score only."
    )
    df["page_pseudonym"] = anonymize_ids(df[page_col], salt=salt)
    df["disclaimer"] = DISCLAIMER
    cols = ["rank", "page_pseudonym", "score", "action", "reason_codes", "explanation",
            "clicks_recent", "clicks_prior", "clicks_delta_pct", "impressions_recent", "ctr_recent", "disclaimer"]
    return df[cols]
