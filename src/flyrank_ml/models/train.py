"""Reproducible model training with scikit-learn."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str = "hist_gradient_boosting",
    params: dict[str, Any] | None = None,
    seed: int = 42,
):
    """Fit a classifier. Raises if the training labels contain only one class."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if pd.Series(y).nunique() < 2:
        raise ValueError("Training labels contain a single class; cannot train a classifier.")
    params = dict(params or {})
    if algorithm == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=seed, **params)
    elif algorithm == "logistic_regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, **params))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    model.fit(X, y)
    return model


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive (decline) class."""
    return model.predict_proba(X)[:, 1]
