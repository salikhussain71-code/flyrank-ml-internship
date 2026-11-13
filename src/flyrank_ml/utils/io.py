"""Small I/O and privacy helpers."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default, sort_keys=True)
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def anonymize_ids(ids: pd.Series, salt: str = "flyrank") -> pd.Series:
    """One-way pseudonyms so public outputs never contain real URLs or page identifiers."""
    def _h(v: Any) -> str:
        return "P-" + hashlib.sha256(f"{salt}|{v}".encode("utf-8")).hexdigest()[:8]
    return ids.map(_h)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
