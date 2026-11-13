"""Configuration loading. Secrets come only from environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class BlockedError(RuntimeError):
    """Raised when real-data work cannot proceed (missing token or unconfigured schema)."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load data, experiment and model configs into one dict."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    return {
        "data": load_yaml(base / "data_config.yaml"),
        "experiment": load_yaml(base / "experiment_config.yaml"),
        "model": load_yaml(base / "model_config.yaml"),
    }


def get_hf_token(required: bool = False) -> str | None:
    """Read the Hugging Face token from HF_TOKEN. Never hard-code tokens."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if required and not token:
        raise BlockedError(
            "HF_TOKEN is not set. Create a read token on Hugging Face, copy .env.example "
            "to .env, and set HF_TOKEN there (see README)."
        )
    return token or None


REQUIRED_COLUMN_KEYS = ["page_id", "date", "clicks", "impressions"]


def require_configured(data_cfg: dict[str, Any]) -> None:
    """Raise BlockedError until the real schema has been mapped in data_config.yaml."""
    missing = [k for k in REQUIRED_COLUMN_KEYS if not data_cfg.get("columns", {}).get(k)]
    if missing or not data_cfg.get("daily_table"):
        raise BlockedError(
            "data_config.yaml is not mapped to the real warehouse schema yet. Missing: "
            f"{missing + ([] if data_cfg.get('daily_table') else ['daily_table'])}. "
            "Run the schema inspection cells in the capstone notebook, then fill the config."
        )


def load_env_file(path: str | Path | None = None) -> bool:
    """Load KEY=VALUE lines from .env into os.environ (without overriding). Returns True if file found."""
    env = Path(path) if path else PROJECT_ROOT / ".env"
    if not env.exists():
        return False
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return True
