"""Check Python, packages, token presence (never prints it) and config mapping."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_dotenv_simple() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    load_dotenv_simple()
    ok = True
    print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "NEEDS 3.10+")
    ok &= sys.version_info >= (3, 10)
    for mod in ["numpy", "pandas", "sklearn", "yaml", "matplotlib", "duckdb", "huggingface_hub"]:
        try:
            importlib.import_module(mod)
            print(f"  {mod}: OK")
        except ImportError:
            print(f"  {mod}: MISSING -> pip install -r requirements.txt")
            ok = False
    token = os.environ.get("HF_TOKEN", "")
    has_token = bool(token) and "your_huggingface" not in token
    print("HF_TOKEN set:", "yes" if has_token else "NO (real-data work is BLOCKED until you set it)")
    from flyrank_ml.config import BlockedError, load_config, require_configured
    cfg = load_config()
    try:
        require_configured(cfg["data"])
        print("data_config.yaml mapped to schema: yes")
    except BlockedError as exc:
        print("data_config.yaml mapped to schema: NO ->", exc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
