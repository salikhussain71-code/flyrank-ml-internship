"""Run the pipeline.

  python scripts/run_pipeline.py                 # REAL FlyRank data (needs HF_TOKEN + mapped config)
  python scripts/run_pipeline.py --synthetic-demo   # SYNTHETIC demo, written to reports/_demo_synthetic/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flyrank_ml.config import BlockedError, load_config  # noqa: E402
from flyrank_ml.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-demo", action="store_true", help="Use SYNTHETIC data (never a real result).")
    args = ap.parse_args()
    cfg = load_config()
    exp = {k: v for k, v in cfg["experiment"].items() if k not in ("lane", "lane_status")}
    exp.update({"algorithm": cfg["model"]["algorithm"], "model_params": cfg["model"].get("model_params", {})})

    if args.synthetic_demo:
        from flyrank_ml.utils.synthetic import make_synthetic_daily
        out = ROOT / "reports" / "_demo_synthetic"
        res = run_pipeline(make_synthetic_daily(), out, data_source="synthetic_demo", cfg=exp)
        print("SYNTHETIC DEMO finished. NOT FlyRank results. Output:", out)
        print("ROC AUC model/baseline:", round(res["model_metrics"]["roc_auc"], 3), round(res["baseline_metrics"]["roc_auc"], 3))
        return 0

    from flyrank_ml.config import load_env_file
    load_env_file()
    try:
        from flyrank_ml.data.loader import load_daily_from_warehouse
        daily = load_daily_from_warehouse(cfg["data"])
    except BlockedError as exc:
        print("BLOCKED:", exc)
        return 2
    res = run_pipeline(daily, ROOT / "reports", data_source="flyrank_real", cfg=exp)
    print("Real-data run finished. Status: RESULTS-GENERATED. See reports/metrics/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
