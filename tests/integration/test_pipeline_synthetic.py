"""End-to-end test on SYNTHETIC data. Proves the code runs; says nothing about FlyRank results."""
import _bootstrap  # noqa: F401
import json
import tempfile
import unittest
from pathlib import Path

from flyrank_ml.pipeline import run_pipeline
from flyrank_ml.utils.synthetic import make_synthetic_daily


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.res = run_pipeline(make_synthetic_daily(n_pages=150, n_days=170), cls.out, data_source="synthetic_demo")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_created(self):
        for rel in ["metrics/results.json", "data_quality/quality.json", "tables/ranked_recommendations.csv",
                    "figures/model_vs_baseline.png", "figures/action_counts.png"]:
            self.assertTrue((self.out / rel).exists(), rel)

    def test_synthetic_is_never_real(self):
        self.assertFalse(self.res["is_flyrank_real_data"])
        saved = json.loads((self.out / "metrics" / "results.json").read_text())
        self.assertEqual(saved["data_source"], "synthetic_demo")

    def test_same_split_and_valid_metrics(self):
        m, b = self.res["model_metrics"], self.res["baseline_metrics"]
        self.assertEqual(m["n"], b["n"])
        self.assertTrue(0 <= m["roc_auc"] <= 1 and 0 <= b["roc_auc"] <= 1)

    def test_leakage_checks_recorded(self):
        self.assertEqual(self.res["leakage_checks"]["group_overlap"], "passed")

    def test_insufficient_history_raises(self):
        with self.assertRaises(ValueError):
            run_pipeline(make_synthetic_daily(n_pages=20, n_days=60), self.out / "x", data_source="synthetic_demo")

    def test_deterministic(self):
        again = run_pipeline(make_synthetic_daily(n_pages=150, n_days=170), self.out / "again", data_source="synthetic_demo")
        self.assertEqual(again["model_metrics"]["roc_auc"], self.res["model_metrics"]["roc_auc"])
