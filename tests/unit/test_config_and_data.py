import _bootstrap  # noqa: F401
import os
import unittest
from unittest import mock

import pandas as pd

from flyrank_ml.config import BlockedError, get_hf_token, load_config, require_configured
from flyrank_ml.data.quality import check_required_columns, clean_daily, quality_report


class TestConfig(unittest.TestCase):
    def test_load_config_has_three_sections(self):
        cfg = load_config()
        self.assertEqual(set(cfg), {"data", "experiment", "model"})
        self.assertEqual(cfg["experiment"]["lane_status"], "provisional")

    def test_unconfigured_schema_is_blocked(self):
        with self.assertRaises(BlockedError):
            require_configured(load_config()["data"])

    def test_configured_schema_passes(self):
        cfg = {"daily_table": "x", "columns": {"page_id": "a", "date": "b", "clicks": "c", "impressions": "d"}}
        require_configured(cfg)

    def test_token_only_from_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_hf_token())
            with self.assertRaises(BlockedError):
                get_hf_token(required=True)
        with mock.patch.dict(os.environ, {"HF_TOKEN": "abc"}, clear=True):
            self.assertEqual(get_hf_token(), "abc")


class TestQuality(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "page_id": ["a", "a", "b", None, "c"],
            "date": ["2025-01-01", "2025-01-01", "bad", "2025-01-02", "2025-01-03"],
            "clicks": [1, 2, 3, 4, -1], "impressions": [10, 10, 10, 10, 10],
        })

    def test_required_columns(self):
        with self.assertRaises(ValueError):
            check_required_columns(self.df, ["nope"])

    def test_quality_report_counts(self):
        r = quality_report(self.df, "page_id", "date", ["clicks", "impressions"])
        self.assertEqual(r["n_rows"], 5)
        self.assertEqual(r["n_unparseable_dates"], 1)
        self.assertEqual(r["n_duplicate_page_date_rows"], 1)
        self.assertEqual(r["negative_metric_values"]["clicks"], 1)

    def test_clean_daily_exclusions_and_collapse(self):
        clean, ex = clean_daily(self.df, "page_id", "date", ["clicks", "impressions"])
        self.assertEqual(ex["unparseable_date"], 1)
        self.assertEqual(ex["missing_page_id"], 1)
        self.assertEqual(ex["missing_or_negative_metric"], 1)
        self.assertEqual(len(clean), 1)
        self.assertEqual(int(clean["clicks"].iloc[0]), 3)
        self.assertEqual(ex["duplicate_rows_collapsed"], 1)
