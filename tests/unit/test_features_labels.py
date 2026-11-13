import _bootstrap  # noqa: F401
import unittest

import pandas as pd

from flyrank_ml.features.builder import FEATURE_COLUMNS, build_features
from flyrank_ml.labels.builder import make_future_label
from helpers import tiny_daily


class TestFeatures(unittest.TestCase):
    def test_windows_and_values(self):
        d = tiny_daily()
        f = build_features(d, "2025-02-28", window_days=14).set_index("page_id")
        # cutoff = day index 58; recent = days 45..58 (all in second half), prior = days 31..44 (second half)
        self.assertEqual(f.loc["p1", "clicks_recent"], 2 * 14)
        self.assertEqual(f.loc["p1", "clicks_prior"], 2 * 14)
        for c in FEATURE_COLUMNS:
            self.assertIn(c, f.columns)

    def test_no_future_rows_used(self):
        d = tiny_daily()
        f1 = build_features(d, "2025-02-01", window_days=7)
        d2 = d.copy()
        d2.loc[d2["date"] > "2025-02-01", "clicks"] = 99999  # changing the future must not change features
        f2 = build_features(d2, "2025-02-01", window_days=7)
        pd.testing.assert_frame_equal(f1, f2)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            build_features(tiny_daily(), "2025-02-01", window_days=0)


class TestLabels(unittest.TestCase):
    def test_label_uses_only_future(self):
        d = tiny_daily()
        cutoff = "2025-01-30"  # day index 29: p1 clicks 10/day before, 2/day after
        f = build_features(d, cutoff, window_days=14)
        lab = make_future_label(d, f, cutoff, horizon_days=14, decline_threshold=0.3, min_recent_clicks=10).set_index("page_id")
        self.assertEqual(lab.loc["p1", "label_decline"], 1)   # 140 -> 28 clicks
        self.assertEqual(lab.loc["p2", "label_decline"], 0)   # flat
        self.assertEqual(lab.loc["p3", "label_decline"], 0)   # growing
        self.assertTrue((pd.to_datetime(lab["label_window_start"]) > pd.Timestamp(cutoff)).all())

    def test_min_recent_clicks_filters(self):
        d = tiny_daily()
        f = build_features(d, "2025-01-30", window_days=14)
        lab = make_future_label(d, f, "2025-01-30", 14, 0.3, min_recent_clicks=1000)
        self.assertEqual(len(lab), 0)

    def test_bad_parameters(self):
        d = tiny_daily(); f = build_features(d, "2025-01-30", window_days=14)
        with self.assertRaises(ValueError):
            make_future_label(d, f, "2025-01-30", 0)
        with self.assertRaises(ValueError):
            make_future_label(d, f, "2025-01-30", 14, decline_threshold=1.5)
