import _bootstrap  # noqa: F401
import unittest

import numpy as np
import pandas as pd

from flyrank_ml.evaluation.leakage import (
    assert_disjoint_groups, assert_features_before_cutoff, assert_no_temporal_overlap, flag_suspicious_features)
from flyrank_ml.evaluation.metrics import bootstrap_diff, evaluate_scores, ndcg_at_k, precision_at_k
from flyrank_ml.evaluation.splits import assign_groups, derive_cutoffs, group_time_split
from flyrank_ml.models.baseline import baseline_score
from flyrank_ml.models.train import predict_scores, train_model
from flyrank_ml.recommendations.engine import build_recommendations
from flyrank_ml.utils.io import anonymize_ids
from helpers import tiny_daily


class TestLeakage(unittest.TestCase):
    def test_temporal_overlap_raises(self):
        with self.assertRaises(AssertionError):
            assert_no_temporal_overlap("2025-03-01", "2025-03-01")
        assert_no_temporal_overlap("2025-03-01", "2025-03-02")

    def test_group_overlap_raises(self):
        with self.assertRaises(AssertionError):
            assert_disjoint_groups(["a", "b"], ["b"])
        assert_disjoint_groups(["a"], ["b"])

    def test_suspicious_feature_flagged(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        X = pd.DataFrame({"leak": y * 2.0, "ok": [1, 5, 2, 1, 7, 3]})
        self.assertEqual(flag_suspicious_features(X, y), ["leak"])

    def test_feature_cutoff_guard(self):
        f = pd.DataFrame({"feature_cutoff": [pd.Timestamp("2025-01-01")], "label_window_start": [pd.Timestamp("2025-01-01")]})
        with self.assertRaises(AssertionError):
            assert_features_before_cutoff(None, f, "2025-01-01")


class TestSplits(unittest.TestCase):
    def test_group_split_disjoint_and_deterministic(self):
        d = pd.DataFrame({"page_id": [f"p{i}" for i in range(200)] * 2, "date": 0})
        a, b = group_time_split(d, valid_fraction=0.3, seed=1)
        assert_disjoint_groups(a["page_id"].unique(), b["page_id"].unique())
        a2, b2 = group_time_split(d, valid_fraction=0.3, seed=1)
        self.assertEqual(set(b["page_id"]), set(b2["page_id"]))
        self.assertTrue(0.2 < b["page_id"].nunique() / 200 < 0.4)

    def test_cutoffs(self):
        c = derive_cutoffs("2025-06-30", 28)
        self.assertEqual(str(c["valid_cutoff"].date()), "2025-06-02")
        self.assertEqual(str(c["train_cutoff"].date()), "2025-05-05")

    def test_bad_fraction(self):
        with self.assertRaises(ValueError):
            assign_groups(pd.Series(["a"]), valid_fraction=1.2)


class TestMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        y = [1, 0, 1, 0]; s = [0.9, 0.8, 0.7, 0.1]
        self.assertEqual(precision_at_k(y, s, 2), 0.5)
        self.assertEqual(precision_at_k(y, s, 1), 1.0)

    def test_perfect_ranking(self):
        y = [1, 1, 0, 0]; s = [4, 3, 2, 1]
        m = evaluate_scores(y, s, ks=(2,))
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["precision_at_2"], 1.0)
        self.assertAlmostEqual(ndcg_at_k(y, s, 2), 1.0)

    def test_single_class_gives_nan_auc(self):
        m = evaluate_scores([0, 0, 0], [1, 2, 3], ks=(2,))
        self.assertTrue(np.isnan(m["roc_auc"]))

    def test_bootstrap_keys(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        d = bootstrap_diff(y, y + rng.normal(0, .3, 200), rng.normal(0, 1, 200), n_boot=50)
        self.assertLessEqual(d["ci_low"], d["ci_high"])


class TestBaselineModel(unittest.TestCase):
    def _feat(self):
        return pd.DataFrame({"clicks_delta_pct": [-0.9, 0.0, 0.5], "clicks_recent": [100, 50, 10], "impressions_recent": [1000, 500, 100]})

    def test_baseline_orders_declining_high_volume_first(self):
        s = baseline_score(self._feat())
        self.assertEqual(int(s.idxmax()), 0)
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_baseline_missing_columns(self):
        with self.assertRaises(ValueError):
            baseline_score(pd.DataFrame({"x": [1]}))

    def test_train_reproducible_and_probabilities(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
        y = (X["a"] + rng.normal(0, .5, 200) > 0).astype(int)
        p1 = predict_scores(train_model(X, y, seed=1), X)
        p2 = predict_scores(train_model(X, y, seed=1), X)
        np.testing.assert_allclose(p1, p2)
        self.assertTrue(((p1 >= 0) & (p1 <= 1)).all())

    def test_train_single_class_raises(self):
        with self.assertRaises(ValueError):
            train_model(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 1]))

    def test_logistic_and_unknown(self):
        X = pd.DataFrame({"a": [0., 1, 2, 3, 4, 5]}); y = pd.Series([0, 0, 0, 1, 1, 1])
        train_model(X, y, "logistic_regression")
        with self.assertRaises(ValueError):
            train_model(X, y, "nope")


class TestRecommendations(unittest.TestCase):
    def _f(self):
        return pd.DataFrame({
            "page_id": ["https://secret.example/a", "b", "c", "d"],
            "clicks_recent": [50, 100, 30, 5], "clicks_prior": [100, 100, 10, 5],
            "clicks_delta_pct": [-0.5, 0.0, 2.0, 0.0], "impressions_recent": [1000, 5000, 300, 50],
            "impressions_delta_pct": [-0.5, 0.0, 0.0, 0.0], "ctr_recent": [0.05, 0.001, 0.1, 0.1],
            "active_days_recent": [20, 20, 20, 3],
        })

    def test_ranking_reason_codes_actions(self):
        r = build_recommendations(self._f(), [0.9, 0.5, 0.1, 0.05], top_k=2)
        self.assertEqual(list(r["rank"]), [1, 2, 3, 4])
        self.assertIn("DECLINING_CLICKS", r.loc[0, "reason_codes"])
        self.assertEqual(r.loc[0, "action"], "REFRESH_REVIEW")
        self.assertIn("VISIBLE_LOW_CTR", r.loc[1, "reason_codes"])
        self.assertEqual(r.loc[1, "action"], "IMPROVE_METADATA_REVIEW")
        self.assertEqual(r.loc[2, "action"], "PROTECT")
        self.assertEqual(r.loc[3, "action"], "MONITOR")
        self.assertTrue(r["explanation"].str.len().gt(0).all())

    def test_no_raw_ids_in_output(self):
        r = build_recommendations(self._f(), [0.9, 0.5, 0.1, 0.05])
        self.assertFalse(r.astype(str).apply(lambda col: col.str.contains("secret.example")).any().any())
        self.assertTrue(r["page_pseudonym"].str.startswith("P-").all())
        self.assertIn("not a confirmed outcome", r.loc[0, "disclaimer"])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            build_recommendations(self._f(), [0.1])

    def test_anonymize_deterministic(self):
        s = pd.Series(["x", "y"])
        pd.testing.assert_series_equal(anonymize_ids(s), anonymize_ids(s))
