"""
Unit tests for spatial evaluation metrics in dft_hotspot.evaluation.metrics.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from dft_hotspot.evaluation.metrics import (
    compute_all_metrics,
    dice_topk,
    iou_topk,
    mae,
    pearson_corr,
    precision_at_k,
    rmse,
    sanitize_metrics_for_json,
    spearman_corr,
    topk_mask,
    topk_recall,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for 2D spatial evaluation metrics."""

    def test_mae_and_rmse(self) -> None:
        """Verifies MAE and RMSE on known arrays."""
        pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = np.array([[2.0, 2.0], [2.0, 6.0]], dtype=np.float32)

        # diffs: [1.0, 0.0, 1.0, 2.0] -> sum = 4.0, mean = 1.0
        self.assertAlmostEqual(mae(pred, target), 1.0, places=5)

        # sq diffs: [1.0, 0.0, 1.0, 4.0] -> sum = 6.0, mean = 1.5, sqrt(1.5) ~= 1.22474487
        self.assertAlmostEqual(rmse(pred, target), math.sqrt(1.5), places=5)

    def test_shape_mismatch_raises_error(self) -> None:
        """Verifies ValueError when pred and target have different shapes."""
        pred = np.zeros((2, 2), dtype=np.float32)
        target = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            mae(pred, target)

    def test_pearson_and_spearman_correlation(self) -> None:
        """Verifies correlation metrics on perfectly correlated and anti-correlated arrays."""
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        pred_pos = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
        pred_neg = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float32)

        self.assertAlmostEqual(pearson_corr(pred_pos, target), 1.0, places=5)
        self.assertAlmostEqual(pearson_corr(pred_neg, target), -1.0, places=5)
        self.assertAlmostEqual(spearman_corr(pred_pos, target), 1.0, places=5)
        self.assertAlmostEqual(spearman_corr(pred_neg, target), -1.0, places=5)

    def test_constant_array_correlation_handling(self) -> None:
        """Verifies Pearson and Spearman safely return NaN on zero-variance arrays without crashing."""
        const_pred = np.ones((4, 4), dtype=np.float32) * 5.0
        target = np.random.RandomState(42).uniform(0, 1, (4, 4)).astype(np.float32)

        with self.assertWarns(UserWarning):
            r_val = pearson_corr(const_pred, target)
            self.assertTrue(math.isnan(r_val))

        with self.assertWarns(UserWarning):
            rho_val = spearman_corr(const_pred, target)
            self.assertTrue(math.isnan(rho_val))

    def test_topk_mask(self) -> None:
        """Verifies top-k hotspot mask extraction."""
        # 100 elements: 0 to 99
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)

        # Top 5% should have 5 elements (95, 96, 97, 98, 99)
        m_5 = topk_mask(arr, 0.05)
        self.assertEqual(m_5.shape, (10, 10))
        self.assertEqual(np.count_nonzero(m_5), 5)
        self.assertTrue(np.all(m_5[arr >= 95.0]))

        # Top 10%
        m_10 = topk_mask(arr, 10.0)  # percentage as 10.0
        self.assertEqual(np.count_nonzero(m_10), 10)
        self.assertTrue(np.all(m_10[arr >= 90.0]))

    def test_constant_array_topk_mask(self) -> None:
        """Verifies constant array top-k mask returns all False."""
        arr = np.zeros((10, 10), dtype=np.float32)
        m = topk_mask(arr, 0.05)
        self.assertEqual(np.count_nonzero(m), 0)

    def test_topk_overlap_metrics(self) -> None:
        """Verifies Recall, Precision, IoU, and Dice on known overlapping sets."""
        # 10 elements:
        # Target top 2: indices 8, 9
        target = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 10], dtype=np.float32)
        # Pred top 2: indices 7, 8 (1 overlap at index 8)
        pred = np.array([0, 1, 2, 3, 4, 5, 6, 10, 10, 7], dtype=np.float32)

        # k = 20% -> 2 elements
        # Intersection = 1 element (index 8)
        # Recall: 1 / 2 = 0.5
        self.assertAlmostEqual(topk_recall(pred, target, 0.20), 0.5, places=5)
        # Precision: 1 / 2 = 0.5
        self.assertAlmostEqual(precision_at_k(pred, target, 0.20), 0.5, places=5)
        # Union: 3 elements (7, 8, 9) -> IoU = 1 / 3 ~= 0.33333
        self.assertAlmostEqual(iou_topk(pred, target, 0.20), 1.0 / 3.0, places=5)
        # Dice: 2 * 1 / (2 + 2) = 0.5
        self.assertAlmostEqual(dice_topk(pred, target, 0.20), 0.5, places=5)

    def test_compute_all_metrics(self) -> None:
        """Verifies compute_all_metrics produces all expected keys."""
        rng = np.random.RandomState(123)
        p = rng.uniform(0, 1, (16, 16)).astype(np.float32)
        t = rng.uniform(0, 1, (16, 16)).astype(np.float32)

        m = compute_all_metrics(p, t)
        expected_keys = [
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "recall_5pct",
            "precision_5pct",
            "iou_5pct",
            "dice_5pct",
            "recall_10pct",
            "precision_10pct",
            "iou_10pct",
            "dice_10pct",
        ]
        for k in expected_keys:
            self.assertIn(k, m)
            self.assertIsInstance(m[k], float)

    def test_sanitize_metrics_for_json(self) -> None:
        """Verifies NaN and Inf values are safely converted to None for JSON."""
        raw = {
            "valid_float": 0.123,
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "nested": {"inner_nan": float("nan"), "inner_ok": 10},
        }
        clean = sanitize_metrics_for_json(raw)
        self.assertEqual(clean["valid_float"], 0.123)
        self.assertIsNone(clean["nan_val"])
        self.assertIsNone(clean["inf_val"])
        self.assertIsNone(clean["nested"]["inner_nan"])
        self.assertEqual(clean["nested"]["inner_ok"], 10)


if __name__ == "__main__":
    unittest.main()
