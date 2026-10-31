"""
Unit and integration tests for classical ML models in dft_hotspot.ml.classical_baselines.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.ml.classical_baselines import (
    MeanBaselineRegressor,
    export_coefficients_csv,
    export_feature_importance_csv,
    train_classical_baselines,
)
from dft_hotspot.ml.tabular_dataset import TabularDataset, load_tabular_dataset
from tests.unit.test_dataset_collection import create_mock_manifest, create_mock_sample


class TestClassicalBaselines(unittest.TestCase):
    """Test suite for classical ML baselines."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _create_synthetic_tabular_dataset(self) -> TabularDataset:
        """Helper to create a small synthetic TabularDataset."""
        s0_dir = self.base_dir / "samples" / "s0"
        s1_dir = self.base_dir / "samples" / "s1"
        create_mock_sample(s0_dir, "s0", feat_shape=(7, 4, 4), lbl_shape=(8, 4, 4), scan_shape=(5, 4, 4), seed=1)
        create_mock_sample(s1_dir, "s1", feat_shape=(7, 4, 4), lbl_shape=(8, 4, 4), scan_shape=(5, 4, 4), seed=2)

        records = [
            {"sample_id": "s0", "sample_dir": str(s0_dir)},
            {"sample_id": "s1", "sample_dir": str(s1_dir)},
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)
        splits = {"train": ["s0"], "val": ["s1"], "test": []}
        return load_tabular_dataset(dataset, splits=splits, target_channel="total_wirelength_density")

    def test_mean_baseline_regressor(self) -> None:
        """Verifies MeanBaselineRegressor predicts exact target mean."""
        X_train = np.zeros((4, 2), dtype=np.float32)
        y_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        reg = MeanBaselineRegressor().fit(X_train, y_train)
        self.assertEqual(reg.mean_value_, 2.5)

        X_test = np.ones((3, 2), dtype=np.float32)
        preds = reg.predict(X_test)
        self.assertEqual(preds.shape, (3,))
        self.assertTrue(np.all(preds == 2.5))

    def test_train_classical_baselines(self) -> None:
        """Verifies training the classical baseline suite on synthetic tabular data."""
        tab_data = self._create_synthetic_tabular_dataset()

        results, skipped = train_classical_baselines(tab_data, random_state=42)
        expected_models = [
            "mean_baseline",
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "random_forest_regressor",
            "gradient_boosting_regressor",
        ]
        for em in expected_models:
            self.assertIn(em, results)
            res = results[em]
            self.assertIn("mae", res.metrics_per_sample["s0"])
            self.assertIn("train", res.metrics_by_split)
            self.assertIn("val", res.metrics_by_split)

        # Coefficients check
        lr_res = results["linear_regression"]
        self.assertIsNotNone(lr_res.coefficients)
        self.assertEqual(len(lr_res.coefficients), 12)

        # Feature importances check
        rf_res = results["random_forest_regressor"]
        self.assertIsNotNone(rf_res.feature_importances)
        self.assertEqual(len(rf_res.feature_importances), 12)

    def test_export_coefficients_and_feature_importances(self) -> None:
        """Verifies CSV exporters for coefficients and feature importances."""
        tab_data = self._create_synthetic_tabular_dataset()
        results, _ = train_classical_baselines(tab_data, random_state=42)

        coef_path = self.base_dir / "coefficients.csv"
        export_coefficients_csv(results, tab_data.feature_names, coef_path)
        self.assertTrue(coef_path.is_file())

        fi_path = self.base_dir / "feature_importance.csv"
        export_feature_importance_csv(results, tab_data.feature_names, fi_path)
        self.assertTrue(fi_path.is_file())

    def test_integration_real_stage4a_dataset(self) -> None:
        """Integration test on real Stage 4a dataset, verifying model rankings and duplicate invariance."""
        manifest_p = Path("data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json")
        splits_p = Path("data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json")
        if not manifest_p.is_file():
            self.skipTest("Stage 4a dataset manifest not found.")

        dataset = PhysicalDesignDataset.load(manifest_p)
        splits = json.loads(splits_p.read_text(encoding="utf-8")) if splits_p.is_file() else None
        tab_data = load_tabular_dataset(dataset, splits=splits, target_channel="total_wirelength_density")

        results, skipped = train_classical_baselines(tab_data, random_state=42)

        # Random forest should achieve high correlation and recall on validation split
        rf_val = results["random_forest_regressor"].metrics_by_split["val"]
        self.assertGreater(rf_val["pearson"]["mean"], 0.75)
        self.assertGreater(rf_val["recall_5pct"]["mean"], 0.70)

        # Duplicate layout invariance: val and test metrics should match
        rf_test = results["random_forest_regressor"].metrics_by_split["test"]
        self.assertAlmostEqual(rf_val["mae"]["mean"], rf_test["mae"]["mean"], places=5)
        self.assertAlmostEqual(rf_val["recall_5pct"]["mean"], rf_test["recall_5pct"]["mean"], places=5)


if __name__ == "__main__":
    unittest.main()
