"""
Unit and integration tests for non-learning heuristic baselines in dft_hotspot.evaluation.heuristics.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.evaluation.heuristics import (
    HeuristicBaseline,
    build_heuristic_predictions,
    evaluate_dataset_heuristics,
    evaluate_heuristics_on_sample,
    get_standard_heuristics,
)
from tests.unit.test_dataset_collection import create_mock_manifest, create_mock_sample


class TestHeuristicBaselines(unittest.TestCase):
    """Test suite for non-learning heuristic baselines and evaluation workflows."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_standard_heuristics_definitions(self) -> None:
        """Verifies get_standard_heuristics returns all 7 standard baselines."""
        baselines = get_standard_heuristics()
        self.assertEqual(len(baselines), 7)
        expected_names = [
            "pin_density",
            "cell_density",
            "sequential_cell_density",
            "scan_cell_density",
            "scan_segment_density",
            "scan_local_wirelength_proxy",
            "simple_weighted_sum",
        ]
        self.assertEqual([b.name for b in baselines], expected_names)

    def test_predict_single_channel_and_normalization(self) -> None:
        """Verifies prediction extraction from a single channel with max-scaling."""
        fused = np.zeros((3, 4, 4), dtype=np.float32)
        fused[0, 0, 0] = 5.0
        fused[0, 1, 1] = 2.5
        names = ["physical/pin_density", "physical/cell_density", "scan/scan_cell_density"]

        baseline = HeuristicBaseline(
            name="test_pin",
            description="Test pin baseline",
            channel_name="physical/pin_density",
            normalize=True,
        )
        pred = baseline.predict(fused, names)
        self.assertEqual(pred.shape, (4, 4))
        self.assertAlmostEqual(float(pred[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(pred[1, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(np.min(pred)), 0.0, places=5)

    def test_predict_weighted_sum(self) -> None:
        """Verifies prediction computation using a weighted sum of multiple channels."""
        fused = np.zeros((2, 4, 4), dtype=np.float32)
        fused[0, 0, 0] = 2.0  # ch0
        fused[1, 0, 0] = 1.0  # ch1
        names = ["physical/pin_density", "physical/cell_density"]

        baseline = HeuristicBaseline(
            name="test_sum",
            description="Test weighted sum",
            weights={"physical/pin_density": 0.5, "physical/cell_density": 0.5},
            normalize=False,
        )
        pred = baseline.predict(fused, names)
        # 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        self.assertAlmostEqual(float(pred[0, 0]), 1.5, places=5)

    def test_missing_channel_raises_key_error(self) -> None:
        """Verifies KeyError when required channel is missing from fused tensor names."""
        fused = np.zeros((1, 4, 4), dtype=np.float32)
        names = ["physical/unrelated_channel"]
        baseline = HeuristicBaseline(
            name="bad_baseline",
            description="Bad",
            channel_name="physical/pin_density",
        )
        with self.assertRaises(KeyError):
            baseline.predict(fused, names)

    def test_evaluate_sample_heuristics(self) -> None:
        """Verifies evaluating all 7 baselines on a single mock sample."""
        sample_dir = self.base_dir / "samples" / "sample_0"
        create_mock_sample(sample_dir, "sample_0", seed=42)
        sample = PhysicalDesignSample.load(sample_dir)

        results = evaluate_heuristics_on_sample(sample, target_channel="total_wirelength_density")
        self.assertEqual(len(results), 7)
        for b_name, metrics in results.items():
            self.assertIn("mae", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("recall_5pct", metrics)
            self.assertIn("precision_5pct", metrics)

    def test_evaluate_dataset_heuristics(self) -> None:
        """Verifies dataset-level evaluation and split aggregation on synthetic data."""
        s0_dir = self.base_dir / "samples" / "s0"
        s1_dir = self.base_dir / "samples" / "s1"
        create_mock_sample(s0_dir, "s0", seed=10)
        create_mock_sample(s1_dir, "s1", seed=20)

        records = [
            {"sample_id": "s0", "sample_dir": str(s0_dir)},
            {"sample_id": "s1", "sample_dir": str(s1_dir)},
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        splits = {"train": ["s0"], "val": ["s1"], "test": []}
        report = evaluate_dataset_heuristics(dataset, splits=splits)

        self.assertEqual(report["sample_count"], 2)
        self.assertIn("all", report["by_split"])
        self.assertIn("train", report["by_split"])
        self.assertIn("val", report["by_split"])
        self.assertIn("summary", report)
        self.assertIsNotNone(report["summary"]["best_baseline_by_mae"])

    def test_integration_real_stage4a_dataset(self) -> None:
        """Integration test running heuristic evaluation on the real Stage 4a dataset."""
        manifest_p = Path("data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json")
        splits_p = Path("data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json")
        if not manifest_p.is_file():
            self.skipTest(f"Real Stage 4a manifest not found at {manifest_p}")

        dataset = PhysicalDesignDataset.load(manifest_p)
        splits = json.loads(splits_p.read_text(encoding="utf-8")) if splits_p.is_file() else None

        report = evaluate_dataset_heuristics(dataset, splits=splits, target_channel="total_wirelength_density")
        self.assertEqual(report["sample_count"], 6)

        # On the val partition, simple_weighted_sum should have strong correlation and top recall
        val_stats = report["by_split"]["val"]["simple_weighted_sum"]
        self.assertGreater(val_stats["pearson"]["mean"], 0.60)
        self.assertGreater(val_stats["recall_5pct"]["mean"], 0.50)

        # Confirm duplicate invariance: val and test partition metrics must be equal
        test_stats = report["by_split"]["test"]["simple_weighted_sum"]
        self.assertAlmostEqual(val_stats["mae"]["mean"], test_stats["mae"]["mean"], places=6)
        self.assertAlmostEqual(val_stats["recall_5pct"]["mean"], test_stats["recall_5pct"]["mean"], places=6)


if __name__ == "__main__":
    unittest.main()
