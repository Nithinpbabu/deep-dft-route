"""
Unit and integration tests for tabular dataset conversion in dft_hotspot.ml.tabular_dataset.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.ml.tabular_dataset import (
    SampleTabularData,
    TabularDataset,
    load_tabular_dataset,
    sample_to_grid_rows,
)
from tests.unit.test_dataset_collection import create_mock_manifest, create_mock_sample


class TestTabularDataset(unittest.TestCase):
    """Test suite for spatial tensor to tabular row conversion."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_sample_to_grid_rows(self) -> None:
        """Verifies (C, H, W) to (H*W, C) flattening and coordinate indices."""
        sample_dir = self.base_dir / "samples" / "s0"
        create_mock_sample(sample_dir, "s0", feat_shape=(7, 8, 8), lbl_shape=(8, 8, 8), scan_shape=(5, 8, 8))
        sample = PhysicalDesignSample.load(sample_dir)

        tab_data = sample_to_grid_rows(sample, target_channel="total_wirelength_density", split="train")

        self.assertEqual(tab_data.sample_id, "s0")
        self.assertEqual(tab_data.split, "train")
        self.assertEqual(tab_data.X.shape, (64, 12))
        self.assertEqual(tab_data.y.shape, (64,))
        self.assertEqual(len(tab_data.row_indices), 64)
        self.assertEqual(len(tab_data.col_indices), 64)
        self.assertEqual(tab_data.grid_shape, (8, 8))
        self.assertEqual(len(tab_data.feature_names), 12)

        # Check row/col coordinate correspondence
        self.assertEqual(tab_data.row_indices[0], 0)
        self.assertEqual(tab_data.col_indices[0], 0)
        self.assertEqual(tab_data.row_indices[7], 0)
        self.assertEqual(tab_data.col_indices[7], 7)
        self.assertEqual(tab_data.row_indices[8], 1)
        self.assertEqual(tab_data.col_indices[8], 0)

    def test_anti_leakage_protection(self) -> None:
        """Verifies ValueError when post-route routing labels are detected in features."""
        sample_dir = self.base_dir / "samples" / "s_leak"
        create_mock_sample(sample_dir, "s_leak")
        sample = PhysicalDesignSample.load(sample_dir)

        # Inject illegal post-route label name into feature channels
        sample.feature_channel_names.append("metal1_wirelength_density")
        with self.assertRaises(ValueError) as ctx:
            sample_to_grid_rows(sample)
        self.assertIn("Target leakage detected", str(ctx.exception))

    def test_missing_target_channel_error(self) -> None:
        """Verifies KeyError when target channel does not exist in labels."""
        sample_dir = self.base_dir / "samples" / "s0"
        create_mock_sample(sample_dir, "s0")
        sample = PhysicalDesignSample.load(sample_dir)

        with self.assertRaises(KeyError):
            sample_to_grid_rows(sample, target_channel="non_existent_routing_channel")

    def test_load_tabular_dataset_splits(self) -> None:
        """Verifies assembling full tabular dataset with train, val, and test splits."""
        s0_dir = self.base_dir / "samples" / "s0"
        s1_dir = self.base_dir / "samples" / "s1"
        s2_dir = self.base_dir / "samples" / "s2"
        create_mock_sample(s0_dir, "s0", feat_shape=(7, 4, 4), lbl_shape=(8, 4, 4), scan_shape=(5, 4, 4))
        create_mock_sample(s1_dir, "s1", feat_shape=(7, 4, 4), lbl_shape=(8, 4, 4), scan_shape=(5, 4, 4))
        create_mock_sample(s2_dir, "s2", feat_shape=(7, 4, 4), lbl_shape=(8, 4, 4), scan_shape=(5, 4, 4))

        records = [
            {"sample_id": "s0", "sample_dir": str(s0_dir)},
            {"sample_id": "s1", "sample_dir": str(s1_dir)},
            {"sample_id": "s2", "sample_dir": str(s2_dir)},
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        splits = {"train": ["s0"], "val": ["s1"], "test": ["s2"]}
        tab_ds = load_tabular_dataset(dataset, splits=splits, target_channel="total_wirelength_density")

        self.assertEqual(tab_ds.num_features, 12)
        # Each sample has 4 * 4 = 16 cells
        self.assertEqual(tab_ds.X_train.shape, (16, 12))
        self.assertEqual(tab_ds.y_train.shape, (16,))
        self.assertEqual(tab_ds.X_val.shape, (16, 12))
        self.assertEqual(tab_ds.y_val.shape, (16,))
        self.assertEqual(tab_ds.X_test.shape, (16, 12))
        self.assertEqual(tab_ds.y_test.shape, (16,))
        self.assertEqual(tab_ds.total_cells, 48)

    def test_integration_real_stage4a_dataset(self) -> None:
        """Integration test on real Stage 4a dataset, verifying shapes and feature names."""
        manifest_p = Path("data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json")
        splits_p = Path("data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json")
        if not manifest_p.is_file():
            self.skipTest("Stage 4a dataset manifest not found.")

        dataset = PhysicalDesignDataset.load(manifest_p)
        import json
        splits = json.loads(splits_p.read_text(encoding="utf-8")) if splits_p.is_file() else None

        tab_ds = load_tabular_dataset(dataset, splits=splits, target_channel="total_wirelength_density")
        self.assertEqual(tab_ds.num_features, 12)
        self.assertEqual(tab_ds.grid_shape, (64, 64))

        # 4 train samples = 16384 rows, 1 val = 4096 rows, 1 test = 4096 rows
        self.assertEqual(tab_ds.X_train.shape, (16384, 12))
        self.assertEqual(tab_ds.y_train.shape, (16384,))
        self.assertEqual(tab_ds.X_val.shape, (4096, 12))
        self.assertEqual(tab_ds.y_val.shape, (4096,))
        self.assertEqual(tab_ds.X_test.shape, (4096, 12))
        self.assertEqual(tab_ds.y_test.shape, (4096,))
        self.assertEqual(tab_ds.total_cells, 24576)


if __name__ == "__main__":
    unittest.main()
