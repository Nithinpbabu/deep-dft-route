"""
Unit tests for Stage 6b: PyGCircuitGraphDataset and create_pyg_dataloaders.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch_geometric
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from dft_hotspot.graphs.dataset import (
    ALL_ROUTING_CHANNELS,
    PyGCircuitGraphDataset,
    create_pyg_dataloaders,
)


@unittest.skipUnless(_HAS_PYG, "PyTorch and PyTorch Geometric required for graph dataset tests")
class TestPyGCircuitGraphDataset(unittest.TestCase):
    """Tests for PyGCircuitGraphDataset loading, target extraction, and multi-channel slicing."""

    def setUp(self):
        self.sample_dir = Path("data/processed/datasets/expanded_v002/samples/dft_counter_nangate45_u45_clk3p0_ar1p0_chains1")
        if not self.sample_dir.is_dir():
            self.skipTest("Sample directory not found; skipping integration tests")

    def test_single_target_loading(self):
        dataset = PyGCircuitGraphDataset(
            sample_dirs=[self.sample_dir],
            target_channel="total_wirelength_density",
            preload=True,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.num_node_features, 10)
        self.assertEqual(dataset.num_edge_features, 4)
        self.assertEqual(dataset.out_channels, 1)

        data = dataset[0]
        self.assertEqual(data.num_nodes, 84)
        self.assertEqual(data.x.shape, (84, 10))
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_attr.shape[1], 4)
        self.assertEqual(data.pos.shape, (84, 2))
        self.assertEqual(data.grid_coords.shape, (84, 2))
        self.assertEqual(data.y.shape, (1, 64, 64))
        self.assertEqual(data.sample_id, "dft_counter_nangate45_u45_clk3p0_ar1p0_chains1")
        self.assertEqual(data.design, "dft_counter")

    def test_multi_target_all_channels(self):
        dataset = PyGCircuitGraphDataset(
            sample_dirs=[self.sample_dir],
            target_channel="all",
            preload=True,
        )

        self.assertEqual(dataset.out_channels, 8)
        data = dataset[0]
        self.assertEqual(data.y.shape, (8, 64, 64))

    def test_custom_target_channel_subset(self):
        targets = ["total_wirelength_density", "via_count_density"]
        dataset = PyGCircuitGraphDataset(
            sample_dirs=[self.sample_dir],
            target_channel=targets,
            preload=True,
        )

        self.assertEqual(dataset.out_channels, 2)
        data = dataset[0]
        self.assertEqual(data.y.shape, (2, 64, 64))

    def test_invalid_target_channel_raises(self):
        with self.assertRaises(ValueError):
            PyGCircuitGraphDataset(
                sample_dirs=[self.sample_dir],
                target_channel="non_existent_wirelength_channel",
            )


@unittest.skipUnless(_HAS_PYG, "PyTorch and PyTorch Geometric required for graph DataLoader tests")
class TestCreatePyGDataLoaders(unittest.TestCase):
    """Tests for create_pyg_dataloaders factory and batch collation."""

    def setUp(self):
        self.dataset_dir = Path("data/processed/datasets/expanded_v002")
        if not self.dataset_dir.is_dir():
            self.skipTest("expanded_v002 dataset not found")

    def test_dataloader_creation_and_splits(self):
        train_loader, val_loader, test_loader, split_info = create_pyg_dataloaders(
            dataset_dir=self.dataset_dir,
            target_channel="total_wirelength_density",
            batch_size=4,
            preload=True,
        )

        self.assertEqual(len(train_loader.dataset), 24)
        self.assertEqual(len(val_loader.dataset), 6)
        self.assertEqual(len(test_loader.dataset), 6)
        self.assertEqual(split_info["total_samples"], 36)

        # Iterate one batch from train_loader
        batch = next(iter(train_loader))
        self.assertEqual(batch.num_graphs, 4)
        self.assertEqual(batch.y.shape, (4, 1, 64, 64))
        self.assertEqual(batch.x.shape[1], 10)
        self.assertEqual(batch.edge_index.shape[0], 2)
        self.assertEqual(batch.edge_attr.shape[1], 4)
        self.assertEqual(batch.grid_coords.shape[1], 2)
        self.assertEqual(batch.batch.shape[0], batch.x.shape[0])

    def test_leakage_detection_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_splits = {
                "train": ["sample_a", "sample_b"],
                "val": ["sample_c"],
                "test": ["sample_b"],  # Leakage! sample_b is in both train and test
            }
            split_file = Path(tmpdir) / "leaky_splits.json"
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(bad_splits, f)

            with self.assertRaises(ValueError) as ctx:
                create_pyg_dataloaders(
                    dataset_dir=self.dataset_dir,
                    splits_path=split_file,
                )
            self.assertIn("leakage", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
