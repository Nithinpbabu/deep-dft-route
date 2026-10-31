"""
Unit Tests for Stage 7b Multimodal Dataset and DataLoader.
Verifies MultimodalPhysicalDesignDataset, collate_multimodal, and create_multimodal_dataloaders.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch, Data
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.graphs.schema import CircuitGraphData
from dft_hotspot.ml.torch_multimodal_dataset import (
    MultimodalPhysicalDesignDataset,
    collate_multimodal,
    create_multimodal_dataloaders,
)


@unittest.skipUnless(_HAS_TORCH, "PyTorch and torch_geometric required for multimodal dataset tests.")
class TestMultimodalDataset(unittest.TestCase):
    """Unit tests for paired multimodal dataset and collation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.samples_dir = self.root_path / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.num_samples = 6
        self.sample_dirs = []

        # Create mock samples
        for i in range(self.num_samples):
            sname = f"sample_{i:02d}"
            sdir = self.samples_dir / sname
            sdir.mkdir(parents=True, exist_ok=True)
            self.sample_dirs.append(sdir)

            # 1. metadata.json
            with open(sdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"design": f"design_{i % 2}", "sample_id": sname}, f)

            # 2. fused_input.npz (12, 64, 64)
            fused = np.random.rand(12, 64, 64).astype(np.float32)
            np.savez_compressed(sdir / "fused_input.npz", fused_features=fused)

            # 3. features_spatial.npz (7, 64, 64)
            spat = np.random.rand(7, 64, 64).astype(np.float32)
            np.savez_compressed(sdir / "features_spatial.npz", features=spat)

            # 4. labels_routing.npz (8, 64, 64)
            labels = np.random.rand(8, 64, 64).astype(np.float32)
            np.savez_compressed(sdir / "labels_routing.npz", labels=labels)

            # 5. circuit_graph.pt
            x = torch.rand(20 + i * 5, 10, dtype=torch.float32)
            edge_index = torch.randint(0, 20 + i * 5, (2, 40), dtype=torch.long)
            grid_coords = torch.randint(0, 64, (20 + i * 5, 2), dtype=torch.long)
            pos = torch.rand(20 + i * 5, 2, dtype=torch.float32)
            gdata = CircuitGraphData(x=x, edge_index=edge_index, grid_coords=grid_coords, pos=pos)
            torch.save(gdata, sdir / "circuit_graph.pt")

        # 6. splits_v001.json
        splits_content = {
            "train": [f"sample_{i:02d}" for i in range(4)],
            "val": ["sample_04"],
            "test": ["sample_05"],
        }
        with open(self.root_path / "splits_v001.json", "w", encoding="utf-8") as f:
            json.dump(splits_content, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dataset_initialization_and_shapes(self) -> None:
        ds = MultimodalPhysicalDesignDataset(
            sample_dirs=self.sample_dirs,
            target_channel="total_wirelength_density",
            include_scan=True,
            preload=True,
        )
        self.assertEqual(len(ds), self.num_samples)

        item = ds[0]
        self.assertIn("x_spatial", item)
        self.assertIn("graph", item)
        self.assertIn("y", item)
        self.assertIn("sample_id", item)
        self.assertIn("design", item)

        self.assertEqual(item["x_spatial"].shape, (12, 64, 64))
        self.assertEqual(item["y"].shape, (1, 64, 64))
        self.assertTrue(isinstance(item["graph"], (CircuitGraphData, Data)))
        self.assertEqual(item["graph"].x.shape[-1], 10)

    def test_dataset_physical_only_spatial_channels(self) -> None:
        ds = MultimodalPhysicalDesignDataset(
            sample_dirs=self.sample_dirs,
            target_channel="total_wirelength_density",
            include_scan=False,
            preload=True,
        )
        item = ds[0]
        self.assertEqual(item["x_spatial"].shape, (7, 64, 64))

    def test_dataset_multitarget_channels(self) -> None:
        ds = MultimodalPhysicalDesignDataset(
            sample_dirs=self.sample_dirs,
            target_channel="all",
            include_scan=True,
            preload=True,
        )
        item = ds[0]
        self.assertEqual(item["y"].shape, (8, 64, 64))

    def test_collate_multimodal(self) -> None:
        ds = MultimodalPhysicalDesignDataset(
            sample_dirs=self.sample_dirs,
            target_channel="total_wirelength_density",
            include_scan=True,
            preload=True,
        )
        batch_items = [ds[0], ds[1], ds[2]]
        collated = collate_multimodal(batch_items)

        self.assertIn("x_spatial", collated)
        self.assertIn("graph", collated)
        self.assertIn("y", collated)

        self.assertEqual(collated["x_spatial"].shape, (3, 12, 64, 64))
        self.assertEqual(collated["y"].shape, (3, 1, 64, 64))
        self.assertTrue(hasattr(collated["graph"], "batch"))
        self.assertEqual(collated["graph"].num_graphs, 3)
        self.assertEqual(len(collated["sample_ids"]), 3)

    def test_create_multimodal_dataloaders(self) -> None:
        train_l, val_l, test_l, info = create_multimodal_dataloaders(
            dataset_dir=self.root_path,
            target_channel="total_wirelength_density",
            include_scan=True,
            batch_size=2,
            preload=True,
        )
        self.assertEqual(info["train_samples"], 4)
        self.assertEqual(info["val_samples"], 1)
        self.assertEqual(info["test_samples"], 1)

        batch = next(iter(train_l))
        self.assertEqual(batch["x_spatial"].shape[0], 2)
        self.assertEqual(batch["y"].shape[0], 2)
        self.assertEqual(batch["graph"].num_graphs, 2)

    def test_real_expanded_v002_splits(self) -> None:
        """Verifies loading against actual expanded_v002 dataset on disk if present."""
        real_ds_dir = Path("data/processed/datasets/expanded_v002")
        if not real_ds_dir.is_dir():
            self.skipTest("expanded_v002 dataset not found on disk.")

        train_l, val_l, test_l, info = create_multimodal_dataloaders(
            dataset_dir=real_ds_dir,
            target_channel="total_wirelength_density",
            include_scan=True,
            batch_size=4,
            preload=True,
        )
        self.assertEqual(info["train_samples"], 24)
        self.assertEqual(info["val_samples"], 6)
        self.assertEqual(info["test_samples"], 6)
        self.assertEqual(info["total_samples"], 36)

        # Confirm batches from all splits
        for loader, expected_size in [(train_l, 4), (val_l, 4), (test_l, 4)]:
            batch = next(iter(loader))
            self.assertEqual(batch["x_spatial"].shape[1:], (12, 64, 64))
            self.assertEqual(batch["y"].shape[1:], (1, 64, 64))
            self.assertTrue(hasattr(batch["graph"], "batch"))



if __name__ == "__main__":
    unittest.main()
