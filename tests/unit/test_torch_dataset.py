"""
Unit tests for PyTorch dataset representation in dft_hotspot.ml.torch_dataset.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.ml.torch_dataset import (
    TorchPhysicalDesignDataset,
    build_split_datasets,
    create_dataloaders,
)
from tests.unit.test_dataset_collection import create_mock_manifest, create_mock_sample


@unittest.skipUnless(_HAS_TORCH, "PyTorch required for TorchPhysicalDesignDataset tests")
class TestTorchPhysicalDesignDataset(unittest.TestCase):
    """Test suite for TorchPhysicalDesignDataset and DataLoader generation."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_torch_dataset_item_shapes(self) -> None:
        """Verifies (12, H, W) and (1, H, W) PyTorch tensor shapes."""
        s0_dir = self.base_dir / "samples" / "s0"
        create_mock_sample(s0_dir, "s0", feat_shape=(7, 16, 16), lbl_shape=(8, 16, 16), scan_shape=(5, 16, 16))
        sample0 = PhysicalDesignSample.load(s0_dir)

        s1_dir = self.base_dir / "samples" / "s1"
        create_mock_sample(s1_dir, "s1", feat_shape=(7, 16, 16), lbl_shape=(8, 16, 16), scan_shape=(5, 16, 16))
        sample1 = PhysicalDesignSample.load(s1_dir)

        ds = TorchPhysicalDesignDataset([sample0, sample1], target_channel="total_wirelength_density", preload=True)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.in_channels, 12)

        item0 = ds[0]
        self.assertIn("input", item0)
        self.assertIn("target", item0)
        self.assertIn("sample_id", item0)

        self.assertEqual(item0["input"].shape, torch.Size([12, 16, 16]))
        self.assertEqual(item0["target"].shape, torch.Size([1, 16, 16]))
        self.assertEqual(item0["input"].dtype, torch.float32)
        self.assertEqual(item0["target"].dtype, torch.float32)
        self.assertEqual(item0["sample_id"], "s0")

    def test_build_split_datasets_and_dataloaders(self) -> None:
        """Verifies split dataset construction and DataLoader mini-batch iteration."""
        samples_dir = self.base_dir / "samples"
        for i in range(4):
            create_mock_sample(
                samples_dir / f"s{i}",
                f"s{i}",
                feat_shape=(7, 8, 8),
                lbl_shape=(8, 8, 8),
                scan_shape=(5, 8, 8),
            )
        records = [
            {"sample_id": f"s{i}", "sample_dir": str(samples_dir / f"s{i}"), "status": "success"}
            for i in range(4)
        ]
        manifest_path = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_path)

        splits = {
            "train": ["s0", "s1"],
            "val": ["s2"],
            "test": ["s3"],
        }

        split_ds = build_split_datasets(dataset, splits, target_channel="total_wirelength_density")
        self.assertEqual(len(split_ds["train"]), 2)
        self.assertEqual(len(split_ds["val"]), 1)
        self.assertEqual(len(split_ds["test"]), 1)

        loaders = create_dataloaders(split_ds, batch_size=2)
        train_batch = next(iter(loaders["train"]))
        self.assertEqual(train_batch["input"].shape, torch.Size([2, 12, 8, 8]))
        self.assertEqual(train_batch["target"].shape, torch.Size([2, 1, 8, 8]))

    def test_torch_dataset_multitarget(self) -> None:
        """Verifies multi-target (8, H, W) and custom channel list extraction."""
        s0_dir = self.base_dir / "samples" / "s0"
        create_mock_sample(s0_dir, "s0", feat_shape=(7, 16, 16), lbl_shape=(8, 16, 16), scan_shape=(5, 16, 16))
        sample0 = PhysicalDesignSample.load(s0_dir)

        # Test "all" targets
        ds_all = TorchPhysicalDesignDataset([sample0], target_channel="all", preload=True)
        self.assertEqual(ds_all.out_channels, 8)
        self.assertEqual(len(ds_all.target_channel_names), 8)
        item = ds_all[0]
        self.assertEqual(item["target"].shape, torch.Size([8, 16, 16]))

        # Test custom subset of targets
        ds_sub = TorchPhysicalDesignDataset(
            [sample0],
            target_channel=["via_count_density", "metal2_wirelength_density"],
            preload=True,
        )
        self.assertEqual(ds_sub.out_channels, 2)
        self.assertEqual(ds_sub.target_channel_names, ["via_count_density", "metal2_wirelength_density"])
        item_sub = ds_sub[0]
        self.assertEqual(item_sub["target"].shape, torch.Size([2, 16, 16]))


if __name__ == "__main__":
    unittest.main()
