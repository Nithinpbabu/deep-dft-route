"""
Unit and integration tests for Stage 4b dataset collection, introspection,
statistics computation, duplicate detection, and split policy generation.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from dft_hotspot.data.dataset import (
    FEATURE_META_NAME,
    FEATURE_NPZ_NAME,
    LABEL_META_NAME,
    LABEL_NPZ_NAME,
    SCAN_META_NAME,
    SCAN_NPZ_NAME,
    PhysicalDesignSample,
)
from dft_hotspot.data.dataset_collection import (
    FUSED_META_NAME,
    FUSED_NPZ_NAME,
    PhysicalDesignDataset,
    SampleRecord,
)


def create_mock_sample(
    sample_dir: Path,
    sample_id: str,
    feat_shape: tuple = (7, 16, 16),
    lbl_shape: tuple = (8, 16, 16),
    scan_shape: tuple = (5, 16, 16),
    utilization: float = 0.5,
    clock_period_ns: float = 3.0,
    seed: int = 42,
    corrupt_grid: bool = False,
    identical_to: Optional[Path] = None,
) -> None:
    """Helper to create a complete mock sample on disk."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    if identical_to is not None and identical_to.is_dir():
        # Copy arrays from identical_to
        with np.load(identical_to / FEATURE_NPZ_NAME) as npz:
            feats = npz["features"]
            feat_channels = [str(x) for x in npz["channel_names"]]
        with np.load(identical_to / LABEL_NPZ_NAME) as npz:
            lbls = npz["labels"]
            lbl_channels = [str(x) for x in npz["channel_names"]]
        with np.load(identical_to / SCAN_NPZ_NAME) as npz:
            scans = npz["scan_features"]
            scan_channels = [str(x) for x in npz["channel_names"]]
        with np.load(identical_to / FUSED_NPZ_NAME) as npz:
            fused = npz["fused_features"]
            fused_channels = [str(x) for x in npz["channel_names"]]
    else:
        rng = np.random.RandomState(seed)
        grid_h = feat_shape[1] + (4 if corrupt_grid else 0)
        grid_w = feat_shape[2]
        feats = rng.uniform(0.0, 1.0, (feat_shape[0], grid_h, grid_w)).astype(np.float32)
        lbls = rng.uniform(0.0, 1.0, (lbl_shape[0], grid_h, grid_w)).astype(np.float32)
        scans = rng.uniform(0.0, 1.0, (scan_shape[0], grid_h, grid_w)).astype(np.float32)
        fused = np.concatenate([feats, scans], axis=0).astype(np.float32)

        feat_channels = [
            "cell_density",
            "sequential_cell_density",
            "combinational_cell_density",
            "pin_density",
            "blockage_or_macro_density",
            "normalized_x",
            "normalized_y",
        ]
        lbl_channels = [
            "total_wirelength_density",
            "horizontal_wirelength_density",
            "vertical_wirelength_density",
            "via_count_density",
            "metal1_wirelength_density",
            "metal2_wirelength_density",
            "metal3_wirelength_density",
            "metal4plus_wirelength_density",
        ]
        scan_channels = [
            "scan_cell_density",
            "scan_chain_endpoint_density",
            "scan_chain_segment_density",
            "scan_chain_order_index",
            "scan_chain_local_wirelength_proxy",
        ]
        fused_channels = [f"physical/{c}" for c in feat_channels] + [f"scan/{c}" for c in scan_channels]

    # Save NPZs
    np.savez_compressed(
        sample_dir / FEATURE_NPZ_NAME,
        features=feats,
        raw_features=feats.copy(),
        channel_names=feat_channels,
    )
    np.savez_compressed(
        sample_dir / LABEL_NPZ_NAME,
        labels=lbls,
        raw_labels=lbls.copy(),
        channel_names=lbl_channels,
    )
    np.savez_compressed(
        sample_dir / SCAN_NPZ_NAME,
        scan_features=scans,
        raw_scan_features=scans.copy(),
        channel_names=scan_channels,
    )
    np.savez_compressed(
        sample_dir / FUSED_NPZ_NAME,
        fused_features=fused,
        raw_fused_features=fused.copy(),
        channel_names=fused_channels,
    )

    # Save JSON metadata
    grid_res = [feats.shape[1], feats.shape[2]]
    feat_meta = {
        "design": "test_uart",
        "pdk": "nangate45",
        "grid_size": grid_res,
        "tensor_shape": list(feats.shape),
        "channel_names": feat_channels,
        "config": {"target_utilization": utilization, "clock_period_ns": clock_period_ns},
        "design_statistics": {"total_components": 150, "total_nets": 180},
    }
    (sample_dir / FEATURE_META_NAME).write_text(json.dumps(feat_meta), encoding="utf-8")

    lbl_meta = {
        "design": "test_uart",
        "pdk": "nangate45",
        "grid_size": grid_res,
        "tensor_shape": list(lbls.shape),
        "channel_names": lbl_channels,
        "routing_statistics": {
            "total_wirelength_um": 1200.0,
            "total_vias": 900,
            "total_nets": 180,
            "routed_nets": 150,
        },
    }
    (sample_dir / LABEL_META_NAME).write_text(json.dumps(lbl_meta), encoding="utf-8")

    scan_meta = {
        "schema_version": "scan_features_v001",
        "scan_source": "synthetic_proxy",
        "design": "test_uart",
        "platform": "nangate45",
        "grid_size": grid_res,
        "tensor_shape": list(scans.shape),
        "channel_names": scan_channels,
        "total_scan_cells": 30,
        "total_scan_wirelength_um": 280.0,
    }
    (sample_dir / SCAN_META_NAME).write_text(json.dumps(scan_meta), encoding="utf-8")

    fused_meta = {
        "schema_version": "fused_input_v001",
        "fused_shape": list(fused.shape),
        "channel_names": fused_channels,
    }
    (sample_dir / FUSED_META_NAME).write_text(json.dumps(fused_meta), encoding="utf-8")


def create_mock_manifest(
    manifest_dir: Path,
    sample_records: List[Dict[str, Any]],
    dataset_id: str = "mock_test_dataset",
) -> Path:
    """Creates a mock dataset_manifest.json."""
    manifest_data = {
        "dataset_id": dataset_id,
        "schema_version": "dataset_manifest_v001",
        "design": "test_uart",
        "platform": "nangate45",
        "grid_size": 16,
        "scan_source": "synthetic_proxy",
        "sample_count": len(sample_records),
        "success_count": len(sample_records),
        "failure_count": 0,
        "samples": sample_records,
        "failures": [],
    }
    manifest_path = manifest_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


class TestDatasetCollection(unittest.TestCase):
    """Unit tests for PhysicalDesignDataset and dataset introspection."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_load_synthetic_manifest(self) -> None:
        """Verifies loading a synthetic manifest and instantiating sample records."""
        s0_dir = self.base_dir / "samples" / "sample_0"
        s1_dir = self.base_dir / "samples" / "sample_1"
        create_mock_sample(s0_dir, "sample_0", seed=10)
        create_mock_sample(s1_dir, "sample_1", seed=20)

        records = [
            {
                "sample_id": "sample_0",
                "sample_dir": str(s0_dir),
                "status": "success",
                "config": {"utilization": 0.45, "clock_period_ns": 3.0},
            },
            {
                "sample_id": "sample_1",
                "sample_dir": str(s1_dir),
                "status": "success",
                "config": {"utilization": 0.55, "clock_period_ns": 3.0},
            },
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)

        dataset = PhysicalDesignDataset.load(manifest_p)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.dataset_id, "mock_test_dataset")
        self.assertIn("sample_0", dataset.keys())
        self.assertIn("sample_1", dataset.keys())

        # Test indexing
        rec0 = dataset[0]
        self.assertEqual(rec0.sample_id, "sample_0")
        self.assertIsNotNone(rec0.sample)
        self.assertEqual(rec0.fused_tensor.shape, (12, 16, 16))
        self.assertIsNotNone(rec0.fused_sha256)
        self.assertIsNotNone(rec0.labels_sha256)

    def test_dataset_validation_passes(self) -> None:
        """Verifies validation passes on valid multi-sample collection."""
        s0_dir = self.base_dir / "samples" / "sample_0"
        s1_dir = self.base_dir / "samples" / "sample_1"
        create_mock_sample(s0_dir, "sample_0", seed=10)
        create_mock_sample(s1_dir, "sample_1", seed=20)

        records = [
            {"sample_id": "sample_0", "sample_dir": str(s0_dir), "status": "success"},
            {"sample_id": "sample_1", "sample_dir": str(s1_dir), "status": "success"},
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        report = dataset.validate(require_scan=True, require_fused=True)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["passed_samples"], 2)
        self.assertEqual(report["failed_samples"], 0)

    def test_dataset_validation_detects_grid_mismatch(self) -> None:
        """Verifies validation fails when samples have conflicting spatial grids."""
        s0_dir = self.base_dir / "samples" / "sample_0"
        s1_dir = self.base_dir / "samples" / "sample_1"
        create_mock_sample(s0_dir, "sample_0", seed=10)
        create_mock_sample(s1_dir, "sample_1", seed=20, corrupt_grid=True)

        records = [
            {"sample_id": "sample_0", "sample_dir": str(s0_dir), "status": "success"},
            {"sample_id": "sample_1", "sample_dir": str(s1_dir), "status": "success"},
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        with self.assertRaises(ValueError) as ctx:
            dataset.validate(raise_on_error=True)
        self.assertIn("Grid shape", str(ctx.exception))

    def test_duplicate_detection_exact_and_near(self) -> None:
        """Verifies exact hash duplicate grouping and reporting."""
        s0_dir = self.base_dir / "samples" / "sample_0"
        s1_dir = self.base_dir / "samples" / "sample_1"
        s2_dir = self.base_dir / "samples" / "sample_2"

        create_mock_sample(s0_dir, "sample_0", seed=100)
        # sample_1 is exact clone of sample_0
        create_mock_sample(s1_dir, "sample_1", identical_to=s0_dir)
        # sample_2 is distinct
        create_mock_sample(s2_dir, "sample_2", seed=200)

        records = [
            {
                "sample_id": "sample_0",
                "sample_dir": str(s0_dir),
                "config": {"utilization": 0.45, "clock_period_ns": 3.0},
            },
            {
                "sample_id": "sample_1",
                "sample_dir": str(s1_dir),
                "config": {"utilization": 0.45, "clock_period_ns": 5.0},
            },
            {
                "sample_id": "sample_2",
                "sample_dir": str(s2_dir),
                "config": {"utilization": 0.55, "clock_period_ns": 3.0},
            },
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        dup_report = dataset.detect_duplicates()
        self.assertEqual(dup_report["total_samples"], 3)
        self.assertEqual(dup_report["unique_layouts"], 2)
        self.assertEqual(dup_report["duplicate_group_count"], 1)

        group0 = dup_report["duplicate_groups"][0]
        self.assertEqual(group0["sample_ids"], ["sample_0", "sample_1"])
        self.assertTrue(group0["is_exact"])
        self.assertEqual(group0["max_abs_diff_fused"], 0.0)
        self.assertEqual(group0["max_abs_diff_labels"], 0.0)
        self.assertEqual(group0["shared_config"]["utilization"], 0.45)
        self.assertEqual(group0["differing_config"]["clock_period_ns"], [3.0, 5.0])

    def test_compute_statistics(self) -> None:
        """Verifies dataset-level channel and sample statistics calculation."""
        s0_dir = self.base_dir / "samples" / "sample_0"
        s1_dir = self.base_dir / "samples" / "sample_1"
        create_mock_sample(s0_dir, "sample_0", seed=1)
        create_mock_sample(s1_dir, "sample_1", seed=2)

        records = [
            {"sample_id": "sample_0", "sample_dir": str(s0_dir), "config": {"utilization": 0.45}},
            {"sample_id": "sample_1", "sample_dir": str(s1_dir), "config": {"utilization": 0.55}},
        ]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        stats = dataset.compute_statistics()
        self.assertEqual(stats["schema_version"], "dataset_statistics_v001")
        self.assertEqual(stats["sample_count"], 2)
        self.assertIn("fused_input_channels", stats["channel_statistics"])
        self.assertIn("routing_label_channels", stats["channel_statistics"])

        # Check channel stats properties
        fused_stats = stats["channel_statistics"]["fused_input_channels"]
        self.assertEqual(len(fused_stats), 12)
        for cname, cstat in fused_stats.items():
            self.assertIn("min", cstat)
            self.assertIn("max", cstat)
            self.assertIn("mean", cstat)
            self.assertIn("std", cstat)
            self.assertIn("nonzero_fraction", cstat)

        # Check per-sample stats
        self.assertIn("sample_0", stats["per_sample_stats"])
        self.assertEqual(stats["per_sample_stats"]["sample_0"]["total_wirelength_um"], 1200.0)
        self.assertEqual(stats["per_sample_stats"]["sample_0"]["total_scan_cells"], 30)

    def test_create_splits_smoke_holdout(self) -> None:
        """Verifies smoke_config_holdout split generation logic and warnings."""
        sample_records = []
        # Create 6 samples across 3 utilizations (0.45, 0.55, 0.65)
        for idx, (u, clk) in enumerate([
            (0.45, 3.0), (0.45, 5.0),
            (0.55, 3.0), (0.55, 5.0),
            (0.65, 3.0), (0.65, 5.0),
        ]):
            sid = f"s_u{int(u*100)}_clk{int(clk)}"
            sdir = self.base_dir / "samples" / sid
            create_mock_sample(sdir, sid, seed=idx, utilization=u, clock_period_ns=clk)
            sample_records.append({
                "sample_id": sid,
                "sample_dir": str(sdir),
                "config": {"utilization": u, "clock_period_ns": clk},
            })

        manifest_p = create_mock_manifest(self.base_dir, sample_records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        splits = dataset.create_splits(policy="smoke_config_holdout")
        self.assertEqual(splits["schema_version"], "splits_v001")
        self.assertEqual(splits["split_type"], "smoke_config_holdout")
        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["val"]), 1)
        self.assertEqual(len(splits["test"]), 1)
        self.assertEqual(splits["val"][0], "s_u65_clk3")
        self.assertEqual(splits["test"][0], "s_u65_clk5")
        self.assertTrue(any("Single design only" in lim for lim in splits["limitations"]))

    def test_export_dataset_analysis(self) -> None:
        """Verifies exporting all 4 analysis artifacts to disk."""
        s0_dir = self.base_dir / "samples" / "sample_0"
        create_mock_sample(s0_dir, "sample_0", seed=5)
        records = [{"sample_id": "sample_0", "sample_dir": str(s0_dir)}]
        manifest_p = create_mock_manifest(self.base_dir, records)
        dataset = PhysicalDesignDataset.load(manifest_p)

        out_dir = self.base_dir / "analysis_out"
        written = dataset.export_dataset_analysis(output_dir=out_dir, write_splits=True)

        self.assertIn("dataset_statistics", written)
        self.assertIn("duplicate_report", written)
        self.assertIn("splits", written)
        self.assertIn("dataset_analysis_summary", written)

        for name, p in written.items():
            self.assertTrue(p.is_file(), f"Expected artifact {name} at {p} to exist")

        # Check JSON parseability
        with open(written["dataset_statistics"], "r", encoding="utf-8") as f:
            stats_json = json.load(f)
            self.assertEqual(stats_json["schema_version"], "dataset_statistics_v001")

        with open(written["duplicate_report"], "r", encoding="utf-8") as f:
            dup_json = json.load(f)
            self.assertEqual(dup_json["schema_version"], "duplicate_report_v001")

        with open(written["splits"], "r", encoding="utf-8") as f:
            splits_json = json.load(f)
            self.assertEqual(splits_json["schema_version"], "splits_v001")

    def test_missing_manifest_error(self) -> None:
        """Verifies FileNotFoundError on missing manifest."""
        with self.assertRaises(FileNotFoundError):
            PhysicalDesignDataset.load(self.base_dir / "nonexistent_manifest.json")

    def test_integration_real_stage4a_dataset(self) -> None:
        """Integration test on real Stage 4a dataset, skipped if not present."""
        real_manifest = Path("data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json")
        if not real_manifest.is_file():
            self.skipTest(f"Real Stage 4a manifest not found at {real_manifest}")

        dataset = PhysicalDesignDataset.load(real_manifest, load_tensors=True, load_scan=True)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.dataset_id, "uart_nangate45_smoke_v001")
        self.assertEqual(dataset.design, "uart_tx")
        self.assertEqual(dataset.platform, "nangate45")
        self.assertEqual(dataset.grid_size, 64)

        # Validate
        val_report = dataset.validate(require_scan=True, require_fused=True)
        self.assertEqual(val_report["status"], "pass")
        self.assertEqual(val_report["passed_samples"], 6)

        # Duplicate detection on real dataset
        dup_report = dataset.detect_duplicates()
        self.assertEqual(dup_report["total_samples"], 6)
        self.assertEqual(dup_report["unique_layouts"], 3)
        self.assertEqual(dup_report["duplicate_group_count"], 3)

        # All 3 groups must have exact match (diff == 0.0)
        for g in dup_report["duplicate_groups"]:
            self.assertTrue(g["is_exact"])
            self.assertEqual(g["max_abs_diff_fused"], 0.0)
            self.assertEqual(g["max_abs_diff_labels"], 0.0)

        # Splits
        splits = dataset.create_splits(policy="smoke_config_holdout")
        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["val"]), 1)
        self.assertEqual(len(splits["test"]), 1)
        self.assertIn("Duplicate layout alert", splits["limitations"][2])


if __name__ == "__main__":
    unittest.main()
