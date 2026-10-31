"""
Unit and integration tests for Stage 2c paired dataset representation.
Pairs physical layout feature tensors (intended to be pre-route-compatible)
with post-route routing label ground-truth targets.
Verifies:
1. Loading real UART sample (skipping gracefully if missing)
2. Missing file error handling
3. Mismatched spatial grid shape error handling
4. Channel count mismatch error handling
5. Finite-value verification (NaN / Inf)
6. Metadata inconsistency error handling (mismatched run dirs and designs)
7. Optional metadata missing fields handling (allowed with UserWarning)
8. Summary dictionary and formatting output fields
"""

from __future__ import annotations

import json
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
from dft_hotspot.features.extractor import CHANNEL_NAMES
from dft_hotspot.labels.routing_extractor import ROUTING_CHANNEL_NAMES


def add_synthetic_scan_features(
    sample_dir: Path,
    scan_shape: tuple = (5, 16, 16),
    scan_channels: Optional[List[str]] = None,
    scan_meta: Optional[Dict[str, Any]] = None,
    corrupt_scan_val: Optional[float] = None,
    scan_source: str = "synthetic_proxy",
) -> None:
    """Helper to add synthetic scan feature artifacts to sample_dir."""
    if scan_channels is None:
        scan_channels = [
            "scan_cell_density",
            "scan_chain_endpoint_density",
            "scan_chain_segment_density",
            "scan_chain_order_index",
            "scan_chain_local_wirelength_proxy",
        ][: scan_shape[0]]
        if len(scan_channels) < scan_shape[0]:
            scan_channels.extend([f"scan_ch_{i}" for i in range(len(scan_channels), scan_shape[0])])

    scans = np.zeros(scan_shape, dtype=np.float32)
    if corrupt_scan_val is not None:
        scans[0, 0, 0] = corrupt_scan_val

    np.savez_compressed(
        sample_dir / SCAN_NPZ_NAME,
        scan_features=scans,
        raw_scan_features=scans.copy(),
        channel_names=scan_channels,
    )

    if scan_meta is None:
        scan_meta = {
            "schema_version": "scan_feature_metadata_v001",
            "scan_source": scan_source,
            "grid_size": [scan_shape[1], scan_shape[2]],
            "tensor_shape": list(scan_shape),
            "channel_names": scan_channels,
        }
    (sample_dir / SCAN_META_NAME).write_text(json.dumps(scan_meta), encoding="utf-8")


def create_synthetic_paired_sample(
    sample_dir: Path,
    feat_shape: tuple = (7, 16, 16),
    lbl_shape: tuple = (8, 16, 16),
    feat_channels: Optional[List[str]] = None,
    lbl_channels: Optional[List[str]] = None,
    feat_meta: Optional[Dict[str, Any]] = None,
    lbl_meta: Optional[Dict[str, Any]] = None,
    corrupt_feat_val: Optional[float] = None,
    corrupt_lbl_val: Optional[float] = None,
) -> None:
    """Helper to create synthetic sample artifacts in sample_dir."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    if feat_channels is None:
        feat_channels = [f"feat_{i}" for i in range(feat_shape[0])]
    if lbl_channels is None:
        lbl_channels = [f"lbl_{i}" for i in range(lbl_shape[0])]

    # 1. Feature NPZ
    feats = np.zeros(feat_shape, dtype=np.float32)
    if corrupt_feat_val is not None:
        feats[0, 0, 0] = corrupt_feat_val
    np.savez_compressed(
        sample_dir / FEATURE_NPZ_NAME,
        features=feats,
        raw_features=feats.copy(),
        channel_names=feat_channels,
    )

    # 2. Feature metadata
    if feat_meta is None:
        feat_meta = {
            "design": "test_design",
            "pdk": "nangate45",
            "source_run_dir": "/runs/test_run_123",
            "grid_size": [feat_shape[1], feat_shape[2]],
            "tensor_shape": list(feat_shape),
            "channel_names": feat_channels,
            "die_area_um": {"width": 100.0, "height": 100.0},
            "design_statistics": {"total_components": 100, "total_nets": 120},
        }
    (sample_dir / FEATURE_META_NAME).write_text(json.dumps(feat_meta), encoding="utf-8")

    # 3. Label NPZ
    lbls = np.zeros(lbl_shape, dtype=np.float32)
    if corrupt_lbl_val is not None:
        lbls[0, 0, 0] = corrupt_lbl_val
    np.savez_compressed(
        sample_dir / LABEL_NPZ_NAME,
        labels=lbls,
        raw_labels=lbls.copy(),
        channel_names=lbl_channels,
    )

    # 4. Label metadata
    if lbl_meta is None:
        lbl_meta = {
            "design": "test_design",
            "pdk": "nangate45",
            "source_run_dir": "/runs/test_run_123",
            "grid_size": [lbl_shape[1], lbl_shape[2]],
            "tensor_shape": list(lbl_shape),
            "channel_names": lbl_channels,
            "die_area_um": {"width": 100.0, "height": 100.0},
            "routing_statistics": {"routed_nets": 100, "total_vias": 50},
        }
    (sample_dir / LABEL_META_NAME).write_text(json.dumps(lbl_meta), encoding="utf-8")


class TestPhysicalDesignSample(unittest.TestCase):
    """Test PhysicalDesignSample abstraction and validation."""

    def test_load_real_uart_sample(self):
        """Test loading existing Stage 2a + 2b UART sample if available."""
        sample_path = Path("data/processed/samples/uart_tx_nangate45_20261011_064013")
        required_files = [
            sample_path / FEATURE_NPZ_NAME,
            sample_path / FEATURE_META_NAME,
            sample_path / LABEL_NPZ_NAME,
            sample_path / LABEL_META_NAME,
        ]

        if not sample_path.is_dir() or not all(p.is_file() for p in required_files):
            self.skipTest(f"UART processed sample directory not complete at {sample_path}")

        sample = PhysicalDesignSample.load(sample_path, validate=True)

        self.assertEqual(sample.sample_id, "uart_tx_nangate45_20261011_064013")
        self.assertEqual(sample.feature_shape, (7, 64, 64))
        self.assertEqual(sample.label_shape, (8, 64, 64))
        self.assertEqual(sample.grid_shape, (64, 64))
        self.assertEqual(sample.num_feature_channels, 7)
        self.assertEqual(sample.num_label_channels, 8)
        self.assertEqual(sample.feature_channel_names, CHANNEL_NAMES)
        self.assertEqual(sample.label_channel_names, ROUTING_CHANNEL_NAMES)

        # Finiteness
        self.assertTrue(np.all(np.isfinite(sample.feature_tensor)))
        self.assertTrue(np.all(np.isfinite(sample.raw_feature_tensor)))
        self.assertTrue(np.all(np.isfinite(sample.label_tensor)))
        self.assertTrue(np.all(np.isfinite(sample.raw_label_tensor)))

        # Summary
        summary = sample.summary()
        self.assertEqual(summary["sample_id"], "uart_tx_nangate45_20261011_064013")
        self.assertEqual(summary["grid_shape"], [64, 64])
        self.assertEqual(summary["feature_tensor_shape"], [7, 64, 64])
        self.assertEqual(summary["label_tensor_shape"], [8, 64, 64])

        formatted = sample.format_summary()
        self.assertIn("uart_tx_nangate45_20261011_064013", formatted)
        self.assertIn("64 x 64", formatted)

    def test_missing_file_error(self):
        """Test error when required sample artifact is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "sample"

            # Non-existent directory
            with self.assertRaises(FileNotFoundError):
                PhysicalDesignSample.load(sample_dir)

            # Test missing one file at a time
            for missing_file in [FEATURE_NPZ_NAME, FEATURE_META_NAME, LABEL_NPZ_NAME, LABEL_META_NAME]:
                with tempfile.TemporaryDirectory() as sub_dir:
                    sub_path = Path(sub_dir)
                    create_synthetic_paired_sample(sub_path)
                    (sub_path / missing_file).unlink()

                    with self.assertRaises(FileNotFoundError) as ctx:
                        PhysicalDesignSample.load(sub_path)
                    self.assertIn(missing_file, str(ctx.exception))

    def test_mismatched_grid_shape_error(self):
        """Test error when spatial dimensions H, W differ between features and labels."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "mismatched_grid"
            create_synthetic_paired_sample(
                sample_dir,
                feat_shape=(7, 64, 64),
                lbl_shape=(8, 32, 32),
                feat_meta={
                    "source_run_dir": "/runs/test",
                    "grid_size": [64, 64],
                    "tensor_shape": [7, 64, 64],
                    "channel_names": [f"f{i}" for i in range(7)],
                },
                lbl_meta={
                    "source_run_dir": "/runs/test",
                    "grid_size": [32, 32],
                    "tensor_shape": [8, 32, 32],
                    "channel_names": [f"l{i}" for i in range(8)],
                },
            )

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir)
            self.assertIn("Grid resolution mismatch", str(ctx.exception))

    def test_channel_count_mismatch_error(self):
        """Test error when channel count does not match channel_names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "mismatched_channels"
            # 7 feature channels in tensor, but only 5 names provided
            create_synthetic_paired_sample(
                sample_dir,
                feat_shape=(7, 16, 16),
                feat_channels=[f"feat_{i}" for i in range(5)],
            )

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir)
            self.assertIn("Feature channel names count", str(ctx.exception))

    def test_finite_value_validation(self):
        """Test error when tensors contain NaN or Inf."""
        # Feature with NaN
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "nan_feature"
            create_synthetic_paired_sample(sample_dir, corrupt_feat_val=float("nan"))
            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir)
            self.assertIn("non-finite values", str(ctx.exception))

        # Label with Inf
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "inf_label"
            create_synthetic_paired_sample(sample_dir, corrupt_lbl_val=float("inf"))
            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir)
            self.assertIn("non-finite values", str(ctx.exception))

    def test_metadata_inconsistency_error(self):
        """Test error when feature metadata and label metadata have conflicting run IDs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "run_mismatch"
            feat_meta = {
                "source_run_dir": "/runs/run_alpha_20261011_000001",
                "grid_size": [16, 16],
                "tensor_shape": [7, 16, 16],
                "channel_names": [f"f{i}" for i in range(7)],
            }
            lbl_meta = {
                "source_run_dir": "/runs/run_beta_20261011_000002",
                "grid_size": [16, 16],
                "tensor_shape": [8, 16, 16],
                "channel_names": [f"l{i}" for i in range(8)],
            }
            create_synthetic_paired_sample(
                sample_dir,
                feat_shape=(7, 16, 16),
                lbl_shape=(8, 16, 16),
                feat_meta=feat_meta,
                lbl_meta=lbl_meta,
            )

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir)
            self.assertIn("Source run directory mismatch", str(ctx.exception))

    def test_metadata_design_mismatch_error(self):
        """Test error when feature metadata and label metadata have conflicting design names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "design_mismatch"
            feat_meta = {
                "design": "uart_tx",
                "source_run_dir": "/runs/common_run",
                "grid_size": [16, 16],
                "tensor_shape": [7, 16, 16],
                "channel_names": [f"f{i}" for i in range(7)],
            }
            lbl_meta = {
                "design": "spi_master",
                "source_run_dir": "/runs/common_run",
                "grid_size": [16, 16],
                "tensor_shape": [8, 16, 16],
                "channel_names": [f"l{i}" for i in range(8)],
            }
            create_synthetic_paired_sample(
                sample_dir,
                feat_shape=(7, 16, 16),
                lbl_shape=(8, 16, 16),
                feat_meta=feat_meta,
                lbl_meta=lbl_meta,
            )

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir)
            self.assertIn("Design name mismatch", str(ctx.exception))

    def test_optional_metadata_missing_fields_allowed_and_warns(self):
        """Test that missing optional metadata fields pass validation and emit warnings instead of errors."""
        import warnings

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "missing_optional_meta"
            # Feature metadata has run and design; label metadata omits both
            feat_meta = {
                "design": "uart_tx",
                "source_run_dir": "/runs/test_run",
                "grid_size": [16, 16],
                "tensor_shape": [7, 16, 16],
                "channel_names": [f"f{i}" for i in range(7)],
            }
            lbl_meta = {
                # Missing 'source_run_dir' and 'design'
                "grid_size": [16, 16],
                "tensor_shape": [8, 16, 16],
                "channel_names": [f"l{i}" for i in range(8)],
            }
            create_synthetic_paired_sample(
                sample_dir,
                feat_shape=(7, 16, 16),
                lbl_shape=(8, 16, 16),
                feat_meta=feat_meta,
                lbl_meta=lbl_meta,
            )

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                sample = PhysicalDesignSample.load(sample_dir, validate=True)

            # Valid sample returned without raising ValueError
            self.assertEqual(sample.grid_shape, (16, 16))
            self.assertEqual(sample.design, "uart_tx")
            self.assertEqual(sample.source_run_dir, "/runs/test_run")

            # Check that UserWarnings were recorded for missing optional fields
            warning_messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
            self.assertTrue(any("source_run_dir" in msg for msg in warning_messages))
            self.assertTrue(any("design" in msg for msg in warning_messages))

    def test_summary_output_fields(self):
        """Test that summary dictionary contains all required fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "summary_test"
            create_synthetic_paired_sample(sample_dir)

            sample = PhysicalDesignSample.load(sample_dir)
            summary = sample.summary()

            expected_keys = [
                "sample_id",
                "sample_dir",
                "design",
                "pdk",
                "source_run_dir",
                "grid_shape",
                "feature_tensor_shape",
                "label_tensor_shape",
                "feature_channel_count",
                "label_channel_count",
                "feature_channels",
                "label_channels",
                "die_area_um",
                "design_statistics",
                "routing_statistics",
            ]
            for key in expected_keys:
                self.assertIn(key, summary, f"Missing expected key '{key}' in summary()")

            self.assertEqual(summary["grid_shape"], [16, 16])
            self.assertEqual(summary["feature_tensor_shape"], [7, 16, 16])
            self.assertEqual(summary["label_tensor_shape"], [8, 16, 16])

    def test_load_real_uart_sample_with_scan(self):
        """Test loading existing UART sample which has scan features from Stage 3e."""
        sample_path = Path("data/processed/samples/uart_tx_nangate45_20261011_064013")
        scan_file = sample_path / SCAN_NPZ_NAME
        scan_meta = sample_path / SCAN_META_NAME
        if not sample_path.is_dir() or not scan_file.is_file() or not scan_meta.is_file():
            self.skipTest("UART sample scan artifacts not present")

        sample = PhysicalDesignSample.load(sample_path, validate=True, load_scan=True)
        self.assertTrue(sample.has_scan_features)
        self.assertEqual(sample.scan_feature_shape, (5, 64, 64))
        self.assertEqual(sample.num_scan_channels, 5)
        self.assertEqual(sample.scan_source, "synthetic_proxy")

        # Fused tensor shape & channels
        fused = sample.get_fused_input_tensor(include_scan=True)
        self.assertEqual(fused.shape, (12, 64, 64))
        self.assertTrue(np.all(np.isfinite(fused)))

        # Physical-only fused tensor
        phys_only = sample.get_fused_input_tensor(include_scan=False)
        self.assertEqual(phys_only.shape, (7, 64, 64))

        # Prefixed channel names
        fused_names = sample.get_fused_channel_names(include_scan=True)
        self.assertEqual(len(fused_names), 12)
        self.assertTrue(all(n.startswith("physical/") for n in fused_names[:7]))
        self.assertTrue(all(n.startswith("scan/") for n in fused_names[7:]))

        phys_names = sample.get_fused_channel_names(include_scan=False)
        self.assertEqual(len(phys_names), 7)
        self.assertTrue(all(n.startswith("physical/") for n in phys_names))

        # Summary
        summary = sample.summary()
        self.assertTrue(summary["has_scan_features"])
        self.assertEqual(summary["scan_feature_tensor_shape"], [5, 64, 64])
        self.assertEqual(summary["scan_source"], "synthetic_proxy")
        self.assertEqual(summary["fused_tensor_shape"], [12, 64, 64])
        self.assertEqual(summary["fused_channel_count"], 12)

        formatted = sample.format_summary()
        self.assertIn("Scan Tensor", formatted)
        self.assertIn("synthetic_proxy", formatted)
        self.assertIn("Fused In Tensor", formatted)

    def test_sample_without_scan_features(self):
        """Test sample without scan features remains valid and operates as physical baseline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "no_scan"
            create_synthetic_paired_sample(sample_dir)

            sample = PhysicalDesignSample.load(sample_dir, validate=True, load_scan=True)
            self.assertFalse(sample.has_scan_features)
            self.assertIsNone(sample.scan_feature_shape)
            self.assertEqual(sample.num_scan_channels, 0)
            self.assertIsNone(sample.scan_source)

            # Fused tensor returns physical only
            fused = sample.get_fused_input_tensor(include_scan=True)
            self.assertEqual(fused.shape, (7, 16, 16))

            # Channels are prefixed with physical/
            fused_names = sample.get_fused_channel_names(include_scan=True)
            self.assertEqual(len(fused_names), 7)
            self.assertTrue(all(n.startswith("physical/") for n in fused_names))

            summary = sample.summary()
            self.assertFalse(summary["has_scan_features"])
            self.assertEqual(summary["fused_tensor_shape"], [7, 16, 16])
            self.assertEqual(summary["fused_channel_count"], 7)

            formatted = sample.format_summary()
            self.assertIn("Physical-only baseline", formatted)

    def test_sample_with_synthetic_scan_features(self):
        """Test sample with synthetic scan features passes validation and fuses correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "with_scan"
            create_synthetic_paired_sample(sample_dir, feat_shape=(7, 16, 16))
            add_synthetic_scan_features(sample_dir, scan_shape=(5, 16, 16), scan_source="openroad_dft")

            sample = PhysicalDesignSample.load(sample_dir, validate=True, load_scan=True)
            self.assertTrue(sample.has_scan_features)
            self.assertEqual(sample.scan_feature_shape, (5, 16, 16))
            self.assertEqual(sample.num_scan_channels, 5)
            self.assertEqual(sample.scan_source, "openroad_dft")

            fused = sample.get_fused_input_tensor(include_scan=True)
            self.assertEqual(fused.shape, (12, 16, 16))

            fused_names = sample.get_fused_channel_names(include_scan=True)
            self.assertEqual(len(fused_names), 12)
            self.assertTrue(all(n.startswith("physical/") for n in fused_names[:7]))
            self.assertTrue(all(n.startswith("scan/") for n in fused_names[7:]))

    def test_scan_grid_mismatch_error(self):
        """Test error when scan grid resolution differs from physical grid."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "scan_grid_mismatch"
            create_synthetic_paired_sample(sample_dir, feat_shape=(7, 16, 16))
            add_synthetic_scan_features(sample_dir, scan_shape=(5, 32, 32))

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir, validate=True)
            self.assertIn("Grid resolution mismatch", str(ctx.exception))

    def test_scan_channel_count_mismatch_error(self):
        """Test error when scan channel count does not match channel names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "scan_channel_mismatch"
            create_synthetic_paired_sample(sample_dir, feat_shape=(7, 16, 16))
            add_synthetic_scan_features(
                sample_dir,
                scan_shape=(5, 16, 16),
                scan_channels=["scan_0", "scan_1"],  # 2 names for 5 channels
            )

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir, validate=True)
            self.assertIn("Scan channel names count", str(ctx.exception))

    def test_scan_non_finite_values_error(self):
        """Test error when scan features contain NaN or Inf."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "scan_nan"
            create_synthetic_paired_sample(sample_dir, feat_shape=(7, 16, 16))
            add_synthetic_scan_features(sample_dir, scan_shape=(5, 16, 16), corrupt_scan_val=float("nan"))

            with self.assertRaises(ValueError) as ctx:
                PhysicalDesignSample.load(sample_dir, validate=True)
            self.assertIn("non-finite values", str(ctx.exception))

    def test_scan_missing_partial_files_error(self):
        """Test error when only one of scan NPZ or scan metadata JSON exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "scan_partial"
            create_synthetic_paired_sample(sample_dir)
            add_synthetic_scan_features(sample_dir)

            # Remove metadata file
            (sample_dir / SCAN_META_NAME).unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                PhysicalDesignSample.load(sample_dir, validate=True)
            self.assertIn("missing scan metadata", str(ctx.exception))

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "scan_partial_npz"
            create_synthetic_paired_sample(sample_dir)
            add_synthetic_scan_features(sample_dir)

            # Remove npz file
            (sample_dir / SCAN_NPZ_NAME).unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                PhysicalDesignSample.load(sample_dir, validate=True)
            self.assertIn("missing scan feature tensor", str(ctx.exception))

    def test_validate_design_sample_cli_require_scan(self):
        """Test validate_design_sample.py CLI with --require-scan."""
        import subprocess
        import sys

        script = Path("scripts/validate_design_sample.py").resolve()

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_no_scan = Path(tmp_dir) / "no_scan"
            create_synthetic_paired_sample(sample_no_scan)

            # Without --require-scan: should pass
            res = subprocess.run(
                [sys.executable, str(script), "--sample-dir", str(sample_no_scan)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Expected 0, got {res.returncode}: {res.stderr}")
            self.assertIn("[RESULT] PASS", res.stdout)

            # With --require-scan: should fail
            res_fail = subprocess.run(
                [sys.executable, str(script), "--sample-dir", str(sample_no_scan), "--require-scan"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(res_fail.returncode, 0)
            self.assertIn("scan features are missing", res_fail.stderr)

            # Add scan features: should now pass with --require-scan
            add_synthetic_scan_features(sample_no_scan)
            res_pass = subprocess.run(
                [sys.executable, str(script), "--sample-dir", str(sample_no_scan), "--require-scan"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_pass.returncode, 0, f"Expected 0, got {res_pass.returncode}: {res_pass.stderr}")
            self.assertIn("[RESULT] PASS", res_pass.stdout)
            self.assertIn("Fused Input Shape", res_pass.stdout)

    def test_export_fused_sample_script(self):
        """Test scripts/export_fused_sample.py exporting fused tensors."""
        import subprocess
        import sys

        script = Path("scripts/export_fused_sample.py").resolve()

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "sample"
            create_synthetic_paired_sample(sample_dir, feat_shape=(7, 16, 16))
            add_synthetic_scan_features(sample_dir, scan_shape=(5, 16, 16))

            out_npz = sample_dir / "fused_input.npz"
            res = subprocess.run(
                [sys.executable, str(script), "--sample-dir", str(sample_dir), "--output", str(out_npz)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Export failed: {res.stderr}")
            self.assertTrue(out_npz.is_file())
            out_meta = sample_dir / "fused_input_metadata.json"
            self.assertTrue(out_meta.is_file())

            # Verify contents
            with np.load(out_npz, allow_pickle=False) as npz:
                self.assertIn("fused_features", npz)
                self.assertIn("raw_fused_features", npz)
                self.assertIn("channel_names", npz)
                self.assertEqual(npz["fused_features"].shape, (12, 16, 16))
                self.assertEqual(npz["raw_fused_features"].shape, (12, 16, 16))
                self.assertEqual(len(npz["channel_names"]), 12)

            meta_data = json.loads(out_meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["tensor_shape"], [12, 16, 16])
            self.assertTrue(meta_data["has_scan_features"])
            self.assertEqual(meta_data["scan_source"], "synthetic_proxy")


if __name__ == "__main__":
    unittest.main()
