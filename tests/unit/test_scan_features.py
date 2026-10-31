"""
Unit and integration tests for scan-aware spatial feature tensor extraction (Stage 3e).
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from dft_hotspot.dft.scan_metadata import (
    ScanCell,
    ScanChain,
    ScanEndpoint,
    ScanMetadata,
    ScanPorts,
    ScanStatistics,
    SyntheticConfig,
)
from dft_hotspot.features.def_parser import DefDieArea
from dft_hotspot.features.scan_features import (
    SCAN_CHANNEL_NAMES,
    ScanFeatureExtractor,
)


def create_synthetic_scan_metadata(
    chain_length: int = 4,
    source: str = "synthetic_proxy",
    dbu_coords: bool = True,
) -> ScanMetadata:
    """Helper to create small in-memory ScanMetadata for unit testing."""
    # Place cells along a diagonal / horizontal path:
    # c0: (10000, 10000), c1: (20000, 10000), c2: (20000, 30000), c3: (40000, 30000)
    coords = [
        (10000, 10000),
        (20000, 10000),
        (20000, 30000),
        (40000, 30000),
    ][:chain_length]

    cells = [
        ScanCell(
            index=i,
            instance=f"ff_{i}",
            master="DFFR_X1",
            x=x,
            y=y,
            x_um=x / 1000.0,
            y_um=y / 1000.0,
            placed=True,
        )
        for i, (x, y) in enumerate(coords)
    ]

    chain = ScanChain(
        chain_id="synthetic_chain_0",
        start=ScanEndpoint(type="SYNTHETIC", name="synthetic_scan_in_0"),
        stop=ScanEndpoint(type="SYNTHETIC", name="synthetic_scan_out_0"),
        partition="synthetic_default",
        cell_count=len(cells),
        cells=cells,
    )

    stats = ScanStatistics(
        chain_count=1,
        total_scan_cells=len(cells),
        min_chain_length=len(cells),
        max_chain_length=len(cells),
        avg_chain_length=float(len(cells)),
        placed_scan_cells=len(cells),
    )

    return ScanMetadata(
        schema_version="scan_metadata_v001",
        source=source,
        design="toy_test",
        platform="nangate45",
        synthetic_config=SyntheticConfig(chain_count=1) if source == "synthetic_proxy" else None,
        chains=[chain],
        statistics=stats,
        limitations=["Synthetic proxy scan metadata only; no scan replacement or ATPG was performed."],
    )


class TestScanFeatures(unittest.TestCase):
    """Test suite for ScanFeatureExtractor."""

    def setUp(self):
        # 100um x 100um die area, 1000 DBU per um
        self.die_area = DefDieArea(
            x_min=0.0,
            y_min=0.0,
            x_max=100000.0,
            y_max=100000.0,
            dbu_per_micron=1000,
        )
        self.extractor = ScanFeatureExtractor(grid_size=64, die_area=self.die_area)

    def test_tensor_construction_shape_and_channels(self):
        """Verifies tensor shape (5, 64, 64), channel count, and channel names."""
        meta = create_synthetic_scan_metadata(chain_length=4)
        norm, raw, meta_dict = self.extractor.extract_from_scan_metadata(meta)

        self.assertEqual(norm.shape, (5, 64, 64))
        self.assertEqual(raw.shape, (5, 64, 64))
        self.assertEqual(norm.dtype, np.float32)
        self.assertEqual(raw.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(norm)))
        self.assertTrue(np.all(np.isfinite(raw)))

        self.assertEqual(meta_dict["schema_version"], "scan_features_v001")
        self.assertEqual(meta_dict["channel_names"], SCAN_CHANNEL_NAMES)
        self.assertEqual(meta_dict["tensor_shape"], [5, 64, 64])

    def test_scan_cell_density_binning(self):
        """Verifies scan cell density correctly accumulates cell counts."""
        meta = create_synthetic_scan_metadata(chain_length=4)
        norm, raw, meta_dict = self.extractor.extract_from_scan_metadata(meta)

        # raw_cell_density is channel 0
        raw_cell_density = raw[0]
        # Total sum of cell density must equal total scan cells (4)
        self.assertAlmostEqual(float(np.sum(raw_cell_density)), 4.0, places=4)
        # 4 distinct coordinates -> exactly 4 active bins with count 1.0 each
        self.assertEqual(int(np.count_nonzero(raw_cell_density > 0)), 4)
        self.assertAlmostEqual(float(np.max(raw_cell_density)), 1.0, places=4)

    def test_endpoint_density(self):
        """Verifies scan_chain_endpoint_density marks exactly start and stop cells."""
        meta = create_synthetic_scan_metadata(chain_length=4)
        norm, raw, _ = self.extractor.extract_from_scan_metadata(meta)

        # raw endpoint density is channel 1
        raw_endpoints = raw[1]
        # Single chain of length 4 -> exactly 2 endpoints (start at ff_0, stop at ff_3)
        self.assertAlmostEqual(float(np.sum(raw_endpoints)), 2.0, places=4)
        self.assertEqual(int(np.count_nonzero(raw_endpoints > 0)), 2)

    def test_segment_rasterization_length_conservation(self):
        """Verifies Manhattan chain segments conserve exact total wirelength in microns."""
        meta = create_synthetic_scan_metadata(chain_length=4)
        norm, raw, meta_dict = self.extractor.extract_from_scan_metadata(meta)

        # Segments:
        # c0 (10, 10) -> c1 (20, 10): dx = 10um, dy = 0um -> 10um
        # c1 (20, 10) -> c2 (20, 30): dx = 0um, dy = 20um -> 20um
        # c2 (20, 30) -> c3 (40, 30): dx = 20um, dy = 0um -> 20um
        # Total Manhattan length = 10 + 20 + 20 = 50 um
        expected_total_wl = 50.0

        raw_segments = raw[2]
        self.assertAlmostEqual(float(np.sum(raw_segments)), expected_total_wl, places=3)
        self.assertEqual(meta_dict["segment_count"], 3)
        self.assertAlmostEqual(meta_dict["total_scan_wirelength_um"], expected_total_wl, places=3)

    def test_order_index_normalization(self):
        """Verifies scan_chain_order_index values are strictly bounded in [0.0, 1.0]."""
        meta = create_synthetic_scan_metadata(chain_length=4)
        norm, raw, _ = self.extractor.extract_from_scan_metadata(meta)

        # Order index is channel 3
        order_idx = raw[3]
        active_vals = order_idx[order_idx > 0]

        # For 4 cells: indices are 0/3=0.0, 1/3=0.333, 2/3=0.667, 3/3=1.0
        # Active positive values: 1/3, 2/3, 1.0 (note index 0 is 0.0)
        self.assertTrue(np.all(active_vals >= 0.0))
        self.assertTrue(np.all(active_vals <= 1.0))
        self.assertAlmostEqual(float(np.max(order_idx)), 1.0, places=4)

    def test_local_wirelength_proxy(self):
        """Verifies scan_chain_local_wirelength_proxy sums incident segment lengths."""
        meta = create_synthetic_scan_metadata(chain_length=4)
        norm, raw, _ = self.extractor.extract_from_scan_metadata(meta)

        # Segments lengths: s01=10um, s12=20um, s23=20um. Total WL = 50um.
        # c0 incident: s01 = 10um
        # c1 incident: s01 + s12 = 10 + 20 = 30um
        # c2 incident: s12 + s23 = 20 + 20 = 40um
        # c3 incident: s23 = 20um
        # Sum of local WL across all cells = 10 + 30 + 40 + 20 = 100um = 2 * total_wl
        raw_local_wl = raw[4]
        self.assertAlmostEqual(float(np.sum(raw_local_wl)), 100.0, places=3)
        self.assertAlmostEqual(float(np.max(raw_local_wl)), 40.0, places=3)

    def test_synthetic_proxy_limitations_propagation(self):
        """Verifies limitation disclosures are propagated for synthetic_proxy source."""
        meta = create_synthetic_scan_metadata(source="synthetic_proxy")
        _, _, meta_dict = self.extractor.extract_from_scan_metadata(meta)

        self.assertEqual(meta_dict["scan_source"], "synthetic_proxy")
        lims = meta_dict["limitations"]
        self.assertTrue(any("Synthetic proxy" in l for l in lims))
        self.assertTrue(any("Scan segment density is a proxy" in l for l in lims))

    def test_real_uart_nangate45_sample_integration(self):
        """Integration test extracting scan features from real UART Nangate45 sample directory."""
        sample_dir = Path("data/processed/samples/uart_tx_nangate45_20261011_064013")
        scan_json = sample_dir / "scan_metadata.json"
        if not scan_json.is_file():
            self.skipTest(f"Artifact {scan_json} not present.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            # Copy scan_metadata.json and metadata.json into temp dir for non-destructive test
            (tmp_p / "scan_metadata.json").write_text(scan_json.read_text(encoding="utf-8"), encoding="utf-8")
            meta_json = sample_dir / "metadata.json"
            if meta_json.is_file():
                (tmp_p / "metadata.json").write_text(meta_json.read_text(encoding="utf-8"), encoding="utf-8")

            extractor = ScanFeatureExtractor(grid_size=64)
            results = extractor.process_and_save_sample(
                sample_dir=tmp_p,
                output_dir=tmp_p,
            )

            self.assertTrue(results["npz"].is_file())
            self.assertTrue(results["metadata"].is_file())
            self.assertTrue(results["preview"].is_file())

            # Check NPZ arrays
            npz_data = np.load(results["npz"])
            self.assertIn("scan_features", npz_data)
            self.assertIn("raw_scan_features", npz_data)
            scan_feat = npz_data["scan_features"]
            raw_feat = npz_data["raw_scan_features"]

            self.assertEqual(scan_feat.shape, (5, 64, 64))
            self.assertEqual(raw_feat.shape, (5, 64, 64))
            self.assertTrue(np.all(np.isfinite(scan_feat)))
            self.assertTrue(np.all(np.isfinite(raw_feat)))

            # Cell density sum must be 33
            self.assertAlmostEqual(float(np.sum(raw_feat[0])), 33.0, places=4)
            # Segment density sum must be positive
            self.assertGreater(float(np.sum(raw_feat[2])), 0.0)

            # Check JSON metadata
            with open(results["metadata"], "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            self.assertEqual(saved_meta["schema_version"], "scan_features_v001")
            self.assertEqual(saved_meta["scan_source"], "synthetic_proxy")
            self.assertEqual(saved_meta["total_scan_cells"], 33)
            self.assertEqual(saved_meta["segment_count"], 32)
            self.assertTrue(any("Synthetic proxy" in l for l in saved_meta["limitations"]))


if __name__ == "__main__":
    unittest.main()
