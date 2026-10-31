"""
Unit and integration tests for Stage 2a physical feature extraction.
Verifies coordinate mapping, DEF parsing, cell classification, tensor shape,
normalization, and end-to-end extraction on the UART physical design run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from dft_hotspot.features.cell_classifier import (
    CellClassifier,
    classify_cell,
    is_combinational,
    is_sequential,
)
from dft_hotspot.features.def_parser import DefData, DefDieArea, DefParser
from dft_hotspot.features.extractor import (
    CHANNEL_NAMES,
    PhysicalFeatureExtractor,
    compute_net_bounding_box_density,
)
from dft_hotspot.features.grid_mapper import GridMapper


SYNTHETIC_DEF = """VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN test_toy ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 100000 100000 ) ;
COMPONENTS 4 ;
    - inst_dff DFFR_X1 + PLACED ( 25000 25000 ) N ;
    - inst_sdff SDFF_X1 + PLACED ( 75000 25000 ) FS ;
    - inst_inv INV_X1 + PLACED ( 25000 75000 ) N ;
    - inst_nand NAND2_X1 + PLACED ( 75000 75000 ) S ;
END COMPONENTS
PINS 2 ;
    - clk + NET clk + DIRECTION INPUT + PLACED ( 0 50000 ) N ;
    - out + NET out + DIRECTION OUTPUT + PLACED ( 100000 50000 ) S ;
END PINS
NETS 3 ;
    - clk ( PIN clk ) ( inst_dff CK ) ( inst_sdff CK ) + USE SIGNAL ;
    - n1 ( inst_dff Q ) ( inst_inv A ) + USE SIGNAL ;
    - out ( PIN out ) ( inst_nand ZN ) + USE SIGNAL ;
END NETS
END DESIGN
"""


class TestCellClassifier(unittest.TestCase):
    """Test heuristic classification of sequential vs combinational cells."""

    def test_sequential_cells(self):
        seq_masters = [
            "DFF_X1",
            "DFFR_X1",
            "DFFS_X1",
            "DFFRS_X2",
            "SDFF_X1",
            "SDFFR_X1",
            "EDFF_X1",
            "LATCH_X1",
            "DLH_X1",
            "DLL_X1",
        ]
        for master in seq_masters:
            with self.subTest(master=master):
                self.assertEqual(classify_cell(master), "sequential")
                self.assertTrue(is_sequential(master))
                self.assertFalse(is_combinational(master))

    def test_combinational_cells(self):
        comb_masters = [
            "INV_X1",
            "BUF_X1",
            "CLKBUF_X1",
            "AND2_X1",
            "NAND2_X1",
            "NAND3_X1",
            "NOR2_X1",
            "OR2_X1",
            "XOR2_X1",
            "XNOR2_X1",
            "AOI21_X1",
            "OAI21_X1",
            "MUX2_X1",
            "HA_X1",
            "FA_X1",
        ]
        for master in comb_masters:
            with self.subTest(master=master):
                self.assertEqual(classify_cell(master), "combinational")
                self.assertFalse(is_sequential(master))
                self.assertTrue(is_combinational(master))


class TestGridMapper(unittest.TestCase):
    """Test physical layout continuous coordinate mapping to discrete bins."""

    def setUp(self):
        self.die = DefDieArea(x_min=0, y_min=0, x_max=1000, y_max=1000, dbu_per_micron=1000)
        self.mapper = GridMapper(self.die, grid_size=4)

    def test_coordinate_mapping(self):
        # Center of first bin (0, 0)
        r, c = self.mapper.map_coordinate(100, 100)
        self.assertEqual((r, c), (0, 0))

        # Center of top-right bin (3, 3)
        r, c = self.mapper.map_coordinate(900, 900)
        self.assertEqual((r, c), (3, 3))

        # Bottom-left boundary
        r, c = self.mapper.map_coordinate(0, 0)
        self.assertEqual((r, c), (0, 0))

        # Top-right boundary clamped
        r, c = self.mapper.map_coordinate(1000, 1000)
        self.assertEqual((r, c), (3, 3))

    def test_out_of_bounds_clipping(self):
        # Negative coords
        r, c = self.mapper.map_coordinate(-100, -50)
        self.assertEqual((r, c), (0, 0))

        # Far positive coords
        r, c = self.mapper.map_coordinate(2000, 5000)
        self.assertEqual((r, c), (3, 3))

    def test_normalized_coordinates(self):
        norm_x, norm_y = self.mapper.get_normalized_coordinates()
        self.assertEqual(norm_x.shape, (4, 4))
        self.assertEqual(norm_y.shape, (4, 4))
        # First column center should be 0.125
        self.assertAlmostEqual(norm_x[0, 0], 0.125)
        # Last column center should be 0.875
        self.assertAlmostEqual(norm_x[0, 3], 0.875)
        # All in range [0, 1]
        self.assertTrue(np.all(norm_x >= 0.0) and np.all(norm_x <= 1.0))
        self.assertTrue(np.all(norm_y >= 0.0) and np.all(norm_y <= 1.0))


class TestDefParser(unittest.TestCase):
    """Test parsing synthetic DEF content."""

    def test_parse_synthetic_def(self):
        data = DefParser.parse_string(SYNTHETIC_DEF)

        self.assertIsNotNone(data.die_area)
        self.assertEqual(data.die_area.x_min, 0)
        self.assertEqual(data.die_area.y_min, 0)
        self.assertEqual(data.die_area.x_max, 100000)
        self.assertEqual(data.die_area.y_max, 100000)
        self.assertEqual(data.dbu_per_micron, 1000)
        self.assertEqual(data.die_area.width_um, 100.0)
        self.assertEqual(data.die_area.height_um, 100.0)

        # Components
        self.assertEqual(data.total_components, 4)
        comp_names = {c.name: c for c in data.components}
        self.assertIn("inst_dff", comp_names)
        self.assertEqual(comp_names["inst_dff"].master, "DFFR_X1")
        self.assertEqual(comp_names["inst_dff"].x, 25000)
        self.assertEqual(comp_names["inst_dff"].y, 25000)

        # Pins
        self.assertEqual(data.total_pins, 2)
        pin_names = {p.name: p for p in data.pins}
        self.assertIn("clk", pin_names)
        self.assertEqual(pin_names["clk"].direction, "INPUT")

        # Nets and connectivity
        self.assertEqual(data.total_nets, 3)
        pin_counts = data.get_instance_pin_counts()
        self.assertEqual(pin_counts["inst_dff"], 2)  # Connected to clk and n1
        self.assertEqual(pin_counts["inst_inv"], 1)  # Connected to n1


class TestPhysicalFeatureExtractor(unittest.TestCase):
    """Test feature extraction tensor creation, shapes, normalization, and artifacts."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.def_file = self.tmp_path / "toy.def"
        self.def_file.write_text(SYNTHETIC_DEF, encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_tensor_shape_and_values(self):
        extractor = PhysicalFeatureExtractor(grid_size=8)
        norm_tensor, raw_tensor, metadata = extractor.extract_from_def_file(self.def_file)

        # Shape C x H x W = 7 x 8 x 8
        self.assertEqual(norm_tensor.shape, (7, 8, 8))
        self.assertEqual(raw_tensor.shape, (7, 8, 8))
        self.assertEqual(len(metadata["channel_names"]), 7)

        # All values finite
        self.assertTrue(np.all(np.isfinite(norm_tensor)))
        self.assertTrue(np.all(np.isfinite(raw_tensor)))

        # Normalized values in [0, 1]
        self.assertTrue(np.all(norm_tensor >= 0.0) and np.all(norm_tensor <= 1.0))

        # Check cell density channel (idx 0) has 4 components
        self.assertEqual(raw_tensor[0].sum(), 4.0)

        # Sequential density has 2 components (inst_dff, inst_sdff)
        self.assertEqual(raw_tensor[1].sum(), 2.0)

        # Combinational density has 2 components (inst_inv, inst_nand)
        self.assertEqual(raw_tensor[2].sum(), 2.0)

        # Pin density sum
        self.assertGreater(raw_tensor[3].sum(), 0.0)

    def test_process_and_save_sample(self):
        extractor = PhysicalFeatureExtractor(grid_size=16)
        out_dir = self.tmp_path / "sample"
        artifacts = extractor.process_and_save_sample(
            def_path=self.def_file,
            output_dir=out_dir,
            design_name="test_toy",
        )

        self.assertTrue(artifacts["npz"].is_file())
        self.assertTrue(artifacts["metadata"].is_file())
        self.assertTrue(artifacts["preview"].is_file())

        # Load NPZ
        data = np.load(artifacts["npz"])
        self.assertIn("features", data)
        self.assertIn("raw_features", data)
        self.assertEqual(data["features"].shape, (7, 16, 16))

        # Load metadata
        meta = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(meta["tensor_shape"], [7, 16, 16])
        self.assertEqual(meta["design_statistics"]["total_components"], 4)


class TestRealUartRunExtraction(unittest.TestCase):
    """Integration test against actual UART run artifact if present on disk."""

    def test_uart_extraction(self):
        run_dir = Path("runs/uart_tx_nangate45_20261011_064013")
        def_path = run_dir / "scripts" / "uart_tx_final.def"
        if not def_path.is_file():
            def_path = run_dir / "uart_tx_final.def"

        if not def_path.is_file():
            self.skipTest(f"UART run artifact not found at {def_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "uart_sample"
            extractor = PhysicalFeatureExtractor(grid_size=64)
            artifacts = extractor.process_and_save_sample(
                def_path=def_path,
                output_dir=sample_dir,
                source_run_dir=run_dir,
                design_name="uart_tx",
                pdk_name="nangate45",
            )

            self.assertTrue(artifacts["npz"].is_file())
            self.assertTrue(artifacts["metadata"].is_file())
            self.assertTrue(artifacts["preview"].is_file())

            # Load and validate tensor
            npz = np.load(artifacts["npz"])
            tensor = npz["features"]
            self.assertEqual(tensor.shape, (7, 64, 64))
            self.assertTrue(np.all(np.isfinite(tensor)))

            # Validate cell density is nonzero
            cell_density = tensor[0]
            self.assertGreater(float(cell_density.sum()), 0.0)
            self.assertGreater(int(np.count_nonzero(cell_density)), 0)

            # Validate metadata
            meta = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(meta["tensor_shape"], [7, 64, 64])
            self.assertEqual(meta["design_statistics"]["total_components"], 154)
            self.assertEqual(meta["design_statistics"]["sequential_components"], 33)
            self.assertEqual(meta["design_statistics"]["combinational_components"], 121)
            self.assertEqual(len(meta["channel_names"]), 7)


class TestNetBoundingBoxDensity(unittest.TestCase):
    """Test net bounding-box density extraction on synthetic and real layouts."""

    def test_synthetic_net_bounding_box(self):
        def_data = DefParser.parse_string(SYNTHETIC_DEF)
        self.assertIsNotNone(def_data.die_area)
        mapper = GridMapper(def_data.die_area, grid_size=4)

        # Unweighted net bounding-box density
        bbox_density = compute_net_bounding_box_density(def_data, mapper, weighted=False)
        self.assertEqual(bbox_density.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(bbox_density)))
        self.assertTrue(np.all(bbox_density >= 0.0))
        # Total sum of bounding box overlaps across the 3 nets
        self.assertGreater(float(bbox_density.sum()), 0.0)

        # Weighted net bounding-box density (normalized by covered cell count)
        weighted_density = compute_net_bounding_box_density(def_data, mapper, weighted=True)
        self.assertEqual(weighted_density.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(weighted_density)))
        self.assertTrue(np.all(weighted_density >= 0.0))
        # Each net with >=2 pins contributes exactly 1.0 total weight
        # 3 nets in synthetic DEF all have >=2 pins, so sum should equal 3.0
        self.assertAlmostEqual(float(weighted_density.sum()), 3.0, places=4)

    def test_extractor_method_real_uart(self):
        run_dir = Path("runs/uart_tx_nangate45_20261011_064013")
        def_path = run_dir / "scripts" / "uart_tx_final.def"
        if not def_path.is_file():
            def_path = run_dir / "uart_tx_final.def"

        if not def_path.is_file():
            self.skipTest(f"UART run artifact not found at {def_path}")

        def_data = DefParser.parse_file(def_path)
        extractor = PhysicalFeatureExtractor(grid_size=64)
        bbox_density = extractor.extract_net_bounding_box_density(def_data, weighted=False)

        self.assertEqual(bbox_density.shape, (64, 64))
        self.assertTrue(np.all(np.isfinite(bbox_density)))
        self.assertGreater(float(bbox_density.sum()), 0.0)
        self.assertGreater(int(np.count_nonzero(bbox_density)), 0)


if __name__ == "__main__":
    unittest.main()
