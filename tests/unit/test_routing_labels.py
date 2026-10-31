"""
Unit and integration tests for Stage 2b post-route routing label extraction.
Verifies:
1. Synthetic DEF horizontal route parsing
2. Synthetic DEF vertical route parsing
3. Synthetic DEF via parsing
4. Coordinate reuse syntax
5. Layer-specific channel assignment
6. Segment rasterization length conservation
7. Label tensor shape, dtype, and finite values
8. Integration test against UART final DEF (skipping gracefully if artifact missing, without hardcoded counts)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from dft_hotspot.features.def_parser import DefDieArea
from dft_hotspot.labels.def_route_parser import (
    DefRouteData,
    DefRouteParser,
    DefRouteSegment,
    DefVia,
)
from dft_hotspot.labels.rasterizer import SegmentRasterizer
from dft_hotspot.labels.routing_extractor import (
    ROUTING_CHANNEL_NAMES,
    RoutingLabelExtractor,
)


SYNTHETIC_HORIZONTAL_DEF = """VERSION 5.8 ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
NETS 1 ;
  - net_horiz ( inst1 A ) ( inst2 B )
    + ROUTED metal2 ( 2000 3000 ) ( 6000 * ) ;
END NETS
END DESIGN
"""

SYNTHETIC_VERTICAL_DEF = """VERSION 5.8 ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
NETS 1 ;
  - net_vert ( inst1 A ) ( inst2 B )
    + ROUTED metal3 ( 4000 1000 ) ( * 5000 ) ;
END NETS
END DESIGN
"""

SYNTHETIC_VIA_DEF = """VERSION 5.8 ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
NETS 1 ;
  - net_via ( inst1 A ) ( inst2 B )
    + ROUTED metal1 ( 2000 2000 ) via1_4
    NEW metal2 ( 2000 2000 ) ( 5000 * )
    NEW metal2 ( 5000 2000 ) via2_5 ;
END NETS
END DESIGN
"""

SYNTHETIC_COORDINATE_REUSE_DEF = """VERSION 5.8 ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
NETS 1 ;
  - net_reuse ( inst1 A ) ( inst2 B )
    + ROUTED metal2 ( 1000 2000 ) ( 4000 * )
    NEW metal3 ( 4000 2000 ) ( * 7000 ) ;
END NETS
END DESIGN
"""

SYNTHETIC_MULTI_LAYER_DEF = """VERSION 5.8 ;
UNITS DISTANCE MICRONS 2000 ;
DIEAREA ( 0 0 ) ( 20000 20000 ) ;
NETS 2 ;
  - net1 ( inst1 A ) ( inst2 B )
    + ROUTED metal1 ( 2000 2000 ) ( 6000 * )
    NEW metal2 ( 6000 2000 ) ( * 8000 )
    NEW metal1 ( 6000 2000 ) via1_4 ;
  - net2 ( inst3 A ) ( inst4 B )
    + ROUTED metal3 ( 4000 4000 ) ( 8000 * )
    NEW metal4 ( 8000 4000 ) ( * 12000 )
    NEW metal3 ( 8000 4000 ) via3_2 ;
END NETS
END DESIGN
"""


class TestDefRouteParser(unittest.TestCase):
    """Test DEF route parsing on synthetic fixtures."""

    def test_synthetic_def_horizontal_route_parsing(self):
        """Test 1: Synthetic DEF horizontal route parsing."""
        data = DefRouteParser.parse_string(SYNTHETIC_HORIZONTAL_DEF)
        self.assertEqual(data.total_nets_count, 1)
        self.assertEqual(data.routed_nets_count, 1)
        self.assertEqual(data.total_segments, 1)

        seg = data.segments[0]
        self.assertEqual(seg.net_name, "net_horiz")
        self.assertEqual(seg.layer, "metal2")
        self.assertEqual(seg.x1, 2000)
        self.assertEqual(seg.y1, 3000)
        self.assertEqual(seg.x2, 6000)
        self.assertEqual(seg.y2, 3000)
        self.assertEqual(seg.length_dbu, 4000.0)
        self.assertEqual(seg.length_um, 4.0)
        self.assertTrue(seg.is_horizontal)
        self.assertFalse(seg.is_vertical)

    def test_synthetic_def_vertical_route_parsing(self):
        """Test 2: Synthetic DEF vertical route parsing."""
        data = DefRouteParser.parse_string(SYNTHETIC_VERTICAL_DEF)
        self.assertEqual(data.total_nets_count, 1)
        self.assertEqual(data.routed_nets_count, 1)
        self.assertEqual(data.total_segments, 1)

        seg = data.segments[0]
        self.assertEqual(seg.net_name, "net_vert")
        self.assertEqual(seg.layer, "metal3")
        self.assertEqual(seg.x1, 4000)
        self.assertEqual(seg.y1, 1000)
        self.assertEqual(seg.x2, 4000)
        self.assertEqual(seg.y2, 5000)
        self.assertEqual(seg.length_dbu, 4000.0)
        self.assertEqual(seg.length_um, 4.0)
        self.assertTrue(seg.is_vertical)
        self.assertFalse(seg.is_horizontal)

    def test_synthetic_def_via_parsing(self):
        """Test 3: Synthetic DEF via parsing."""
        data = DefRouteParser.parse_string(SYNTHETIC_VIA_DEF)
        self.assertEqual(data.total_vias, 2)
        via_names = {v.via_name for v in data.vias}
        self.assertIn("via1_4", via_names)
        self.assertIn("via2_5", via_names)

        v1 = [v for v in data.vias if v.via_name == "via1_4"][0]
        self.assertEqual(v1.x, 2000)
        self.assertEqual(v1.y, 2000)
        self.assertEqual(v1.layer, "metal1")

        v2 = [v for v in data.vias if v.via_name == "via2_5"][0]
        self.assertEqual(v2.x, 5000)
        self.assertEqual(v2.y, 2000)
        self.assertEqual(v2.layer, "metal2")

    def test_coordinate_reuse_syntax(self):
        """Test 4: Coordinate reuse syntax with ( * y ) and ( x * )."""
        data = DefRouteParser.parse_string(SYNTHETIC_COORDINATE_REUSE_DEF)
        self.assertEqual(data.total_segments, 2)

        # Segment 1 used ( 4000 * ) -> y remained 2000
        seg1 = data.segments[0]
        self.assertEqual((seg1.x1, seg1.y1, seg1.x2, seg1.y2), (1000, 2000, 4000, 2000))
        self.assertTrue(seg1.is_horizontal)

        # Segment 2 used ( * 7000 ) -> x remained 4000
        seg2 = data.segments[1]
        self.assertEqual((seg2.x1, seg2.y1, seg2.x2, seg2.y2), (4000, 2000, 4000, 7000))
        self.assertTrue(seg2.is_vertical)


class TestSegmentRasterizer(unittest.TestCase):
    """Test length-conserving coordinate-to-grid segment rasterization."""

    def setUp(self):
        self.die = DefDieArea(x_min=0, y_min=0, x_max=10000, y_max=10000, dbu_per_micron=1000)
        self.rasterizer = SegmentRasterizer(self.die, grid_size=10)

    def test_segment_rasterization_length_conservation(self):
        """Test 6: Segment rasterization length conservation across grid boundaries."""
        # Horizontal conservation test
        grid_h = self.rasterizer.create_empty_grid()
        seg_h = DefRouteSegment(
            net_name="n",
            layer="metal2",
            x1=1500,
            y1=3500,
            x2=8500,
            y2=3500,
            length_dbu=7000,
            length_um=7.0,
            orientation="horizontal",
        )
        self.rasterizer.rasterize_segment(seg_h, grid_h, value=7.0)
        self.assertAlmostEqual(float(grid_h.sum()), 7.0, places=5)
        # Check boundary cells fractional allocation
        self.assertAlmostEqual(grid_h[3, 1], 0.5, places=5)
        self.assertAlmostEqual(grid_h[3, 8], 0.5, places=5)

        # Vertical conservation test
        grid_v = self.rasterizer.create_empty_grid()
        seg_v = DefRouteSegment(
            net_name="n",
            layer="metal3",
            x1=5500,
            y1=2000,
            x2=5500,
            y2=7000,
            length_dbu=5000,
            length_um=5.0,
            orientation="vertical",
        )
        self.rasterizer.rasterize_segment(seg_v, grid_v, value=5.0)
        self.assertAlmostEqual(float(grid_v.sum()), 5.0, places=5)

    def test_via_rasterization(self):
        grid = self.rasterizer.create_empty_grid()
        via = DefVia(net_name="n", via_name="via1_4", layer="metal1", x=3500, y=7500)
        self.rasterizer.rasterize_via(via, grid, count=1.0)
        self.assertEqual(grid.sum(), 1.0)
        self.assertEqual(grid[7, 3], 1.0)


class TestRoutingLabelExtractor(unittest.TestCase):
    """Test 8-channel tensor assembly, normalization, metadata, and artifact saving."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.def_file = self.tmp_path / "synthetic_routes.def"
        self.def_file.write_text(SYNTHETIC_MULTI_LAYER_DEF, encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_layer_specific_channel_assignment(self):
        """Test 5: Layer-specific channel assignment (metal1, metal2, metal3, metal4+)."""
        extractor = RoutingLabelExtractor(grid_size=8)
        norm_labels, raw_labels, metadata = extractor.extract_from_def_file(self.def_file)

        # In SYNTHETIC_MULTI_LAYER_DEF:
        # net1: metal1 dx=4000 (2.0 um), metal2 dy=6000 (3.0 um)
        # net2: metal3 dx=4000 (2.0 um), metal4 dy=8000 (4.0 um)
        # Channel 4: metal1_wirelength_density
        # Channel 5: metal2_wirelength_density
        # Channel 6: metal3_wirelength_density
        # Channel 7: metal4plus_wirelength_density
        self.assertAlmostEqual(float(raw_labels[4].sum()), 2.0, places=4)
        self.assertAlmostEqual(float(raw_labels[5].sum()), 3.0, places=4)
        self.assertAlmostEqual(float(raw_labels[6].sum()), 2.0, places=4)
        self.assertAlmostEqual(float(raw_labels[7].sum()), 4.0, places=4)

        # Sum of layer channels equals total wirelength channel (channel 0)
        layer_sum = float(raw_labels[4:8].sum())
        total_wl = float(raw_labels[0].sum())
        self.assertAlmostEqual(layer_sum, total_wl, places=4)

    def test_label_tensor_shape_dtype_and_finite_values(self):
        """Test 7: Label tensor shape (8, H, W), float32 dtype, and finite values."""
        extractor = RoutingLabelExtractor(grid_size=8)
        norm_labels, raw_labels, metadata = extractor.extract_from_def_file(self.def_file)

        # Expected shape (8, 8, 8) and dtype float32
        self.assertEqual(norm_labels.shape, (8, 8, 8))
        self.assertEqual(raw_labels.shape, (8, 8, 8))
        self.assertEqual(norm_labels.dtype, np.float32)
        self.assertEqual(raw_labels.dtype, np.float32)
        self.assertEqual(len(metadata["channel_names"]), 8)
        self.assertEqual(metadata["channel_names"], ROUTING_CHANNEL_NAMES)

        # All values strictly finite
        self.assertTrue(np.all(np.isfinite(norm_labels)))
        self.assertTrue(np.all(np.isfinite(raw_labels)))

        # Normalized values in [0.0, 1.0]
        self.assertTrue(np.all(norm_labels >= 0.0) and np.all(norm_labels <= 1.0))

        # Check total wirelength sum matches synthetic data (11.0 um)
        self.assertAlmostEqual(float(raw_labels[0].sum()), 11.0, places=4)

        # Check via count channel (idx 3) equals 2
        self.assertEqual(float(raw_labels[3].sum()), 2.0)

    def test_process_and_save_sample(self):
        extractor = RoutingLabelExtractor(grid_size=16)
        out_dir = self.tmp_path / "sample_out"
        artifacts = extractor.process_and_save_sample(
            def_path=self.def_file,
            output_dir=out_dir,
            design_name="synth_test",
        )

        self.assertTrue(artifacts["npz"].is_file())
        self.assertTrue(artifacts["metadata"].is_file())
        self.assertTrue(artifacts["preview"].is_file())

        # Validate npz contents and shapes
        npz = np.load(artifacts["npz"])
        self.assertIn("labels", npz)
        self.assertIn("raw_labels", npz)
        self.assertEqual(npz["labels"].shape, (8, 16, 16))
        self.assertEqual(npz["raw_labels"].shape, (8, 16, 16))

        # Validate metadata
        meta = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(meta["tensor_shape"], [8, 16, 16])
        self.assertEqual(meta["routing_statistics"]["total_vias"], 2)


class TestRealUartRoutingExtraction(unittest.TestCase):
    """
    Test 8: Integration test against UART final DEF.
    Skips gracefully if the DEF artifact is missing.
    Does NOT assert hard-coded exact net/segment/via counts; verifies sane nonzero counts.
    """

    def test_uart_final_def_routing_extraction(self):
        run_dir = Path("runs/uart_tx_nangate45_20261011_064013")
        def_path = run_dir / "scripts" / "uart_tx_final.def"
        if not def_path.is_file():
            def_path = run_dir / "uart_tx_final.def"

        if not def_path.is_file():
            self.skipTest(f"UART run artifact not found at {def_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = Path(tmp_dir) / "uart_routing_sample"
            extractor = RoutingLabelExtractor(grid_size=64)
            artifacts = extractor.process_and_save_sample(
                def_path=def_path,
                output_dir=sample_dir,
                source_run_dir=run_dir,
                design_name="uart_tx",
                pdk_name="nangate45",
            )

            # Artifact existence
            self.assertTrue(artifacts["npz"].is_file())
            self.assertTrue(artifacts["metadata"].is_file())
            self.assertTrue(artifacts["preview"].is_file())

            # Load and validate tensor
            npz = np.load(artifacts["npz"])
            tensor = npz["labels"]
            raw_tensor = npz["raw_labels"]

            # Shapes and dtypes
            self.assertEqual(tensor.shape, (8, 64, 64))
            self.assertEqual(raw_tensor.shape, (8, 64, 64))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertEqual(raw_tensor.dtype, np.float32)

            # Strict finiteness
            self.assertTrue(np.all(np.isfinite(tensor)))
            self.assertTrue(np.all(np.isfinite(raw_tensor)))

            # Validate wirelength density is non-zero
            total_wl = float(raw_tensor[0].sum())
            self.assertGreater(total_wl, 0.0)
            self.assertGreater(int(np.count_nonzero(tensor[0])), 0)

            # Validate metadata statistics have sane nonzero counts (not hardcoded exact numbers)
            meta = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
            r_stats = meta["routing_statistics"]
            self.assertGreater(r_stats["routed_nets"], 0)
            self.assertGreater(r_stats["total_segments"], 0)
            self.assertGreater(r_stats["total_vias"], 0)
            self.assertEqual(len(meta["channel_names"]), 8)


if __name__ == "__main__":
    unittest.main()
