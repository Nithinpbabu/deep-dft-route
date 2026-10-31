"""
Unit and integration tests for synthetic scan metadata generation (Stage 3d).
"""

import json
import tempfile
import unittest
from pathlib import Path

from dft_hotspot.dft.scan_metadata import ScanMetadata
from dft_hotspot.dft.synthetic_scan import (
    DEFAULT_SYNTHETIC_LIMITATIONS,
    NoSequentialCellsFoundError,
    SyntheticConfig,
    SyntheticScanGenerator,
)
from dft_hotspot.features.def_parser import DefComponent, DefData, DefParser


SYNTHETIC_DEF_MIXED = """VERSION 5.8 ;
DESIGN toy_mixed ;
UNITS DISTANCE MICRONS 1000 ;

COMPONENTS 6 ;
- g0 NAND2_X1 + PLACED ( 1000 2000 ) N ;
- ff0 DFF_X1 + PLACED ( 3000 1000 ) N ;
- g1 INV_X1 + PLACED ( 1500 2500 ) N ;
- ff1 DFFR_X1 + PLACED ( 1000 1000 ) N ;
- ff2 DFF_X2 + PLACED ( 2000 3000 ) N ;
- g2 NOR2_X1 + PLACED ( 4000 4000 ) N ;
END COMPONENTS

END DESIGN
"""


class TestSyntheticScan(unittest.TestCase):
    """Test suite for deterministic synthetic scan chain generation."""

    def test_extract_sequential_and_exclude_combinational(self):
        """Verifies extracting sequential cells while strictly excluding combinational cells."""
        def_data = DefParser.parse_string(SYNTHETIC_DEF_MIXED)
        seq_cells = SyntheticScanGenerator.extract_sequential_cells(def_data)

        self.assertEqual(len(seq_cells), 3)
        seq_names = {c.name for c in seq_cells}
        self.assertEqual(seq_names, {"ff0", "ff1", "ff2"})

        # Combinational cells (g0, g1, g2) must be excluded
        for name in ("g0", "g1", "g2"):
            self.assertNotIn(name, seq_names)

    def test_deterministic_row_major_ordering(self):
        """Verifies sorting sequential cells by y, then x, then instance name."""
        cells = [
            DefComponent(name="ff_y20_x10", master="DFF_X1", x=10.0, y=20.0),
            DefComponent(name="ff_y10_x20", master="DFF_X1", x=20.0, y=10.0),
            DefComponent(name="ff_y10_x10_b", master="DFF_X1", x=10.0, y=10.0),
            DefComponent(name="ff_y10_x10_a", master="DFF_X1", x=10.0, y=10.0),
            DefComponent(name="ff_y30_x05", master="DFF_X1", x=5.0, y=30.0),
        ]
        ordered = SyntheticScanGenerator.order_cells(cells, method="row_major_by_y_then_x")

        expected_order = [
            "ff_y10_x10_a",  # y=10, x=10, name '...a'
            "ff_y10_x10_b",  # y=10, x=10, name '...b'
            "ff_y10_x20",    # y=10, x=20
            "ff_y20_x10",    # y=20, x=10
            "ff_y30_x05",    # y=30, x=5
        ]
        self.assertEqual([c.name for c in ordered], expected_order)

    def test_nearest_neighbor_ordering(self):
        """Verifies deterministic greedy nearest neighbor ordering."""
        cells = [
            DefComponent(name="c0", master="DFF_X1", x=0.0, y=0.0),
            DefComponent(name="c1", master="DFF_X1", x=10.0, y=0.0),
            DefComponent(name="c2", master="DFF_X1", x=2.0, y=0.0),
            DefComponent(name="c3", master="DFF_X1", x=50.0, y=50.0),
        ]
        ordered = SyntheticScanGenerator.order_cells(cells, method="nearest_neighbor")
        # Starting at lowest (0,0), closest is (2,0), then (10,0), then (50,50)
        self.assertEqual([c.name for c in ordered], ["c0", "c2", "c1", "c3"])

    def test_single_chain_generation(self):
        """Verifies single-chain generation conforms to schema and cell structure."""
        def_data = DefParser.parse_string(SYNTHETIC_DEF_MIXED)
        seq_cells = SyntheticScanGenerator.extract_sequential_cells(def_data)
        ordered = SyntheticScanGenerator.order_cells(seq_cells, method="row_major_by_y_then_x")
        groups = SyntheticScanGenerator.partition_cells(ordered, chain_count=1)
        chains = SyntheticScanGenerator.build_chains(groups, dbu=1000.0)

        self.assertEqual(len(chains), 1)
        chain = chains[0]
        self.assertEqual(chain.chain_id, "synthetic_chain_0")
        self.assertEqual(chain.start.type, "SYNTHETIC")
        self.assertEqual(chain.start.name, "synthetic_scan_in_0")
        self.assertEqual(chain.stop.type, "SYNTHETIC")
        self.assertEqual(chain.stop.name, "synthetic_scan_out_0")
        self.assertEqual(chain.partition, "synthetic_default")
        self.assertEqual(chain.cell_count, 3)

        # Check cell fields
        for idx, cell in enumerate(chain.cells):
            self.assertEqual(cell.index, idx)
            self.assertIsNone(cell.scan_in_pin)
            self.assertIsNone(cell.scan_out_pin)
            self.assertTrue(cell.placed)
            self.assertIsNotNone(cell.master)
            self.assertIsNotNone(cell.x)
            self.assertIsNotNone(cell.y)
            self.assertIsNotNone(cell.x_um)
            self.assertIsNotNone(cell.y_um)

        # In row-major:
        # ff1 (1000, 1000), ff0 (3000, 1000), ff2 (2000, 3000)
        self.assertEqual([c.instance for c in chain.cells], ["ff1", "ff0", "ff2"])

    def test_multi_chain_partitioning_contiguous_and_round_robin(self):
        """Verifies deterministic chain partitioning across multiple chains."""
        cells = [DefComponent(name=f"c_{i}", master="DFF_X1", x=float(i * 10), y=0.0) for i in range(7)]

        # Contiguous chunks: 7 cells into 3 chains -> lengths [3, 2, 2]
        chunks = SyntheticScanGenerator.partition_cells(cells, chain_count=3, method="contiguous_chunks")
        self.assertEqual(len(chunks), 3)
        self.assertEqual([len(ch) for ch in chunks], [3, 2, 2])
        self.assertEqual([c.name for c in chunks[0]], ["c_0", "c_1", "c_2"])
        self.assertEqual([c.name for c in chunks[1]], ["c_3", "c_4"])
        self.assertEqual([c.name for c in chunks[2]], ["c_5", "c_6"])

        chains = SyntheticScanGenerator.build_chains(chunks, dbu=1000.0)
        self.assertEqual(len(chains), 3)
        self.assertEqual(chains[0].chain_id, "synthetic_chain_0")
        self.assertEqual(chains[1].chain_id, "synthetic_chain_1")
        self.assertEqual(chains[2].chain_id, "synthetic_chain_2")
        self.assertEqual(chains[1].start.name, "synthetic_scan_in_1")
        self.assertEqual(chains[1].stop.name, "synthetic_scan_out_1")

        # Cell indices must be 0-based within each chain
        self.assertEqual([cell.index for cell in chains[1].cells], [0, 1])

        # Round robin: 7 cells into 3 chains -> lengths [3, 2, 2]
        rr = SyntheticScanGenerator.partition_cells(cells, chain_count=3, method="round_robin")
        self.assertEqual([c.name for c in rr[0]], ["c_0", "c_3", "c_6"])
        self.assertEqual([c.name for c in rr[1]], ["c_1", "c_4"])
        self.assertEqual([c.name for c in rr[2]], ["c_2", "c_5"])

    def test_validation_errors(self):
        """Verifies clear failures on missing DEF, empty sequential cells, unplaced cells, and bad chain count."""
        # Missing DEF
        with self.assertRaises(FileNotFoundError):
            SyntheticScanGenerator.generate(def_path="/nonexistent/path/test.def")

        # Zero sequential cells
        comb_only_def = """VERSION 5.8 ;
DESIGN comb_only ;
COMPONENTS 1 ;
- g0 NAND2_X1 + PLACED ( 100 200 ) N ;
END COMPONENTS
END DESIGN
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            def_p = Path(tmp_dir) / "comb.def"
            def_p.write_text(comb_only_def, encoding="utf-8")
            with self.assertRaises(NoSequentialCellsFoundError):
                SyntheticScanGenerator.generate(def_path=def_p)

        # Unplaced sequential cell
        unplaced_def = """VERSION 5.8 ;
DESIGN unplaced_test ;
COMPONENTS 1 ;
- ff0 DFF_X1 + UNPLACED ;
END COMPONENTS
END DESIGN
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            def_p = Path(tmp_dir) / "unplaced.def"
            def_p.write_text(unplaced_def, encoding="utf-8")
            with self.assertRaises(ValueError):
                SyntheticScanGenerator.generate(def_path=def_p)

        # Invalid chain count
        cells = [DefComponent(name="ff0", master="DFF_X1", x=0.0, y=0.0)]
        with self.assertRaises(ValueError):
            SyntheticScanGenerator.partition_cells(cells, chain_count=0)
        with self.assertRaises(ValueError):
            SyntheticScanGenerator.partition_cells(cells, chain_count=5)

    def test_json_schema_source_and_limitations(self):
        """Verifies JSON schema version, source='synthetic_proxy', and required limitation text."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            def_file = tmp_p / "toy_mixed.def"
            def_file.write_text(SYNTHETIC_DEF_MIXED, encoding="utf-8")

            metadata = SyntheticScanGenerator.generate(
                def_path=def_file,
                chain_count=1,
                ordering="row_major_by_y_then_x",
            )

            out_json = tmp_p / "scan_metadata.json"
            metadata.save_json(out_json)
            self.assertTrue(out_json.is_file())

            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["schema_version"], "scan_metadata_v001")
            self.assertEqual(data["source"], "synthetic_proxy")
            self.assertIn("synthetic_config", data)
            self.assertEqual(data["synthetic_config"]["chain_count"], 1)
            self.assertEqual(data["synthetic_config"]["ordering"], "row_major_by_y_then_x")
            self.assertEqual(data["synthetic_config"]["source_cell_filter"], "sequential_master_heuristic")
            self.assertTrue(len(data["synthetic_config"]["notes"]) >= 1)

            # Check scan_ports are empty lists
            self.assertEqual(data["scan_ports"]["scan_enable"], [])
            self.assertEqual(data["scan_ports"]["scan_in"], [])
            self.assertEqual(data["scan_ports"]["scan_out"], [])

            # Check limitations
            self.assertTrue(any("Synthetic proxy" in lim for lim in data["limitations"]))
            self.assertTrue(any("routing demand proxy" in lim for lim in data["limitations"]))

            # Roundtrip load
            loaded = ScanMetadata.load_json(out_json)
            self.assertEqual(loaded.source, "synthetic_proxy")
            self.assertIsNotNone(loaded.synthetic_config)
            self.assertEqual(loaded.statistics.total_scan_cells, 3)

    def test_real_uart_nangate45_integration(self):
        """Integration test against actual UART Nangate45 run directory."""
        run_dir = Path("runs/uart_tx_nangate45_20261011_064013")
        def_file = run_dir / "scripts" / "uart_tx_final.def"
        if not def_file.is_file():
            self.skipTest(f"UART Nangate45 DEF artifact not found at {def_file}")

        metadata = SyntheticScanGenerator.generate(
            run_dir=run_dir,
            chain_count=1,
            ordering="row_major_by_y_then_x",
        )

        self.assertEqual(metadata.schema_version, "scan_metadata_v001")
        self.assertEqual(metadata.source, "synthetic_proxy")
        self.assertEqual(metadata.design, "uart_tx")
        self.assertEqual(metadata.platform, "nangate45")

        # Exactly 33 sequential cells in UART Nangate45 layout
        self.assertEqual(metadata.statistics.total_scan_cells, 33)
        self.assertEqual(metadata.statistics.placed_scan_cells, 33)
        self.assertEqual(metadata.statistics.chain_count, 1)
        self.assertEqual(metadata.statistics.min_chain_length, 33)
        self.assertEqual(metadata.statistics.max_chain_length, 33)
        self.assertEqual(metadata.statistics.avg_chain_length, 33.0)

        chain = metadata.chains[0]
        self.assertEqual(chain.chain_id, "synthetic_chain_0")
        self.assertEqual(chain.cell_count, 33)
        self.assertEqual(len(chain.cells), 33)

        # Check row-major ordering is monotonically non-decreasing in (y, x)
        for i in range(len(chain.cells) - 1):
            c1 = chain.cells[i]
            c2 = chain.cells[i + 1]
            self.assertTrue(
                (c1.y < c2.y) or (c1.y == c2.y and c1.x <= c2.x),
                f"Row-major ordering violation at cell {i}: ({c1.x}, {c1.y}) vs ({c2.x}, {c2.y})"
            )

        # Limitations must clearly indicate synthetic proxy
        self.assertTrue(any("Synthetic proxy" in lim for lim in metadata.limitations))


if __name__ == "__main__":
    unittest.main()
