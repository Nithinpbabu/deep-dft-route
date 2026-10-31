"""
Unit and integration tests for OpenROAD scan metadata extraction (Stage 3c).
"""

import json
import tempfile
import unittest
from pathlib import Path

from dft_hotspot.dft.scan_metadata import (
    ScanCell,
    ScanChain,
    ScanEndpoint,
    ScanMetadata,
    ScanMetadataExtractor,
    NoScanChainsFoundError,
)
from dft_hotspot.features.def_parser import DefParser


STAGE3B_SCANCHAINS_SNIPPET = """
VERSION 5.8 ;
DESIGN dft_counter ;
UNITS DISTANCE MICRONS 1000 ;

SCANCHAINS 1 ;

- chain_0
+ START PIN scan_in_0
+ ORDERED
  _142_ ( IN SCD ) ( OUT Q )
  _135_ ( IN SCD ) ( OUT Q )
  _136_ ( IN SCD ) ( OUT Q )
  _137_ ( IN SCD ) ( OUT Q )
  _138_ ( IN SCD ) ( OUT Q )
  _134_ ( IN SCD ) ( OUT Q )
  _141_ ( IN SCD ) ( OUT Q )
  _140_ ( IN SCD ) ( OUT Q )
  _139_ ( IN SCD ) ( OUT Q )
  _149_ ( IN SCD ) ( OUT Q )
  _148_ ( IN SCD ) ( OUT Q )
  _147_ ( IN SCD ) ( OUT Q )
  _144_ ( IN SCD ) ( OUT Q )
  _145_ ( IN SCD ) ( OUT Q )
  _146_ ( IN SCD ) ( OUT Q )
  _143_ ( IN SCD ) ( OUT Q )
+ PARTITION default
+ STOP PIN q[8] ;

END SCANCHAINS

END DESIGN
"""


class TestScanMetadata(unittest.TestCase):
    """Test suite for DEF SCANCHAINS parsing and metadata extraction."""

    def test_parse_stage3b_scanchains_snippet(self):
        """Verifies parsing of the verified Stage 3b SCANCHAINS snippet."""
        chains = ScanMetadataExtractor.parse_def_scanchains(STAGE3B_SCANCHAINS_SNIPPET)
        self.assertEqual(len(chains), 1)
        chain = chains[0]
        self.assertEqual(chain.chain_id, "chain_0")
        self.assertEqual(chain.cell_count, 16)
        self.assertEqual(len(chain.cells), 16)

    def test_ordered_cell_sequence_preservation(self):
        """Verifies that the exact cell order and pin directions are preserved."""
        chains = ScanMetadataExtractor.parse_def_scanchains(STAGE3B_SCANCHAINS_SNIPPET)
        chain = chains[0]

        expected_instances = [
            "_142_", "_135_", "_136_", "_137_", "_138_", "_134_",
            "_141_", "_140_", "_139_", "_149_", "_148_", "_147_",
            "_144_", "_145_", "_146_", "_143_"
        ]
        self.assertEqual([c.instance for c in chain.cells], expected_instances)

        # Check indexing and pin tuples
        for idx, cell in enumerate(chain.cells):
            self.assertEqual(cell.index, idx)
            self.assertEqual(cell.scan_in_pin, "SCD")
            self.assertEqual(cell.scan_out_pin, "Q")

    def test_start_stop_partition_parsing(self):
        """Verifies parsing of start, stop, and partition fields."""
        chains = ScanMetadataExtractor.parse_def_scanchains(STAGE3B_SCANCHAINS_SNIPPET)
        chain = chains[0]

        self.assertEqual(chain.start.type, "PIN")
        self.assertEqual(chain.start.name, "scan_in_0")
        self.assertEqual(chain.stop.type, "PIN")
        self.assertEqual(chain.stop.name, "q[8]")
        self.assertEqual(chain.partition, "default")

    def test_multiple_chains_parsing(self):
        """Verifies parsing of synthetic DEF with multiple scan chains."""
        multi_chain_def = """
SCANCHAINS 2 ;

- chain_0
+ START PIN scan_in_0
+ ORDERED
  ff_0 ( IN SI ) ( OUT SO )
  ff_1 ( IN SI ) ( OUT SO )
+ PARTITION p_clk0
+ STOP PIN scan_out_0 ;

- chain_1
+ START PIN scan_in_1
+ ORDERED
  ff_2 ( IN SI ) ( OUT SO )
  ff_3 ( IN SI ) ( OUT SO )
  ff_4 ( IN SI ) ( OUT SO )
+ PARTITION p_clk1
+ STOP PIN scan_out_1 ;

END SCANCHAINS
"""
        chains = ScanMetadataExtractor.parse_def_scanchains(multi_chain_def)
        self.assertEqual(len(chains), 2)

        self.assertEqual(chains[0].chain_id, "chain_0")
        self.assertEqual(chains[0].cell_count, 2)
        self.assertEqual(chains[0].start.name, "scan_in_0")
        self.assertEqual(chains[0].stop.name, "scan_out_0")
        self.assertEqual(chains[0].partition, "p_clk0")
        self.assertEqual([c.instance for c in chains[0].cells], ["ff_0", "ff_1"])

        self.assertEqual(chains[1].chain_id, "chain_1")
        self.assertEqual(chains[1].cell_count, 3)
        self.assertEqual(chains[1].start.name, "scan_in_1")
        self.assertEqual(chains[1].stop.name, "scan_out_1")
        self.assertEqual(chains[1].partition, "p_clk1")
        self.assertEqual([c.instance for c in chains[1].cells], ["ff_2", "ff_3", "ff_4"])

    def test_placement_enrichment(self):
        """Verifies cell enrichment with master, coordinates (DBU & microns), and placement status."""
        synthetic_def = """
VERSION 5.8 ;
UNITS DISTANCE MICRONS 1000 ;

COMPONENTS 2 ;
- c0 sky130_fd_sc_hd__sdfrbp_1 + PLACED ( 40000 20000 ) N ;
- c1 sky130_fd_sc_hd__sdfrbp_1 + UNPLACED ;
END COMPONENTS

SCANCHAINS 1 ;
- chain_0
+ START PIN scan_in_0
+ ORDERED
  c0 ( IN SCD ) ( OUT Q )
  c1 ( IN SCD ) ( OUT Q )
  c2 ( IN SCD ) ( OUT Q )
+ STOP PIN q[0] ;
END SCANCHAINS
"""
        def_data = DefParser.parse_string(synthetic_def)
        chains = ScanMetadataExtractor.parse_def_scanchains(synthetic_def)
        ScanMetadataExtractor.enrich_cells_from_components(chains, def_data)

        chain = chains[0]
        # c0: placed
        self.assertTrue(chain.cells[0].placed)
        self.assertEqual(chain.cells[0].master, "sky130_fd_sc_hd__sdfrbp_1")
        self.assertEqual(chain.cells[0].x, 40000)
        self.assertEqual(chain.cells[0].y, 20000)
        self.assertEqual(chain.cells[0].x_um, 40.0)
        self.assertEqual(chain.cells[0].y_um, 20.0)

        # c1: unplaced
        self.assertFalse(chain.cells[1].placed)
        self.assertEqual(chain.cells[1].master, "sky130_fd_sc_hd__sdfrbp_1")
        self.assertIsNone(chain.cells[1].x)

        # c2: missing from components
        self.assertFalse(chain.cells[2].placed)
        self.assertIsNone(chain.cells[2].master)
        self.assertIsNone(chain.cells[2].x)

    def test_no_scanchains_behavior(self):
        """Verifies error raising on missing SCANCHAINS, and fallback when allow_missing is True."""
        no_chains_def = """
VERSION 5.8 ;
DESIGN dummy ;
COMPONENTS 1 ;
- inst_0 DFF_X1 + PLACED ( 100 200 ) N ;
END COMPONENTS
END DESIGN
"""
        # Should raise NoScanChainsFoundError by default
        with self.assertRaises(NoScanChainsFoundError):
            ScanMetadataExtractor.parse_def_scanchains(no_chains_def, allow_missing=False)

        # Should return empty list when allow_missing is True
        chains = ScanMetadataExtractor.parse_def_scanchains(no_chains_def, allow_missing=True)
        self.assertEqual(chains, [])

    def test_json_serialization_schema(self):
        """Verifies that serialized dictionary matches scan_metadata_v001 schema requirements."""
        synthetic_def = """
VERSION 5.8 ;
DESIGN test_design ;
UNITS DISTANCE MICRONS 1000 ;

COMPONENTS 1 ;
- ff0 SDFF_X1 + PLACED ( 1000 2000 ) N ;
END COMPONENTS

SCANCHAINS 1 ;
- chain_0
+ START PIN scan_in_0
+ ORDERED
  ff0 ( IN SI ) ( OUT SO )
+ STOP PIN scan_out_0 ;
END SCANCHAINS
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            def_path = tmp_p / "test_design_scan.def"
            def_path.write_text(synthetic_def, encoding="utf-8")

            metadata = ScanMetadataExtractor.extract_from_directory(
                dft_dir=tmp_p,
                def_path=def_path,
            )

            out_json = tmp_p / "scan_metadata.json"
            metadata.save_json(out_json)
            self.assertTrue(out_json.is_file())

            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check required top-level keys
            expected_keys = {
                "schema_version", "source", "design", "platform",
                "source_dir", "def_path", "netlist_path", "created_at",
                "scan_config", "scan_ports", "chains", "statistics",
                "limitations"
            }
            self.assertTrue(expected_keys.issubset(set(data.keys())))
            self.assertEqual(data["schema_version"], "scan_metadata_v001")
            self.assertEqual(data["source"], "openroad_dft")
            self.assertEqual(data["statistics"]["chain_count"], 1)
            self.assertEqual(data["statistics"]["total_scan_cells"], 1)

            # Verify round-trip load
            loaded = ScanMetadata.load_json(out_json)
            self.assertEqual(loaded.schema_version, "scan_metadata_v001")
            self.assertEqual(loaded.statistics.chain_count, 1)
            self.assertEqual(loaded.chains[0].cells[0].instance, "ff0")

    def test_real_sky130hd_artifact_extraction(self):
        """Integration test against actual Stage 3b Sky130hd DFT output directory."""
        real_dir = Path("data/dft_validation/dft_smoke_sky130hd_20261011_185057")
        if not real_dir.is_dir():
            self.skipTest(f"Directory {real_dir} not present in workspace.")

        metadata = ScanMetadataExtractor.extract_from_directory(dft_dir=real_dir)

        self.assertEqual(metadata.schema_version, "scan_metadata_v001")
        self.assertEqual(metadata.source, "openroad_dft")
        self.assertEqual(metadata.design, "dft_counter")
        self.assertEqual(metadata.platform, "sky130hd")

        # Statistics
        self.assertEqual(metadata.statistics.chain_count, 1)
        self.assertEqual(metadata.statistics.total_scan_cells, 16)
        self.assertEqual(metadata.statistics.placed_scan_cells, 16)
        self.assertEqual(metadata.statistics.min_chain_length, 16)
        self.assertEqual(metadata.statistics.max_chain_length, 16)
        self.assertEqual(metadata.statistics.avg_chain_length, 16.0)

        # Chains
        chain = metadata.chains[0]
        self.assertEqual(chain.chain_id, "chain_0")
        self.assertEqual(chain.cell_count, 16)
        self.assertEqual(chain.start.type, "PIN")
        self.assertEqual(chain.start.name, "scan_in_0")
        self.assertEqual(chain.stop.type, "PIN")
        self.assertEqual(chain.stop.name, "q[8]")

        # First and last cell verification
        self.assertEqual(chain.cells[0].instance, "_142_")
        self.assertEqual(chain.cells[0].master, "sky130_fd_sc_hd__sdfrbp_1")
        self.assertTrue(chain.cells[0].placed)
        self.assertEqual(chain.cells[0].x, 43513)
        self.assertEqual(chain.cells[0].y, 23256)
        self.assertEqual(chain.cells[0].x_um, 43.513)
        self.assertEqual(chain.cells[0].y_um, 23.256)

        self.assertEqual(chain.cells[-1].instance, "_143_")
        self.assertEqual(chain.cells[-1].master, "sky130_fd_sc_hd__sdfrbp_1")
        self.assertTrue(chain.cells[-1].placed)

        # Scan ports
        self.assertEqual(metadata.scan_ports.scan_enable, ["scan_enable_0"])
        self.assertEqual(metadata.scan_ports.scan_in, ["scan_in_0"])
        self.assertEqual(metadata.scan_ports.scan_out, ["q[8]"])

        # Scan config
        self.assertEqual(metadata.scan_config.max_chains, 1)
        self.assertEqual(metadata.scan_config.max_length, 16)
        self.assertEqual(metadata.scan_config.clock_mixing, "No Mix")

        # Limitations
        self.assertTrue(len(metadata.limitations) >= 2)


if __name__ == "__main__":
    unittest.main()
