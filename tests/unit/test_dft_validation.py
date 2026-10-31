"""
Unit tests for OpenROAD DFT validation reporting, schema, log parser, and scan evidence extraction.
"""

import json
import tempfile
import unittest
from pathlib import Path

from dft_hotspot.dft import DftValidationReport, DftValidator


class TestDftValidation(unittest.TestCase):
    """Tests for DFT validation analysis, report serialization, and evidence parsing."""

    def test_report_dataclass_serialization(self):
        """Validates DftValidationReport serialization and round-trip JSON deserialization."""
        report = DftValidationReport(
            platform="sky130hd",
            design="dft_counter",
            openroad_version="26Q4-110-g91e5f5ff2f",
            liberty_has_test_cell=True,
            commands_attempted=["report_dft_config", "scan_replace", "report_dft_plan", "execute_dft_plan"],
            commands_succeeded=["report_dft_config", "scan_replace", "report_dft_plan", "execute_dft_plan"],
            failure_command=None,
            failure_reason=None,
            scan_replace_succeeded=True,
            report_dft_plan_succeeded=True,
            execute_dft_plan_succeeded=True,
            scan_metadata_available=True,
            generated_scan_ports_or_nets=["scan_in_0", "scan_enable_0"],
            limitations=[]
        )

        d = report.to_dict()
        self.assertEqual(d["platform"], "sky130hd")
        self.assertTrue(d["scan_replace_succeeded"])
        self.assertEqual(d["generated_scan_ports_or_nets"], ["scan_in_0", "scan_enable_0"])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "summary.json"
            report.save_json(json_file)
            self.assertTrue(json_file.exists())

            loaded = DftValidationReport.load_json(json_file)
            self.assertEqual(loaded.platform, "sky130hd")
            self.assertEqual(loaded.commands_succeeded, report.commands_succeeded)
            self.assertEqual(loaded.scan_metadata_available, True)

    def test_parse_def_scanchains_with_valid_chains(self):
        """Validates detection and parsing of SCANCHAINS section in DEF."""
        def_content = """
VERSION 5.8 ;
DESIGN dft_counter ;
UNITS DISTANCE MICRONS 1000 ;

SCANCHAINS 1 ;

- chain_0
+ START PIN scan_in_0
+ ORDERED
  _142_ ( IN SCD ) ( OUT Q )
  _135_ ( IN SCD ) ( OUT Q )
+ PARTITION default
+ STOP PIN q[8] ;

END SCANCHAINS

END DESIGN
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            def_path = Path(tmpdir) / "sample_scan.def"
            def_path.write_text(def_content)

            result = DftValidator.parse_def_scanchains(def_path)
            self.assertTrue(result["has_scanchains"])
            self.assertEqual(result["chain_count"], 1)
            self.assertEqual(result["chains"], ["chain_0"])

    def test_parse_def_scanchains_without_chains(self):
        """Validates parsing of DEF file without SCANCHAINS section."""
        def_content = """
VERSION 5.8 ;
DESIGN dft_counter ;
COMPONENTS 10 ;
END COMPONENTS
END DESIGN
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            def_path = Path(tmpdir) / "sample_noscan.def"
            def_path.write_text(def_content)

            result = DftValidator.parse_def_scanchains(def_path)
            self.assertFalse(result["has_scanchains"])
            self.assertEqual(result["chain_count"], 0)
            self.assertEqual(result["chains"], [])

    def test_parse_verilog_scan_ports(self):
        """Validates extraction of scan control ports from stitched Verilog netlist."""
        verilog_content = """
module dft_counter (
    clk,
    rst_n,
    scan_enable_0,
    scan_in_0,
    scan_out_0,
    q
);
input clk;
input rst_n;
input scan_enable_0;
input scan_in_0;
output scan_out_0;
output [15:0] q;
endmodule
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            v_path = Path(tmpdir) / "sample.v"
            v_path.write_text(verilog_content)

            ports = DftValidator.parse_verilog_scan_ports(v_path)
            self.assertEqual(ports, ["scan_enable_0", "scan_in_0", "scan_out_0"])

    def test_analyze_validation_run_failure_case(self):
        """Validates analysis of a failure log due to missing test_cell annotations."""
        dummy_log = """
OpenROAD 26Q4-110-g91e5f5ff2f
DFT Config Report:
Scan Architect Config:
- Max Length: 16
=== DFT_STAGE_BEGIN: SCAN_REPLACE ===
[WARNING DFT-0002] Can't scan replace cell '_016_', that has lib cell 'DFFR_X1'. No scan equivalent lib cell found
=== DFT_STAGE_END: SCAN_REPLACE ===
=== DFT_STAGE_BEGIN: REPORT_DFT_PLAN ===
[WARNING DFT-0007] Cell '_016_' is not a scan cell. Can't use it for scan architect
Number of chains: 0
=== DFT_STAGE_END: REPORT_DFT_PLAN ===
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_file = Path(tmpdir) / "dummy.lib"
            lib_file.write_text("library(dummy) { cell(DFFR_X1) {} }")

            report = DftValidator.analyze_validation_run(
                platform_name="nangate45",
                design_name="dft_counter",
                log_text=dummy_log,
                liberty_path=lib_file
            )

            self.assertFalse(report.liberty_has_test_cell)
            self.assertFalse(report.scan_replace_succeeded)
            self.assertFalse(report.report_dft_plan_succeeded)
            self.assertFalse(report.execute_dft_plan_succeeded)
            self.assertFalse(report.scan_metadata_available)
            self.assertEqual(report.failure_command, "scan_replace")
            self.assertIn("No scan equivalent", report.failure_reason)
            self.assertTrue(len(report.limitations) > 0)


if __name__ == "__main__":
    unittest.main()
