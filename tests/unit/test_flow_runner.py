"""
Unit tests for FlowRunner EDA orchestration, dry-runs, PDK checks, and manifest output.
"""

import json
import unittest
import tempfile
from pathlib import Path
from dft_hotspot.eda import FlowRunner


class TestFlowRunner(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Create dummy RTL file
        self.rtl_dir = self.tmp_path / "rtl"
        self.rtl_dir.mkdir(parents=True, exist_ok=True)
        self.rtl_file = self.rtl_dir / "test_design.v"
        self.rtl_file.write_text("module test_design(input clk, output reg out); endmodule\n", encoding="utf-8")

        # Create dummy config
        self.config_file = self.tmp_path / "test_flow.yaml"
        self.config_content = f"""
design:
  name: "test_design"
  top_module: "test_design"
  rtl_files:
    - "{self.rtl_file.as_posix()}"
  clock_port: "clk"

pdk:
  name: "dummy_pdk"
  liberty_file: "{ (self.tmp_path / 'dummy.lib').as_posix() }"
  tech_lef: "{ (self.tmp_path / 'dummy.tech.lef').as_posix() }"
  stdcell_lef: "{ (self.tmp_path / 'dummy.macro.lef').as_posix() }"

flow:
  backend: "openroad_standalone"
"""
        self.config_file.write_text(self.config_content, encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_dry_run_creates_manifest_and_scripts(self):
        out_dir = self.tmp_path / "runs"
        manifest = FlowRunner.run_flow(self.config_file, dry_run=True, output_root=out_dir)

        self.assertEqual(manifest["status"], "dry_run")
        self.assertTrue(manifest["dry_run"])
        self.assertTrue(manifest["dry_run_completed"])
        self.assertTrue(Path(manifest["run_dir"]).exists())
        self.assertTrue((Path(manifest["run_dir"]) / "manifest.json").exists())
        self.assertTrue((Path(manifest["run_dir"]) / "scripts" / "synth.tcl").exists())
        self.assertTrue((Path(manifest["run_dir"]) / "scripts" / "openroad.tcl").exists())
        self.assertTrue((Path(manifest["run_dir"]) / "rtl" / "test_design.v").exists())

    def test_missing_pdk_files_fails(self):
        out_dir = self.tmp_path / "runs"
        manifest = FlowRunner.run_flow(self.config_file, dry_run=False, output_root=out_dir)

        self.assertEqual(manifest["status"], "blocked_missing_pdk_assets")
        self.assertIsNotNone(manifest["failure_reason"])
        self.assertIn("Missing PDK physical assets", manifest["failure_reason"])
        self.assertTrue((Path(manifest["run_dir"]) / "manifest.json").exists())

    def test_blocked_missing_eda_env(self):
        # Create dummy valid PDK files so PDK validation and Yosys lib parser pass
        dummy_lib_content = """library(dummy_pdk) {
  delay_model : table_lookup;
  cell(buf) {
    pin(A) { direction: input; }
    pin(Y) { direction: output; function: "A"; }
  }
}"""
        (self.tmp_path / 'dummy.lib').write_text(dummy_lib_content, encoding="utf-8")
        (self.tmp_path / 'dummy.tech.lef').touch()
        (self.tmp_path / 'dummy.macro.lef').touch()

        out_dir = self.tmp_path / "runs"
        manifest = FlowRunner.run_flow(self.config_file, dry_run=False, output_root=out_dir)

        # On environments where OpenROAD is missing, status should be blocked_missing_eda_env
        tool_status = manifest["tool_availability"]
        if not tool_status["eda_ready"]:
            self.assertEqual(manifest["status"], "blocked_missing_eda_env")
            self.assertIn("OpenROAD", manifest["failure_reason"])
        
        self.assertTrue((Path(manifest["run_dir"]) / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
