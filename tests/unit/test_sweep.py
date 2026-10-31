"""
Unit tests for Stage 4a parameter sweep configuration, expansion, and manifest schemas.
Ensures deterministic parameter expansion, unique ID generation, dry-run plan behavior,
and compliance with the dataset manifest schema without requiring full EDA execution.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import yaml

from dft_hotspot.sweep import SweepConfig, SweepPoint, SweepRunner


class TestSweepConfigAndExpansion(unittest.TestCase):
    """Test suite for SweepConfig parsing, expansion, and planning."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.tmp_dir.name)

        # Create dummy base flow config
        self.base_flow_yaml = self.root_path / "uart_nangate45.yaml"
        self.base_flow_yaml.write_text(
            yaml.safe_dump(
                {
                    "design": {"name": "uart_tx", "clock_period": 10.0, "clock_port": "clk"},
                    "pdk": {"name": "nangate45"},
                    "flow": {"target_utilization": 0.60, "backend": "openroad_standalone"},
                }
            ),
            encoding="utf-8",
        )

        # Create dummy sweep config
        self.sweep_yaml = self.root_path / "uart_smoke.yaml"
        self.sweep_dict = {
            "dataset_id": "uart_test_sweep_v001",
            "schema_version": "sweep_config_v001",
            "design": "uart_tx",
            "platform": "nangate45",
            "base_flow_config": str(self.base_flow_yaml),
            "grid_size": 64,
            "chain_count": 1,
            "ordering": "row_major_by_y_then_x",
            "scan_source": "synthetic_proxy",
            "output_dataset_dir": str(self.root_path / "dataset"),
            "sweep_parameters": {
                "utilization": [0.45, 0.55, 0.65],
                "clock_period_ns": [3.0, 5.0],
            },
        }
        self.sweep_yaml.write_text(yaml.safe_dump(self.sweep_dict), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_load_sweep_config(self) -> None:
        """Verifies loading and parsing of sweep configuration."""
        cfg = SweepConfig.load(self.sweep_yaml)
        self.assertEqual(cfg.dataset_id, "uart_test_sweep_v001")
        self.assertEqual(cfg.design, "uart_tx")
        self.assertEqual(cfg.platform, "nangate45")
        self.assertEqual(cfg.grid_size, 64)
        self.assertEqual(cfg.chain_count, 1)
        self.assertEqual(cfg.ordering, "row_major_by_y_then_x")
        self.assertEqual(cfg.scan_source, "synthetic_proxy")
        self.assertEqual(cfg.output_dataset_dir, self.root_path / "dataset")

    def test_deterministic_expansion_count(self) -> None:
        """Verifies exact 6-run Cartesian product expansion and values."""
        cfg = SweepConfig.load(self.sweep_yaml)
        points = cfg.expand_points()

        # 3 utilizations * 2 clock periods = 6 points
        self.assertEqual(len(points), 6)

        # First point: util=0.45, clk=3.0
        self.assertEqual(points[0].index, 0)
        self.assertAlmostEqual(points[0].utilization, 0.45)
        self.assertAlmostEqual(points[0].clock_period_ns, 3.0)

        # Second point: util=0.45, clk=5.0
        self.assertEqual(points[1].index, 1)
        self.assertAlmostEqual(points[1].utilization, 0.45)
        self.assertAlmostEqual(points[1].clock_period_ns, 5.0)

        # Last point: util=0.65, clk=5.0
        self.assertEqual(points[5].index, 5)
        self.assertAlmostEqual(points[5].utilization, 0.65)
        self.assertAlmostEqual(points[5].clock_period_ns, 5.0)

    def test_unique_sample_and_run_ids(self) -> None:
        """Verifies that all generated sample_id and run_id strings are unique and deterministic."""
        cfg = SweepConfig.load(self.sweep_yaml)
        points = cfg.expand_points()

        sample_ids = [p.sample_id for p in points]
        run_ids = [p.run_id for p in points]

        self.assertEqual(len(sample_ids), len(set(sample_ids)), "Sample IDs must be unique across sweep")
        self.assertEqual(len(run_ids), len(set(run_ids)), "Run IDs must be unique across sweep")

        self.assertIn("uart_tx_nangate45_u45_clk3p0", sample_ids)
        self.assertIn("uart_tx_nangate45_u65_clk5p0", sample_ids)

    def test_dry_run_plan_behavior(self) -> None:
        """Verifies that dry-run plan produces plan text and writes no files."""
        cfg = SweepConfig.load(self.sweep_yaml)
        runner = SweepRunner(config=cfg, dry_run_plan=True)

        plan_text = runner.generate_plan_text()
        self.assertIn("STAGE 4a: MULTI-RUN PARAMETER SWEEP PLAN", plan_text)
        self.assertIn("Total Planned Runs  : 6", plan_text)
        self.assertIn("uart_tx_nangate45_u45_clk3p0", plan_text)
        self.assertIn("uart_tx_nangate45_u65_clk5p0", plan_text)

        res = runner.run_sweep()
        self.assertEqual(res["status"], "dry_run_plan")
        self.assertEqual(len(res["planned_points"]), 6)

        # Confirm no artifacts were written to output_dataset_dir
        self.assertFalse(cfg.output_dataset_dir.exists())

    def test_limit_flag_on_expansion(self) -> None:
        """Verifies that the limit flag truncates active sweep points."""
        cfg = SweepConfig.load(self.sweep_yaml)
        runner = SweepRunner(config=cfg, dry_run_plan=True, limit=2)

        res = runner.run_sweep()
        self.assertEqual(res["sample_count"], 2)
        self.assertEqual(len(res["planned_points"]), 2)

    def test_manifest_and_failure_schema_construction(self) -> None:
        """Verifies dataset manifest JSON, samples.txt, failures.json, and dataset_summary.md schemas."""
        cfg = SweepConfig.load(self.sweep_yaml)
        runner = SweepRunner(config=cfg)

        cfg.output_dataset_dir.mkdir(parents=True, exist_ok=True)

        mock_successful = [
            {
                "sample_id": "uart_tx_nangate45_u45_clk3p0",
                "run_id": "uart_tx_nangate45_u45_clk3p0",
                "run_dir": "/runs/mock_run_0",
                "sample_dir": str(cfg.samples_dir / "uart_tx_nangate45_u45_clk3p0"),
                "status": "success",
                "config": {
                    "utilization": 0.45,
                    "clock_period_ns": 3.0,
                    "chain_count": 1,
                    "grid_size": 64,
                },
                "artifacts": {
                    "features_spatial": "/mock/features_spatial.npz",
                    "labels_routing": "/mock/labels_routing.npz",
                    "scan_metadata": "/mock/scan_metadata.json",
                    "features_scan": "/mock/features_scan.npz",
                    "fused_input": "/mock/fused_input.npz",
                    "metadata": "/mock/metadata.json",
                    "routing_label_metadata": "/mock/routing_label_metadata.json",
                    "scan_feature_metadata": "/mock/scan_feature_metadata.json",
                    "fused_input_metadata": "/mock/fused_input_metadata.json",
                },
                "tensor_shapes": {
                    "physical": [7, 64, 64],
                    "routing_labels": [8, 64, 64],
                    "scan": [5, 64, 64],
                    "fused": [12, 64, 64],
                },
                "validation": {"sample_validation": "pass"},
            }
        ]

        mock_failures = [
            {
                "config": {
                    "utilization": 0.65,
                    "clock_period_ns": 3.0,
                    "chain_count": 1,
                    "grid_size": 64,
                },
                "stage": "run_flow",
                "status": "failed",
                "reason": "OpenROAD physical design execution failed: Routing congestion",
                "sample_id": "uart_tx_nangate45_u65_clk3p0",
                "run_dir": "/runs/mock_fail_run",
            }
        ]

        active_points = cfg.expand_points()
        manifest = runner._write_dataset_manifest(active_points, mock_successful, mock_failures)
        runner._write_dataset_files(mock_successful, mock_failures)

        # 1. Check dataset_manifest.json
        manifest_file = cfg.output_dataset_dir / "dataset_manifest.json"
        self.assertTrue(manifest_file.is_file())
        data = json.loads(manifest_file.read_text(encoding="utf-8"))

        self.assertEqual(data["dataset_id"], "uart_test_sweep_v001")
        self.assertEqual(data["schema_version"], "dataset_manifest_v001")
        self.assertEqual(data["design"], "uart_tx")
        self.assertEqual(data["platform"], "nangate45")
        self.assertEqual(data["grid_size"], 64)
        self.assertEqual(data["scan_source"], "synthetic_proxy")
        self.assertEqual(data["success_count"], 1)
        self.assertEqual(data["failure_count"], 1)
        self.assertEqual(len(data["samples"]), 1)
        self.assertEqual(len(data["failures"]), 1)

        # 2. Check samples.txt
        samples_txt = cfg.output_dataset_dir / "samples.txt"
        self.assertTrue(samples_txt.is_file())
        self.assertIn("uart_tx_nangate45_u45_clk3p0", samples_txt.read_text(encoding="utf-8"))

        # 3. Check failures.json
        failures_json = cfg.output_dataset_dir / "failures.json"
        self.assertTrue(failures_json.is_file())
        fail_data = json.loads(failures_json.read_text(encoding="utf-8"))
        self.assertEqual(fail_data[0]["stage"], "run_flow")
        self.assertIn("congestion", fail_data[0]["reason"])

        # 4. Check dataset_summary.md
        summary_md = cfg.output_dataset_dir / "dataset_summary.md"
        self.assertTrue(summary_md.is_file())
        md_content = summary_md.read_text(encoding="utf-8")
        self.assertIn("# Dataset Summary: uart_test_sweep_v001", md_content)
        self.assertIn("uart_tx_nangate45_u45_clk3p0", md_content)
        self.assertIn("Routing congestion", md_content)

    def test_determine_failure_stage(self) -> None:
        """Verifies stage categorization heuristic from exceptions."""
        cfg = SweepConfig.load(self.sweep_yaml)
        runner = SweepRunner(config=cfg)

        self.assertEqual(runner._determine_failure_stage(RuntimeError("Physical design flow failed: timeout")), "run_flow")
        self.assertEqual(runner._determine_failure_stage(ValueError("PhysicalFeatureExtractor failed to bin components")), "features")
        self.assertEqual(runner._determine_failure_stage(RuntimeError("RoutingLabelExtractor could not find wires")), "labels")
        self.assertEqual(runner._determine_failure_stage(ValueError("SyntheticScan generator failed")), "scan_metadata")
        self.assertEqual(runner._determine_failure_stage(ValueError("ScanFeatureExtractor shape mismatch")), "scan_features")
        self.assertEqual(runner._determine_failure_stage(ValueError("Validate sample resolution mismatch")), "validate")
        self.assertEqual(runner._determine_failure_stage(ValueError("fused export error")), "fused_export")

    def test_multi_design_sweep_expansion(self) -> None:
        """Verifies multi-design configuration loading and expansion with aspect ratio."""
        multi_yaml = self.root_path / "multi_sweep.yaml"
        multi_dict = {
            "dataset_id": "multi_test_sweep_v001",
            "schema_version": "sweep_config_v002",
            "design": "multi",
            "platform": "nangate45",
            "grid_size": 64,
            "chain_count": 1,
            "scan_source": "synthetic_proxy",
            "output_dataset_dir": str(self.root_path / "multi_dataset"),
            "designs": [
                {
                    "design": "uart_tx",
                    "platform": "nangate45",
                    "base_flow_config": str(self.base_flow_yaml),
                    "sweep_points": [
                        {"utilization": 0.45, "clock_period_ns": 3.0, "aspect_ratio": 1.0},
                        {"utilization": 0.45, "clock_period_ns": 3.0, "aspect_ratio": 1.5},
                    ],
                },
                {
                    "design": "dft_counter",
                    "platform": "nangate45",
                    "base_flow_config": str(self.base_flow_yaml),
                    "sweep_points": [
                        {"utilization": 0.55, "clock_period_ns": 3.0, "aspect_ratio": 1.0},
                    ],
                },
            ],
        }
        multi_yaml.write_text(yaml.safe_dump(multi_dict), encoding="utf-8")

        cfg = SweepConfig.load(multi_yaml)
        points = cfg.expand_points()

        self.assertEqual(len(points), 3)
        self.assertEqual(points[0].design, "uart_tx")
        self.assertEqual(points[0].sample_id, "uart_tx_nangate45_u45_clk3p0_ar1p0")
        self.assertAlmostEqual(points[0].aspect_ratio, 1.0)

        self.assertEqual(points[1].design, "uart_tx")
        self.assertEqual(points[1].sample_id, "uart_tx_nangate45_u45_clk3p0_ar1p5")
        self.assertAlmostEqual(points[1].aspect_ratio, 1.5)

        self.assertEqual(points[2].design, "dft_counter")
        self.assertEqual(points[2].sample_id, "dft_counter_nangate45_u55_clk3p0_ar1p0")
        self.assertAlmostEqual(points[2].aspect_ratio, 1.0)

        runner = SweepRunner(config=cfg, dry_run_plan=True)
        res = runner.run_sweep()
        self.assertEqual(res["sample_count"], 3)
        self.assertEqual(len(res["planned_points"]), 3)

    def test_multi_design_chain_count_sweep_expansion(self) -> None:
        """Verifies multi-design parameter expansion across 3 designs and chain counts (36 planned runs)."""
        expanded_cfg_path = Path("configs/sweeps/expanded_v002.yaml").resolve()
        if not expanded_cfg_path.is_file():
            self.skipTest("configs/sweeps/expanded_v002.yaml not found on disk")

        cfg = SweepConfig.load(expanded_cfg_path)
        points = cfg.expand_points()

        # 3 designs * 3 utilizations * 1 clock * 2 aspect ratios * 2 chain counts = 36 runs
        self.assertEqual(len(points), 36)

        sample_ids = [p.sample_id for p in points]
        run_ids = [p.run_id for p in points]

        # Verify all IDs are completely unique
        self.assertEqual(len(sample_ids), len(set(sample_ids)), "All 36 sample IDs must be unique")
        self.assertEqual(len(run_ids), len(set(run_ids)), "All 36 run IDs must be unique")

        # Verify all 3 designs are represented (12 runs each)
        designs = [p.design for p in points]
        self.assertEqual(designs.count("uart_tx"), 12)
        self.assertEqual(designs.count("dft_counter"), 12)
        self.assertEqual(designs.count("spi"), 12)

        # Verify both chain counts are present (18 runs each)
        chain_counts = [p.chain_count for p in points]
        self.assertEqual(chain_counts.count(1), 18)
        self.assertEqual(chain_counts.count(2), 18)

        # Verify naming contains _chains1 and _chains2
        for sid in sample_ids:
            self.assertTrue("_chains1" in sid or "_chains2" in sid, f"Sample ID '{sid}' must include chain count suffix")

        # Dry run runner
        runner = SweepRunner(config=cfg, dry_run_plan=True)
        res = runner.run_sweep()
        self.assertEqual(res["sample_count"], 36)
        self.assertEqual(len(res["planned_points"]), 36)


if __name__ == "__main__":
    unittest.main()


