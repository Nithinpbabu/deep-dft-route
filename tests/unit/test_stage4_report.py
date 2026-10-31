"""Unit tests for Stage 4 benchmark reporting and synthesis module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dft_hotspot.reporting.stage4_benchmark import (
    build_stage4_summary_dict,
    generate_stage4_benchmark_report,
    load_stage4_artifacts,
    render_stage4_benchmark_markdown,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT_DIR / "data" / "processed" / "datasets" / "uart_nangate45_smoke_v001"
HEURISTICS_DIR = ROOT_DIR / "experiments" / "heuristics" / "uart_nangate45_smoke_v001"
CLASSICAL_DIR = ROOT_DIR / "experiments" / "classical" / "uart_nangate45_smoke_v001"


class TestStage4BenchmarkReporting(unittest.TestCase):
    """Test suite for Stage 4 benchmark reporting module."""

    def test_load_stage4_artifacts_on_real_repo(self) -> None:
        """Verify loading existing Stage 4 artifacts from the repo."""
        if not DATASET_DIR.is_dir() or not HEURISTICS_DIR.is_dir() or not CLASSICAL_DIR.is_dir():
            self.skipTest("Real experiment directories not found in repository.")

        artifacts = load_stage4_artifacts(
            dataset_dir=DATASET_DIR,
            heuristics_dir=HEURISTICS_DIR,
            classical_dir=CLASSICAL_DIR,
        )

        self.assertIsNotNone(artifacts.get("manifest"))
        self.assertIsNotNone(artifacts.get("statistics"))
        self.assertIsNotNone(artifacts.get("duplicate_report"))
        self.assertIsNotNone(artifacts.get("splits"))
        self.assertGreater(len(artifacts.get("heuristic_metrics_by_split", [])), 0)
        self.assertGreater(len(artifacts.get("classical_metrics_by_split", [])), 0)
        self.assertGreater(len(artifacts.get("feature_importance", [])), 0)
        self.assertGreater(len(artifacts.get("coefficients", [])), 0)

    def test_load_stage4_artifacts_missing_dir(self) -> None:
        """Verify graceful fallback when directories or files are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifacts = load_stage4_artifacts(
                dataset_dir=tmp_path / "nonexistent_dataset",
                heuristics_dir=tmp_path / "nonexistent_heuristics",
                classical_dir=tmp_path / "nonexistent_classical",
            )
            self.assertIsNone(artifacts["manifest"])
            self.assertIsNone(artifacts["statistics"])
            self.assertEqual(artifacts["heuristic_metrics_by_split"], [])
            self.assertEqual(artifacts["classical_metrics_by_split"], [])

    def test_build_stage4_summary_dict_schema(self) -> None:
        """Verify summary dictionary adheres to stage4_smoke_benchmark_v001 schema."""
        if not DATASET_DIR.is_dir() or not HEURISTICS_DIR.is_dir() or not CLASSICAL_DIR.is_dir():
            self.skipTest("Real experiment directories not found in repository.")

        artifacts = load_stage4_artifacts(DATASET_DIR, HEURISTICS_DIR, CLASSICAL_DIR)
        summary = build_stage4_summary_dict(artifacts)

        self.assertEqual(summary.get("schema_version"), "stage4_smoke_benchmark_v001")
        self.assertEqual(summary.get("dataset_id"), "uart_nangate45_smoke_v001")
        self.assertEqual(summary.get("sample_count"), 6)
        self.assertEqual(summary.get("unique_layout_count"), 3)
        self.assertEqual(summary.get("target_channel"), "total_wirelength_density")

        best_heur = summary.get("best_heuristic", {})
        self.assertIn("name", best_heur)
        self.assertIn("mae", best_heur)
        self.assertIn("recall_5pct", best_heur)

        best_mae = summary.get("best_classical_by_mae", {})
        self.assertIn("name", best_mae)
        self.assertIn("mae", best_mae)
        self.assertLess(best_mae["mae"], 0.02)

        best_rec = summary.get("best_classical_by_recall_at_5", {})
        self.assertIn("name", best_rec)
        self.assertIn("recall_5pct", best_rec)
        self.assertGreater(best_rec["recall_5pct"], 0.8)

        top_feats = summary.get("top_feature_importances", [])
        self.assertGreater(len(top_feats), 0)
        self.assertEqual(top_feats[0]["feature_name"], "scan/scan_chain_segment_density")
        self.assertEqual(top_feats[1]["feature_name"], "physical/pin_density")

        limitations = summary.get("limitations", [])
        self.assertGreater(len(limitations), 4)
        self.assertIn("Stage 5a", summary.get("recommended_next_stage", ""))

    def test_render_stage4_benchmark_markdown_all_sections(self) -> None:
        """Verify markdown report renders all 12 required sections and key elements."""
        if not DATASET_DIR.is_dir() or not HEURISTICS_DIR.is_dir() or not CLASSICAL_DIR.is_dir():
            self.skipTest("Real experiment directories not found in repository.")

        artifacts = load_stage4_artifacts(DATASET_DIR, HEURISTICS_DIR, CLASSICAL_DIR)
        summary = build_stage4_summary_dict(artifacts)
        md = render_stage4_benchmark_markdown(artifacts, summary)

        expected_sections = [
            "## 1. Executive Summary",
            "## 2. Stage 4 Scope and Non-Goals",
            "## 3. Dataset Overview",
            "## 4. Dataset Duplicate Analysis",
            "## 5. Split Policy",
            "## 6. Heuristic Baseline Results",
            "## 7. Classical ML Baseline Results",
            "## 8. Feature Importance and DFT-Aware Signal",
            "## 9. Artifacts Produced",
            "## 10. Limitations",
            "## 11. Readiness Review",
            "## 12. Recommended Next Stage",
        ]
        for sec in expected_sections:
            self.assertIn(sec, md, f"Missing section in report: {sec}")

        # Check key content elements
        self.assertIn("End-to-End Smoke Benchmark Only", md)
        self.assertIn("uart_tx", md)
        self.assertIn("nangate45", md)
        self.assertIn("total_wirelength_density", md)
        self.assertIn("random_forest_regressor", md)
        self.assertIn("simple_weighted_sum", md)
        self.assertIn("scan/scan_chain_segment_density", md)
        self.assertIn("Stage 5a: Expanded Dataset v002", md)

    def test_generate_stage4_benchmark_report_to_tempdir(self) -> None:
        """Verify end-to-end report and summary file generation in temporary directory."""
        if not DATASET_DIR.is_dir() or not HEURISTICS_DIR.is_dir() or not CLASSICAL_DIR.is_dir():
            self.skipTest("Real experiment directories not found in repository.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_md = Path(tmp_dir) / "reports" / "test_report.md"
            out_json = Path(tmp_dir) / "reports" / "test_summary.json"

            generated_md, generated_json = generate_stage4_benchmark_report(
                dataset_dir=DATASET_DIR,
                heuristics_dir=HEURISTICS_DIR,
                classical_dir=CLASSICAL_DIR,
                output_md_path=out_md,
                summary_json_path=out_json,
            )

            self.assertTrue(generated_md.is_file())
            self.assertTrue(generated_json.is_file())

            content = generated_md.read_text(encoding="utf-8")
            self.assertIn("# Stage 4 Smoke Benchmark Report and Readiness Review", content)

            with open(generated_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data.get("schema_version"), "stage4_smoke_benchmark_v001")
            self.assertEqual(data.get("sample_count"), 6)
            self.assertEqual(data.get("unique_layout_count"), 3)


if __name__ == "__main__":
    unittest.main()
