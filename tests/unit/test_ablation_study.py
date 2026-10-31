"""
Unit Tests for Stage 7c Multimodal Ablation Study Suite.
Verifies ablation configuration validity, report generation, and ablation mode execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from torch.utils.data import DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.ml.torch_multimodal_dataset import collate_multimodal
from dft_hotspot.models.fusion import CNNGNNFusionModel
from scripts.run_fusion_ablations import (
    ABLATION_CONFIGS,
    EXTERNAL_BASELINES,
    generate_ablation_report,
)
from tests.unit.test_train_fusion import MockMultimodalDataset


@unittest.skipUnless(_HAS_TORCH, "PyTorch required for ablation study tests.")
class TestAblationStudySuite(unittest.TestCase):
    """Unit tests for Stage 7c ablation configurations and report generation."""

    def test_ablation_configs_structure(self) -> None:
        self.assertEqual(len(ABLATION_CONFIGS), 5)
        for cfg in ABLATION_CONFIGS:
            self.assertIn("name", cfg)
            self.assertIn("tag", cfg)
            self.assertIn("mode", cfg)
            self.assertIn(cfg["mode"], ["multimodal", "spatial_only", "graph_only"])
            self.assertIn("fusion_mode", cfg)
            self.assertIn(cfg["fusion_mode"], ["concat", "gated", "add"])
            self.assertIsInstance(cfg["include_scan"], bool)

    def test_external_baselines_structure(self) -> None:
        self.assertGreaterEqual(len(EXTERNAL_BASELINES), 4)
        for b in EXTERNAL_BASELINES:
            self.assertIn("model_variant", b)
            self.assertIn("test_mae", b)
            self.assertIn("test_rmse", b)
            self.assertIn("iou_5pct", b)

    def test_generate_ablation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir) / "output"
            out_root.mkdir(parents=True, exist_ok=True)
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            mock_results = [
                {
                    "cfg": ABLATION_CONFIGS[0],
                    "tag": ABLATION_CONFIGS[0]["tag"],
                    "name": ABLATION_CONFIGS[0]["name"],
                    "parameters": 518481,
                    "best_epoch": 20,
                    "total_epochs": 25,
                    "training_time_s": 15.5,
                    "best_val_loss": 0.002,
                    "test_loss": 0.0025,
                    "test_metrics": {
                        "mae": 0.009,
                        "rmse": 0.038,
                        "pearson": 0.90,
                        "spearman": 0.60,
                        "top5_recall": 0.85,
                        "iou_5pct": 0.84,
                        "dice_5pct": 0.91,
                    },
                }
            ]

            generate_ablation_report(mock_results, output_root=out_root, reports_dir=reports_dir)

            self.assertTrue((out_root / "ablation_study_summary.csv").is_file())
            self.assertTrue((out_root / "ablation_study_summary.json").is_file())
            self.assertTrue((reports_dir / "stage7c_multimodal_ablation_benchmark.md").is_file())

            with open(out_root / "ablation_study_summary.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("ablations", data)
                self.assertEqual(len(data["ablations"]), 1)

    def test_all_ablation_modes_forward_pass(self) -> None:
        """Runs a forward pass for all 5 ablation configurations using synthetic data."""
        dataset = MockMultimodalDataset(num_samples=2, spatial_channels=12)
        batch = collate_multimodal([dataset[0], dataset[1]])

        for cfg in ABLATION_CONFIGS:
            spatial_ch = 12 if cfg["include_scan"] else 7
            x_spatial = batch["x_spatial"][:, :spatial_ch, :, :]
            model = CNNGNNFusionModel(
                spatial_in_channels=spatial_ch,
                base_channels=8,
                node_in_channels=10,
                gnn_hidden_channels=16,
                gnn_out_channels=16,
                num_gnn_layers=2,
                fusion_mode=cfg["fusion_mode"],
                out_channels=1,
                default_mode=cfg["mode"],
            )
            model.eval()
            with torch.no_grad():
                out = model(x_spatial=x_spatial, graph_data=batch["graph"], mode=cfg["mode"])
            self.assertEqual(out.shape, (2, 1, 64, 64))
            self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
