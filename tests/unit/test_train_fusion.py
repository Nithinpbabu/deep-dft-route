"""
Unit Tests for Stage 7b Multimodal Fusion Training Engine.
Verifies train_fusion_epoch, evaluate_fusion_model, and train_fusion orchestration loop.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch_geometric.data import Batch, Data
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.graphs.schema import CircuitGraphData
from dft_hotspot.ml.torch_multimodal_dataset import collate_multimodal
from dft_hotspot.ml.train_fusion import (
    evaluate_fusion_model,
    train_fusion,
    train_fusion_epoch,
)
from dft_hotspot.models.fusion import CNNGNNFusionModel


class MockMultimodalDataset(Dataset):
    """Mock in-memory multimodal dataset for fast deterministic unit tests."""

    def __init__(self, num_samples: int = 4, spatial_channels: int = 12, out_channels: int = 1) -> None:
        self.num_samples = num_samples
        self.target_channels = ["total_wirelength_density"] if out_channels == 1 else [f"ch_{i}" for i in range(out_channels)]

        self.items = []
        for i in range(num_samples):
            x_spat = torch.randn(spatial_channels, 64, 64)
            y = torch.rand(out_channels, 64, 64)

            num_nodes = 25
            x_g = torch.randn(num_nodes, 10)
            edge_idx = torch.randint(0, num_nodes, (2, 50), dtype=torch.long)
            coords = torch.randint(0, 64, (num_nodes, 2), dtype=torch.long)
            pos = torch.rand(num_nodes, 2)
            gdata = CircuitGraphData(x=x_g, edge_index=edge_idx, grid_coords=coords, pos=pos)

            self.items.append({
                "x_spatial": x_spat,
                "graph": gdata,
                "y": y,
                "sample_id": f"mock_sample_{i}",
                "design": "mock_design",
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.items[idx]


@unittest.skipUnless(_HAS_TORCH, "PyTorch and torch_geometric required for training tests.")
class TestTrainFusionEngine(unittest.TestCase):
    """Unit tests for train_fusion engine."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.dataset = MockMultimodalDataset(num_samples=4, spatial_channels=12, out_channels=1)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False, collate_fn=collate_multimodal)

        self.model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=8,
            node_in_channels=10,
            gnn_hidden_channels=16,
            gnn_out_channels=16,
            num_gnn_layers=2,
            out_channels=1,
            final_activation="none",
        )

    def test_train_fusion_epoch(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        loss = train_fusion_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_evaluate_fusion_model(self) -> None:
        criterion = nn.MSELoss()
        loss, metrics, preds = evaluate_fusion_model(
            model=self.model,
            dataloader=self.dataloader,
            criterion=criterion,
            device=self.device,
        )
        self.assertIsInstance(loss, float)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("pearson", metrics)
        self.assertIn("top5_recall", metrics)
        self.assertEqual(len(preds), 4)

    def test_train_fusion_orchestration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            train_loader = DataLoader(MockMultimodalDataset(num_samples=4), batch_size=2, collate_fn=collate_multimodal)
            val_loader = DataLoader(MockMultimodalDataset(num_samples=2), batch_size=2, collate_fn=collate_multimodal)
            test_loader = DataLoader(MockMultimodalDataset(num_samples=2), batch_size=2, collate_fn=collate_multimodal)

            results = train_fusion(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=2,
                lr=1e-3,
                device=self.device,
                output_dir=out_dir,
            )

            self.assertIn("best_epoch", results)
            self.assertIn("training_history", results)
            self.assertEqual(len(results["training_history"]), 2)
            self.assertTrue((out_dir / "best_model.pt").is_file())
            self.assertTrue((out_dir / "last_model.pt").is_file())
            self.assertTrue((out_dir / "training_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
