"""
Unit tests for Stage 6c (GraphSAGE) and Stage 6d (Graph-to-Grid Pooling).
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
    from torch_geometric.data import Batch, Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from dft_hotspot.graphs.dataset import create_pyg_dataloaders
from dft_hotspot.graphs.pooling import GraphToGridPooling
from dft_hotspot.ml.train_gnn import evaluate_gnn_model, train_gnn, train_gnn_epoch
from dft_hotspot.models.gnn import (
    GNNHotspotPredictor,
    GraphSAGEEncoder,
    GraphSAGELayer,
    SpatialConvHead,
)


@unittest.skipUnless(_HAS_PYG, "PyTorch and PyTorch Geometric required for GNN tests")
class TestGraphToGridPooling(unittest.TestCase):
    """Tests for Stage 6d GraphToGridPooling layer."""

    def test_single_graph_pooling(self):
        pooler = GraphToGridPooling(grid_size=64, aggr="mean", include_density=True)
        self.assertEqual(pooler.get_out_channels(16), 17)

        # 4 nodes
        x = torch.randn(4, 16, requires_grad=True)
        coords = torch.tensor([[10, 20], [10, 20], [30, 40], [50, 60]], dtype=torch.long)

        out = pooler(x, coords)  # B=1 implicitly
        self.assertEqual(out.shape, (1, 17, 64, 64))

        # Check that nodes 0 and 1 are averaged at (10, 20)
        expected_mean = (x[0] + x[1]) / 2.0
        torch.testing.assert_close(out[0, :16, 10, 20], expected_mean)

        # Check density at (10, 20) has count 2 / 10 = 0.2
        self.assertAlmostEqual(out[0, 16, 10, 20].item(), 0.2, places=4)

        # Autograd check
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_batched_pooling(self):
        pooler = GraphToGridPooling(grid_size=(32, 32), aggr="sum", include_density=False)
        self.assertEqual(pooler.get_out_channels(8), 8)

        # Batch of 2 graphs: Graph 0 has 2 nodes, Graph 1 has 3 nodes
        x = torch.randn(5, 8)
        coords = torch.tensor([[5, 5], [10, 10], [5, 5], [15, 15], [20, 20]], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

        out = pooler(x, coords, batch=batch)
        self.assertEqual(out.shape, (2, 8, 32, 32))

        # In Graph 0, node 0 is at (5, 5)
        torch.testing.assert_close(out[0, :, 5, 5], x[0])
        # In Graph 1, node 2 is at (5, 5)
        torch.testing.assert_close(out[1, :, 5, 5], x[2])

    def test_cat_mean_sum_aggregation(self):
        pooler = GraphToGridPooling(grid_size=16, aggr="cat_mean_sum", include_density=True)
        self.assertEqual(pooler.get_out_channels(10), 21)  # 10*2 + 1 = 21

        x = torch.randn(3, 10)
        coords = torch.tensor([[1, 1], [1, 1], [2, 2]], dtype=torch.long)

        out = pooler(x, coords)
        self.assertEqual(out.shape, (1, 21, 16, 16))

    def test_empty_graph_handling(self):
        pooler = GraphToGridPooling(grid_size=64, in_channels=16)
        x = torch.empty((0, 16))
        coords = torch.empty((0, 2), dtype=torch.long)

        out = pooler(x, coords, batch_size=2)
        self.assertEqual(out.shape, (2, 17, 64, 64))
        self.assertEqual(out.sum().item(), 0.0)

    def test_coordinate_clamping_safety(self):
        pooler = GraphToGridPooling(grid_size=32)
        x = torch.randn(2, 4)
        # Coordinates outside [0, 31]
        coords = torch.tensor([[-5, 50], [100, -10]], dtype=torch.long)

        # Should not raise IndexError
        out = pooler(x, coords)
        self.assertEqual(out.shape, (1, 5, 32, 32))


@unittest.skipUnless(_HAS_PYG, "PyTorch and PyTorch Geometric required for GNN tests")
class TestGraphSAGEArchitecture(unittest.TestCase):
    """Tests for Stage 6c GraphSAGE layers and encoder."""

    def test_layer_forward_and_residual(self):
        layer = GraphSAGELayer(in_channels=10, out_channels=32, residual=True)
        x = torch.randn(6, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)

        out = layer(x, edge_index)
        self.assertEqual(out.shape, (6, 32))

        # Check residual when in_channels == out_channels
        layer_same = GraphSAGELayer(in_channels=32, out_channels=32, residual=True)
        out_same = layer_same(out, edge_index)
        self.assertEqual(out_same.shape, (6, 32))

    def test_encoder_multilayers(self):
        encoder = GraphSAGEEncoder(
            in_channels=10,
            hidden_channels=32,
            out_channels=64,
            num_layers=3,
            aggr="mean",
            norm="layer",
        )
        x = torch.randn(8, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out = encoder(x, edge_index)
        self.assertEqual(out.shape, (8, 64))


@unittest.skipUnless(_HAS_PYG, "PyTorch and PyTorch Geometric required for GNN tests")
class TestGNNHotspotPredictor(unittest.TestCase):
    """Tests for Stage 6c GNNHotspotPredictor end-to-end model."""

    def setUp(self):
        self.model = GNNHotspotPredictor(
            node_in_channels=10,
            gnn_hidden_channels=32,
            gnn_out_channels=32,
            num_gnn_layers=2,
            grid_size=64,
            pooling_aggr="mean",
            include_density=True,
            head_hidden_channels=16,
            out_channels=1,
            final_activation="none",
        )

    def test_parameter_count(self):
        param_count = self.model.count_parameters()
        self.assertGreater(param_count, 1000)
        self.assertLess(param_count, 100_000)

    def test_forward_with_pyg_data(self):
        x = torch.randn(12, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
        coords = torch.randint(0, 64, (12, 2), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, grid_coords=coords)
        out = self.model(data)
        self.assertEqual(out.shape, (1, 1, 64, 64))

    def test_forward_and_backward_with_batch(self):
        # 2 graphs in a Batch
        d1 = Data(
            x=torch.randn(10, 10),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            grid_coords=torch.randint(0, 64, (10, 2), dtype=torch.long),
        )
        d2 = Data(
            x=torch.randn(8, 10),
            edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
            grid_coords=torch.randint(0, 64, (8, 2), dtype=torch.long),
        )
        batch = Batch.from_data_list([d1, d2])

        pred = self.model(batch)
        self.assertEqual(pred.shape, (2, 1, 64, 64))

        # Test loss and backward pass
        target = torch.randn(2, 1, 64, 64)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        # Check gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for parameter {name}")

    def test_multi_target_predictor(self):
        multi_model = GNNHotspotPredictor(
            node_in_channels=10,
            gnn_hidden_channels=32,
            gnn_out_channels=32,
            out_channels=8,
            final_activation="none",
        )
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        coords = torch.randint(0, 64, (10, 2), dtype=torch.long)

        out = multi_model(Data(x=x, edge_index=edge_index, grid_coords=coords))
        self.assertEqual(out.shape, (1, 8, 64, 64))

    def test_real_dataset_loader_integration(self):
        """Validates forward pass through GNNHotspotPredictor using real batches from expanded_v002."""
        dataset_dir = Path("data/processed/datasets/expanded_v002")
        if not dataset_dir.is_dir():
            self.skipTest("expanded_v002 dataset not found")

        train_loader, _, _, _ = create_pyg_dataloaders(
            dataset_dir=dataset_dir,
            target_channel="total_wirelength_density",
            batch_size=2,
            preload=True,
        )

        batch = next(iter(train_loader))
        out = self.model(batch)
        self.assertEqual(out.shape, (2, 1, 64, 64))
        self.assertEqual(batch.y.shape, (2, 1, 64, 64))


@unittest.skipUnless(_HAS_PYG, "PyTorch and PyTorch Geometric required for GNN tests")
class TestGNNTrainingEngine(unittest.TestCase):
    """Tests for train_gnn, train_gnn_epoch, and evaluate_gnn_model."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = GNNHotspotPredictor(
            node_in_channels=10,
            gnn_hidden_channels=16,
            gnn_out_channels=16,
            num_gnn_layers=2,
            grid_size=16,
            head_hidden_channels=8,
            out_channels=1,
            final_activation="none",
        ).to(self.device)

        from torch_geometric.loader import DataLoader as PyGDataLoader
        from dft_hotspot.graphs.schema import CircuitGraphData

        # 4 synthetic graphs
        data_list = []
        for i in range(4):
            d = CircuitGraphData(
                x=torch.randn(8, 10),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
                edge_attr=torch.randn(3, 4),
                pos=torch.rand(8, 2),
                grid_coords=torch.randint(0, 16, (8, 2), dtype=torch.long),
                y=torch.rand(1, 16, 16),
                sample_id=f"synth_{i}",
                design="synth",
            )
            data_list.append(d)

        self.loader = PyGDataLoader(data_list, batch_size=2)

    def test_train_epoch_and_eval(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        loss = train_gnn_epoch(self.model, self.loader, optimizer, criterion, self.device)
        self.assertGreater(loss, 0.0)

        eval_loss, metrics, preds = evaluate_gnn_model(
            self.model, self.loader, criterion, self.device, clamp_output=True
        )
        self.assertGreater(eval_loss, 0.0)
        self.assertIn("mae", metrics)
        self.assertEqual(len(preds), 4)

    def test_train_gnn_full_loop(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            summary = train_gnn(
                model=self.model,
                train_loader=self.loader,
                val_loader=self.loader,
                output_dir=out_dir,
                device=self.device,
                max_epochs=2,
                early_stopping_patience=2,
            )
            self.assertEqual(summary["total_epochs"], 2)
            self.assertTrue((out_dir / "best_model.pt").is_file())
            self.assertTrue((out_dir / "last_model.pt").is_file())
            self.assertTrue((out_dir / "training_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
