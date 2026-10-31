"""
Unit Tests for Stage 7a Multimodal CNN + GNN Fusion Architecture.
Verifies FusionNeck, CNNGNNFusionModel, forward passes, autograd backprop, and ablation modes.
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch, Data
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.models.fusion import (
    CNNGNNFusionModel,
    FusionNeck,
)


@unittest.skipUnless(_HAS_TORCH, "PyTorch and torch_geometric required for fusion model tests.")
class TestFusionNeck(unittest.TestCase):
    """Unit tests for FusionNeck cross-modal integration module."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.B, self.H, self.W = 2, 64, 64
        self.f_spatial = torch.randn(self.B, 16, self.H, self.W)
        self.f_graph = torch.randn(self.B, 16, self.H, self.W)

    def test_concat_fusion_mode(self) -> None:
        neck = FusionNeck(spatial_channels=16, graph_channels=16, out_channels=16, fusion_mode="concat")
        out = neck(self.f_spatial, self.f_graph)
        self.assertEqual(out.shape, (self.B, 16, self.H, self.W))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_gated_fusion_mode(self) -> None:
        neck = FusionNeck(spatial_channels=16, graph_channels=16, out_channels=16, fusion_mode="gated")
        out = neck(self.f_spatial, self.f_graph)
        self.assertEqual(out.shape, (self.B, 16, self.H, self.W))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_add_fusion_mode(self) -> None:
        neck = FusionNeck(spatial_channels=16, graph_channels=16, out_channels=16, fusion_mode="add")
        out = neck(self.f_spatial, self.f_graph)
        self.assertEqual(out.shape, (self.B, 16, self.H, self.W))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_invalid_fusion_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            FusionNeck(spatial_channels=16, graph_channels=16, out_channels=16, fusion_mode="invalid_xyz")


@unittest.skipUnless(_HAS_TORCH, "PyTorch and torch_geometric required for fusion model tests.")
class TestCNNGNNFusionModel(unittest.TestCase):
    """Unit tests for CNNGNNFusionModel."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.grid_size = 64

    def _create_mock_graph_data(self, num_nodes: int = 50, num_edges: int = 120) -> Data:
        """Helper creating a synthetic PyG Data sample."""
        x = torch.rand(num_nodes, 10, dtype=torch.float32)
        edge_src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        edge_dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        grid_coords = torch.randint(0, self.grid_size, (num_nodes, 2), dtype=torch.long)
        pos = torch.rand(num_nodes, 2, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, grid_coords=grid_coords, pos=pos)
        data.num_nodes = num_nodes
        return data

    def test_model_initialization_and_parameter_accounting(self) -> None:
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            node_in_channels=10,
            gnn_hidden_channels=64,
            gnn_out_channels=64,
            num_gnn_layers=3,
            out_channels=1,
            final_activation="none",
        )
        total_params = model.count_parameters()
        self.assertGreater(total_params, 400_000)
        self.assertLess(total_params, 600_000)

        breakdown = model.count_branch_parameters()
        self.assertIn("spatial_branch", breakdown)
        self.assertIn("gnn_branch", breakdown)
        self.assertIn("fusion_neck", breakdown)
        self.assertIn("decoder", breakdown)
        self.assertEqual(
            total_params,
            breakdown["spatial_branch"]
            + breakdown["gnn_branch"]
            + breakdown["fusion_neck"]
            + breakdown["decoder"],
        )

        summary = model.summary()
        self.assertEqual(summary["spatial_in_channels"], 12)
        self.assertEqual(summary["base_channels"], 16)
        self.assertEqual(summary["node_in_channels"], 10)
        self.assertEqual(summary["out_channels"], 1)

    def test_single_sample_forward_pass(self) -> None:
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=1,
            final_activation="none",
        )
        model.eval()

        x_spatial = torch.randn(1, 12, 64, 64)
        graph_data = self._create_mock_graph_data(num_nodes=40)

        with torch.no_grad():
            out = model(x_spatial=x_spatial, graph_data=graph_data)

        self.assertEqual(out.shape, (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_batched_forward_pass(self) -> None:
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=1,
            final_activation="none",
        )
        model.eval()

        B = 3
        x_spatial = torch.randn(B, 12, 64, 64)
        graphs = [self._create_mock_graph_data(num_nodes=30 + i * 10) for i in range(B)]
        batch_graph = Batch.from_data_list(graphs)

        with torch.no_grad():
            out = model(x_spatial=x_spatial, graph_data=batch_graph)

        self.assertEqual(out.shape, (B, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_multitarget_forward_pass(self) -> None:
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=8,
            final_activation="none",
        )
        model.eval()

        B = 2
        x_spatial = torch.randn(B, 12, 64, 64)
        graphs = [self._create_mock_graph_data(num_nodes=45) for _ in range(B)]
        batch_graph = Batch.from_data_list(graphs)

        with torch.no_grad():
            out = model(x_spatial=x_spatial, graph_data=batch_graph)

        self.assertEqual(out.shape, (B, 8, 64, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_autograd_backpropagation(self) -> None:
        """Verifies that gradients flow back into every architectural component."""
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=1,
            final_activation="none",
        )
        model.train()

        B = 2
        x_spatial = torch.randn(B, 12, 64, 64, requires_grad=True)
        graphs = [self._create_mock_graph_data(num_nodes=35) for _ in range(B)]
        batch_graph = Batch.from_data_list(graphs)

        out = model(x_spatial=x_spatial, graph_data=batch_graph)
        target = torch.rand(B, 1, 64, 64)

        loss = nn.functional.mse_loss(out, target)
        loss.backward()

        # Check gradients in spatial branch
        for p in model.spatial_inc.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.abs().sum().item(), 0.0)

        # Check gradients in GNN branch
        for p in model.gnn_encoder.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.abs().sum().item(), 0.0)

        # Check gradients in fusion neck
        for p in model.fusion_neck.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.abs().sum().item(), 0.0)

        # Check gradients in U-Net decoder
        for p in model.down1.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.abs().sum().item(), 0.0)

        for p in model.outc.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.abs().sum().item(), 0.0)

    def test_ablation_spatial_only_mode(self) -> None:
        """Verifies model forward pass when graph branch is disabled."""
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=1,
            final_activation="none",
        )
        model.eval()

        x_spatial = torch.randn(2, 12, 64, 64)
        with torch.no_grad():
            out = model(x_spatial=x_spatial, graph_data=None, mode="spatial_only")

        self.assertEqual(out.shape, (2, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_ablation_graph_only_mode(self) -> None:
        """Verifies model forward pass when spatial CNN branch is disabled."""
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=1,
            final_activation="none",
        )
        model.eval()

        graphs = [self._create_mock_graph_data(num_nodes=30) for _ in range(2)]
        batch_graph = Batch.from_data_list(graphs)

        with torch.no_grad():
            out = model(x_spatial=None, graph_data=batch_graph, mode="graph_only")

        self.assertEqual(out.shape, (2, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_physical_only_spatial_channels(self) -> None:
        """Verifies model when spatial inputs contain only 7 physical channels (no scan channels)."""
        model = CNNGNNFusionModel(
            spatial_in_channels=7,
            base_channels=16,
            out_channels=1,
        )
        model.eval()

        x_spatial = torch.randn(1, 7, 64, 64)
        graph_data = self._create_mock_graph_data(num_nodes=40)

        with torch.no_grad():
            out = model(x_spatial=x_spatial, graph_data=graph_data)

        self.assertEqual(out.shape, (1, 1, 64, 64))

    def test_direct_tensor_keyword_inputs(self) -> None:
        """Verifies forward pass without a PyG Data container."""
        model = CNNGNNFusionModel(
            spatial_in_channels=12,
            base_channels=16,
            out_channels=1,
        )
        model.eval()

        x_spatial = torch.randn(1, 12, 64, 64)
        x_graph = torch.randn(30, 10)
        edge_index = torch.randint(0, 30, (2, 60), dtype=torch.long)
        grid_coords = torch.randint(0, 64, (30, 2), dtype=torch.long)

        with torch.no_grad():
            out = model(
                x_spatial=x_spatial,
                x_graph=x_graph,
                edge_index=edge_index,
                grid_coords=grid_coords,
            )

        self.assertEqual(out.shape, (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
