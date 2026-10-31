"""
GraphSAGE and GNN Hotspot Predictor Architectures (Stage 6c).

Provides:
- GraphSAGELayer: SAGEConv + Normalization + Non-Linearity + Residual + Dropout.
- GraphSAGEEncoder: Inductive multi-layer GraphSAGE message passing over netlist & scan topology.
- GNNHotspotPredictor: Standalone end-to-end model coupling GraphSAGE, GraphToGridPooling,
  and a multi-scale convolutional spatial prediction head.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import SAGEConv
    _HAS_GNN_DEPS = True
except ImportError:
    _HAS_GNN_DEPS = False
    nn = object  # type: ignore

from dft_hotspot.graphs.pooling import GraphToGridPooling


class GraphSAGELayer(nn.Module):
    """
    Single GraphSAGE message-passing layer with normalization, activation, residual, and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        norm: str = "layer",
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        if not _HAS_GNN_DEPS:
            raise ImportError("PyTorch and torch_geometric are required for GraphSAGELayer.")

        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels, aggr=aggr)
        self.residual = residual
        self.use_projection = residual and (in_channels != out_channels)

        if self.use_projection:
            self.res_proj = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.res_proj = None

        if norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm '{norm}'. Expected 'layer', 'batch', or 'none'.")

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x, edge_index)
        out = self.norm(out)

        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        out = self.act(out)
        out = self.dropout(out)
        return out


class GraphSAGEEncoder(nn.Module):
    """
    Inductive multi-layer GraphSAGE encoder.

    Maps node physical and scan features X in R^(N x D_in) through L message-passing layers
    into contextual node embeddings H in R^(N x D_out).
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        aggr: str = "mean",
        norm: str = "layer",
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        if not _HAS_GNN_DEPS:
            raise ImportError("PyTorch and torch_geometric are required for GraphSAGEEncoder.")

        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(
                GraphSAGELayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    aggr=aggr,
                    norm=norm,
                    dropout=dropout,
                    residual=residual,
                )
            )
        else:
            # First layer
            self.layers.append(
                GraphSAGELayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    aggr=aggr,
                    norm=norm,
                    dropout=dropout,
                    residual=residual,
                )
            )
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.layers.append(
                    GraphSAGELayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        aggr=aggr,
                        norm=norm,
                        dropout=dropout,
                        residual=residual,
                    )
                )
            # Final layer
            self.layers.append(
                GraphSAGELayer(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    aggr=aggr,
                    norm=norm,
                    dropout=dropout,
                    residual=residual,
                )
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing node embeddings.
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
        Returns:
            Node representations (N, out_channels)
        """
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class SpatialConvHead(nn.Module):
    """
    Convolutional spatial refinement decoder for pooled 2D feature maps.
    Refines pooled (B, C_in, H, W) graph representations into dense target heatmaps (B, C_out, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 32,
        final_activation: str = "none",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation.lower() if final_activation else "none"

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)

        if self.final_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.final_activation == "relu":
            self.act = nn.ReLU()
        elif self.final_activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown final_activation '{final_activation}'. Expected 'sigmoid', 'relu', or 'none'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        logits = self.out_conv(x)
        return self.act(logits)


class GNNHotspotPredictor(nn.Module):
    """
    Standalone GNN model for dense routing hotspot prediction (Stage 6c / 6e).

    End-to-end pipeline:
        1. GraphSAGEEncoder: Message passing over circuit netlist & scan chains.
        2. GraphToGridPooling: Differentiable coordinate scatter onto (64 x 64) layout grid.
        3. SpatialConvHead: Convolutional spatial refinement and dense heatmap prediction.
    """

    def __init__(
        self,
        node_in_channels: int = 10,
        gnn_hidden_channels: int = 64,
        gnn_out_channels: int = 64,
        num_gnn_layers: int = 3,
        gnn_aggr: str = "mean",
        grid_size: Union[int, Tuple[int, int]] = 64,
        pooling_aggr: str = "mean",
        include_density: bool = True,
        head_hidden_channels: int = 32,
        out_channels: int = 1,
        final_activation: str = "none",
        dropout: float = 0.1,
    ) -> None:
        if not _HAS_GNN_DEPS:
            raise ImportError("PyTorch and torch_geometric are required for GNNHotspotPredictor.")

        super().__init__()
        self.node_in_channels = node_in_channels
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_out_channels = gnn_out_channels
        self.out_channels = out_channels

        # 1. GraphSAGE Backbone
        self.encoder = GraphSAGEEncoder(
            in_channels=node_in_channels,
            hidden_channels=gnn_hidden_channels,
            out_channels=gnn_out_channels,
            num_layers=num_gnn_layers,
            aggr=gnn_aggr,
            dropout=dropout,
        )

        # 2. Graph-to-Grid Pooling Layer
        self.pooler = GraphToGridPooling(
            grid_size=grid_size,
            aggr=pooling_aggr,
            include_density=include_density,
            in_channels=gnn_out_channels,
        )

        pool_out_dim = self.pooler.get_out_channels(gnn_out_channels)

        # 3. Spatial Convolutional Refinement Head
        self.head = SpatialConvHead(
            in_channels=pool_out_dim,
            out_channels=out_channels,
            hidden_channels=head_hidden_channels,
            final_activation=final_activation,
        )

    def forward(
        self,
        data_or_x: Union[Data, Batch, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        grid_coords: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting 2D routing density maps from circuit graph.

        Supports either:
            forward(batch_data) where batch_data is a PyG Data or Batch object.
            or
            forward(x, edge_index, grid_coords, batch, batch_size)
        """
        if hasattr(data_or_x, "x"):
            x = data_or_x.x
            e_idx = data_or_x.edge_index
            coords = data_or_x.grid_coords
            b = getattr(data_or_x, "batch", None)
            b_size = getattr(data_or_x, "num_graphs", None)
        else:
            x = data_or_x
            e_idx = edge_index
            coords = grid_coords
            b = batch
            b_size = batch_size

        if e_idx is None or coords is None:
            raise ValueError("edge_index and grid_coords are required for GNNHotspotPredictor forward pass.")

        # 1. Graph Message Passing
        node_emb = self.encoder(x, e_idx)

        # 2. Graph-to-Grid Spatial Scatter
        grid_features = self.pooler(node_emb, coords, batch=b, batch_size=b_size)

        # 3. Convolutional Spatial Refinement
        pred_map = self.head(grid_features)

        return pred_map

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
