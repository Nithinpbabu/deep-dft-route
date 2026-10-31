"""
Multimodal CNN + GNN Fusion Architecture for Physical Design Hotspot Prediction.
Stage 7a of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- FusionNeck: Multi-modal fusion block supporting concatenation, gated attention, and additive fusion.
- CNNGNNFusionModel: Deep multimodal neural network unifying:
    1. Spatial Layout CNN Branch: Multi-scale 2D convolutions over physical & scan density grids.
    2. Circuit Graph GNN Branch: Inductive GraphSAGE message passing over netlist & scan topology.
    3. Graph-to-Grid Spatial Readout: Differentiable coordinate scatter onto the physical die grid.
    4. Cross-Modal Fusion Neck: Seamless feature integration at native physical resolution.
    5. Hierarchical Multi-Scale Decoder: U-Net decoder with skip connections for high-fidelity dense predictions.
- Supports single-target (total_wirelength_density) and multi-target (all 8 PDK routing channels).
- Supports runtime ablation modes: 'multimodal' (full model), 'spatial_only' (CNN only), and 'graph_only' (GNN only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False
    nn = object  # type: ignore

from dft_hotspot.graphs.pooling import GraphToGridPooling
from dft_hotspot.ml.unet import DoubleConv, Down, Up
from dft_hotspot.models.gnn import GraphSAGEEncoder


class FusionNeck(nn.Module):
    """
    Cross-modal fusion neck integrating spatial CNN representations and graph-to-grid representations.

    Supported fusion modes:
    - 'concat': Channel concatenation followed by 2-stage conv refinement.
    - 'gated': Learnable spatial attention gate weighting graph vs spatial representations.
    - 'add': Linear projection of both modalities into a shared subspace followed by elementwise addition.
    """

    def __init__(
        self,
        spatial_channels: int,
        graph_channels: int,
        out_channels: int,
        fusion_mode: str = "concat",
        dropout: float = 0.0,
    ) -> None:
        if not _HAS_DEPS:
            raise ImportError("PyTorch and torch_geometric are required for FusionNeck.")

        super().__init__()
        self.spatial_channels = spatial_channels
        self.graph_channels = graph_channels
        self.out_channels = out_channels
        self.fusion_mode = fusion_mode.lower()

        if self.fusion_mode == "concat":
            in_ch = spatial_channels + graph_channels
            self.fusion_block = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif self.fusion_mode == "gated":
            in_ch = spatial_channels + graph_channels
            self.proj_spatial = (
                nn.Conv2d(spatial_channels, out_channels, kernel_size=1, bias=False)
                if spatial_channels != out_channels
                else nn.Identity()
            )
            self.proj_graph = (
                nn.Conv2d(graph_channels, out_channels, kernel_size=1, bias=False)
                if graph_channels != out_channels
                else nn.Identity()
            )
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.refine = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif self.fusion_mode == "add":
            self.proj_spatial = (
                nn.Conv2d(spatial_channels, out_channels, kernel_size=1, bias=False)
                if spatial_channels != out_channels
                else nn.Identity()
            )
            self.proj_graph = (
                nn.Conv2d(graph_channels, out_channels, kernel_size=1, bias=False)
                if graph_channels != out_channels
                else nn.Identity()
            )
            self.refine = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(
                f"Unknown fusion_mode '{fusion_mode}'. Expected 'concat', 'gated', or 'add'."
            )

    def forward(self, f_spatial: torch.Tensor, f_graph: torch.Tensor) -> torch.Tensor:
        """
        Merges spatial and graph feature maps into a unified representation.

        Args:
            f_spatial: (B, C_spatial, H, W)
            f_graph: (B, C_graph, H, W)
        Returns:
            f_fused: (B, C_out, H, W)
        """
        if self.fusion_mode == "concat":
            cat = torch.cat([f_spatial, f_graph], dim=1)
            return self.fusion_block(cat)

        elif self.fusion_mode == "gated":
            cat = torch.cat([f_spatial, f_graph], dim=1)
            gate = self.gate_conv(cat)
            s_proj = self.proj_spatial(f_spatial)
            g_proj = self.proj_graph(f_graph)
            gated = gate * g_proj + (1.0 - gate) * s_proj
            return self.refine(gated)

        elif self.fusion_mode == "add":
            s_proj = self.proj_spatial(f_spatial)
            g_proj = self.proj_graph(f_graph)
            added = s_proj + g_proj
            return self.refine(added)

        raise RuntimeError(f"Unhandled fusion mode: {self.fusion_mode}")


class CNNGNNFusionModel(nn.Module):
    """
    Multimodal CNN + GNN Hotspot Prediction Model (Stage 7a).

    Unifies 2D continuous placement features and graph netlist / scan topology
    into a joint representation for dense routing congestion and wirelength prediction.

    Pipeline:
        1. Spatial CNN:
           x_spatial (B, C_spatial, H, W) -> DoubleConv -> f_spatial (B, base_channels, H, W)
        2. Circuit GNN:
           (x_graph, edge_index) -> GraphSAGEEncoder -> H_nodes (N, D_gnn)
           H_nodes + grid_coords -> GraphToGridPooling -> grid_map (B, D_gnn + 1, H, W)
           grid_map -> Conv2d -> f_graph (B, base_channels, H, W)
        3. Cross-Modal Fusion Neck:
           (f_spatial, f_graph) -> FusionNeck -> f_0 (B, base_channels, H, W)
        4. Hierarchical Multi-Scale Decoder (U-Net):
           f_0 -> Down1 -> Down2 -> Down3 (Bottleneck) -> Up1 -> Up2 -> Up3 -> Conv1x1 -> y_pred
    """

    def __init__(
        self,
        # Spatial CNN branch config
        spatial_in_channels: int = 12,
        base_channels: int = 16,
        # Graph GNN branch config
        node_in_channels: int = 10,
        gnn_hidden_channels: int = 64,
        gnn_out_channels: int = 64,
        num_gnn_layers: int = 3,
        gnn_aggr: str = "mean",
        gnn_dropout: float = 0.1,
        # Graph pooling config
        grid_size: Union[int, Tuple[int, int]] = 64,
        pooling_aggr: str = "mean",
        include_density: bool = True,
        # Fusion neck config
        fusion_mode: str = "concat",
        fusion_dropout: float = 0.0,
        # Decoder & output config
        out_channels: int = 1,
        final_activation: str = "none",
        bilinear: bool = False,
        # Operational mode for ablation
        default_mode: str = "multimodal",
    ) -> None:
        if not _HAS_DEPS:
            raise ImportError(
                "PyTorch and torch_geometric are required for CNNGNNFusionModel."
            )

        super().__init__()
        self.spatial_in_channels = spatial_in_channels
        self.base_channels = base_channels
        self.node_in_channels = node_in_channels
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_out_channels = gnn_out_channels
        self.num_gnn_layers = num_gnn_layers
        self.grid_size = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        self.out_channels = out_channels
        self.final_activation = final_activation.lower() if final_activation else "none"
        self.default_mode = default_mode.lower()
        self.fusion_mode = fusion_mode.lower()

        # ---------------------------------------------------------------------
        # 1. Spatial CNN Branch
        # ---------------------------------------------------------------------
        self.spatial_inc = DoubleConv(spatial_in_channels, base_channels)

        # ---------------------------------------------------------------------
        # 2. Graph GNN Branch
        # ---------------------------------------------------------------------
        self.gnn_encoder = GraphSAGEEncoder(
            in_channels=node_in_channels,
            hidden_channels=gnn_hidden_channels,
            out_channels=gnn_out_channels,
            num_layers=num_gnn_layers,
            aggr=gnn_aggr,
            dropout=gnn_dropout,
            residual=True,
        )

        self.graph_pool = GraphToGridPooling(
            grid_size=self.grid_size,
            aggr=pooling_aggr,
            include_density=include_density,
        )

        # Output channels from pooling: gnn_out_channels (+1 if density included)
        pool_channels = gnn_out_channels + (1 if include_density else 0)
        self.graph_proj = nn.Sequential(
            nn.Conv2d(pool_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # 3. Cross-Modal Fusion Neck
        # ---------------------------------------------------------------------
        self.fusion_neck = FusionNeck(
            spatial_channels=base_channels,
            graph_channels=base_channels,
            out_channels=base_channels,
            fusion_mode=fusion_mode,
            dropout=fusion_dropout,
        )

        # ---------------------------------------------------------------------
        # 4. Multi-Scale Hierarchical Decoder (U-Net)
        # ---------------------------------------------------------------------
        factor = 2 if bilinear else 1
        self.down1 = Down(base_channels, base_channels * 2)                  # 16 -> 32  (32x32)
        self.down2 = Down(base_channels * 2, base_channels * 4)              # 32 -> 64  (16x16)
        self.down3 = Down(base_channels * 4, (base_channels * 8) // factor)  # 64 -> 128 (8x8)

        self.up1 = Up(base_channels * 8, (base_channels * 4) // factor, bilinear)  # 128 -> 64
        self.up2 = Up(base_channels * 4, (base_channels * 2) // factor, bilinear)  # 64 -> 32
        self.up3 = Up(base_channels * 2, base_channels, bilinear)            # 32 -> 16

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Output activation
        if self.final_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.final_activation == "relu":
            self.act = nn.ReLU()
        elif self.final_activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(
                f"Unknown final_activation '{final_activation}'. Expected 'sigmoid', 'relu', or 'none'."
            )

    def extract_spatial_features(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """Runs the spatial CNN branch, returning (B, base_channels, H, W)."""
        return self.spatial_inc(x_spatial)

    def extract_graph_features(
        self,
        x_graph: torch.Tensor,
        edge_index: torch.Tensor,
        grid_coords: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs the circuit graph branch, returning (B, base_channels, H, W).
        """
        if grid_coords is None:
            if pos is not None:
                row = (pos[:, 1] * self.grid_size[0]).long().clamp(0, self.grid_size[0] - 1)
                col = (pos[:, 0] * self.grid_size[1]).long().clamp(0, self.grid_size[1] - 1)
                grid_coords = torch.stack([row, col], dim=-1)
            else:
                raise ValueError("Either grid_coords or pos must be provided for the graph branch.")

        h_nodes = self.gnn_encoder(x_graph, edge_index)
        grid_map = self.graph_pool(
            x=h_nodes,
            grid_coords=grid_coords,
            batch=batch_idx,
            batch_size=batch_size,
        )
        return self.graph_proj(grid_map)

    def forward(
        self,
        x_spatial: Optional[torch.Tensor] = None,
        graph_data: Optional[Union[Batch, Data, Dict[str, Any]]] = None,
        *,
        # Direct keyword overrides if graph_data is not provided as a PyG object
        x_graph: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        grid_coords: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multimodal prediction.

        Args:
            x_spatial: (B, C_spatial, H, W) spatial feature grid.
            graph_data: PyG Data or Batch instance containing x, edge_index, grid_coords/pos.
            x_graph: Optional direct node features (N, D_node).
            edge_index: Optional direct edge indices (2, E).
            grid_coords: Optional direct grid coordinates (N, 2).
            pos: Optional direct normalized coordinates (N, 2).
            batch_idx: Optional direct batch assignment vector (N,).
            mode: Operational mode ('multimodal', 'spatial_only', or 'graph_only').
                  Defaults to self.default_mode.

        Returns:
            y_pred: (B, C_target, H, W) continuous routing density prediction.
        """
        active_mode = (mode or self.default_mode).lower()

        # Determine batch size
        if x_spatial is not None:
            batch_size = x_spatial.size(0)
        elif graph_data is not None and hasattr(graph_data, "num_graphs"):
            batch_size = graph_data.num_graphs
        elif batch_idx is not None and batch_idx.numel() > 0:
            batch_size = int(batch_idx.max().item()) + 1
        else:
            batch_size = 1

        # -----------------------------------------------------------------
        # Spatial-Only Mode (Ablation)
        # -----------------------------------------------------------------
        if active_mode == "spatial_only":
            if x_spatial is None:
                raise ValueError("x_spatial must be provided when mode='spatial_only'.")
            f_0 = self.extract_spatial_features(x_spatial)

        # -----------------------------------------------------------------
        # Graph-Only Mode (Ablation)
        # -----------------------------------------------------------------
        elif active_mode == "graph_only":
            f_0 = self._run_graph_branch(
                graph_data=graph_data,
                x_graph=x_graph,
                edge_index=edge_index,
                grid_coords=grid_coords,
                pos=pos,
                batch_idx=batch_idx,
                batch_size=batch_size,
            )

        # -----------------------------------------------------------------
        # Multimodal Fusion Mode (Full Model)
        # -----------------------------------------------------------------
        elif active_mode == "multimodal":
            if x_spatial is None:
                raise ValueError("x_spatial must be provided when mode='multimodal'.")

            f_spatial = self.extract_spatial_features(x_spatial)
            f_graph = self._run_graph_branch(
                graph_data=graph_data,
                x_graph=x_graph,
                edge_index=edge_index,
                grid_coords=grid_coords,
                pos=pos,
                batch_idx=batch_idx,
                batch_size=batch_size,
            )
            f_0 = self.fusion_neck(f_spatial, f_graph)

        else:
            raise ValueError(
                f"Unknown mode '{active_mode}'. Expected 'multimodal', 'spatial_only', or 'graph_only'."
            )

        # -----------------------------------------------------------------
        # Multi-Scale U-Net Hierarchy
        # -----------------------------------------------------------------
        x1 = f_0
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return self.act(logits)

    def _run_graph_branch(
        self,
        graph_data: Optional[Union[Batch, Data, Dict[str, Any]]],
        x_graph: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        grid_coords: Optional[torch.Tensor],
        pos: Optional[torch.Tensor],
        batch_idx: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Internal helper resolving and executing the graph branch."""
        if graph_data is not None:
            if isinstance(graph_data, dict):
                gx = graph_data.get("x")
                gei = graph_data.get("edge_index")
                ggc = graph_data.get("grid_coords")
                gpos = graph_data.get("pos")
                gbatch = graph_data.get("batch")
            else:
                gx = getattr(graph_data, "x", None)
                gei = getattr(graph_data, "edge_index", None)
                ggc = getattr(graph_data, "grid_coords", None)
                gpos = getattr(graph_data, "pos", None)
                gbatch = getattr(graph_data, "batch", None)

            if gx is None or gei is None:
                raise ValueError("graph_data must contain 'x' and 'edge_index'.")
        else:
            gx, gei, ggc, gpos, gbatch = x_graph, edge_index, grid_coords, pos, batch_idx

        if gx is None or gei is None:
            raise ValueError(
                "Either graph_data or both (x_graph, edge_index) must be supplied."
            )

        if gbatch is None:
            gbatch = torch.zeros(gx.size(0), dtype=torch.long, device=gx.device)

        return self.extract_graph_features(
            x_graph=gx,
            edge_index=gei,
            grid_coords=ggc,
            pos=gpos,
            batch_idx=gbatch,
            batch_size=batch_size,
        )

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_branch_parameters(self) -> Dict[str, int]:
        """Returns a breakdown of trainable parameters across architectural branches."""
        def _count(mod: Optional[nn.Module]) -> int:
            if mod is None:
                return 0
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)

        spatial_params = _count(self.spatial_inc)
        gnn_params = _count(self.gnn_encoder) + _count(self.graph_proj)
        fusion_params = _count(self.fusion_neck)
        decoder_params = (
            _count(self.down1)
            + _count(self.down2)
            + _count(self.down3)
            + _count(self.up1)
            + _count(self.up2)
            + _count(self.up3)
            + _count(self.outc)
        )
        return {
            "spatial_branch": spatial_params,
            "gnn_branch": gnn_params,
            "fusion_neck": fusion_params,
            "decoder": decoder_params,
            "total": self.count_parameters(),
        }

    def summary(self) -> Dict[str, Any]:
        """Returns structured metadata describing the model architecture."""
        return {
            "spatial_in_channels": self.spatial_in_channels,
            "base_channels": self.base_channels,
            "node_in_channels": self.node_in_channels,
            "gnn_hidden_channels": self.gnn_hidden_channels,
            "gnn_out_channels": self.gnn_out_channels,
            "num_gnn_layers": self.num_gnn_layers,
            "fusion_mode": self.fusion_mode,
            "out_channels": self.out_channels,
            "final_activation": self.final_activation,
            "default_mode": self.default_mode,
            "parameter_breakdown": self.count_branch_parameters(),
        }
