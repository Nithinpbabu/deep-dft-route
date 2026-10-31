"""
Graph-to-Grid Pooling Layer (Stage 6d).

Differentiably projects node-level graph embeddings onto a 2D spatial layout grid
(B, C_out, H, W) using physical placement coordinates (grid row and column bins).
Supports batch collation, multiple aggregation strategies, and density channel extraction.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # type: ignore


class GraphToGridPooling(nn.Module):
    """
    Spatial scatter and readout layer mapping graph node embeddings onto a 2D physical grid.

    Given:
        - Node representations X of shape (N_total, D)
        - Spatial grid coordinates (row, col) in [0, H-1] x [0, W-1] of shape (N_total, 2)
        - Batch assignment vector of shape (N_total,) mapping each node to a graph in [0, B-1]
    
    Produces:
        - 4D spatial feature map of shape (B, C_out, H, W)
    """

    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]] = 64,
        aggr: str = "mean",
        include_density: bool = True,
        in_channels: Optional[int] = None,
    ) -> None:
        """
        Args:
            grid_size: (H, W) or single integer H=W for grid resolution (default: 64).
            aggr: Aggregation mode across multiple cells in the same grid bin:
                  'mean': Average node feature per bin (0 for unoccupied bins).
                  'sum': Sum of node features per bin.
                  'cat_mean_sum': Concatenates both mean and sum embeddings (2 * D).
            include_density: If True, appends a normalized node count / occupancy channel (+1).
            in_channels: Optional input channel dimension for calculating out_channels.
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for GraphToGridPooling.")

        super().__init__()

        if isinstance(grid_size, int):
            self.grid_h = grid_size
            self.grid_w = grid_size
        else:
            self.grid_h, self.grid_w = grid_size

        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({self.grid_h}, {self.grid_w})")

        valid_aggrs = ("mean", "sum", "cat_mean_sum")
        if aggr.lower() not in valid_aggrs:
            raise ValueError(f"Invalid aggr '{aggr}'. Supported options: {valid_aggrs}")

        self.aggr = aggr.lower()
        self.include_density = include_density
        self.in_channels = in_channels

    def get_out_channels(self, in_channels: Optional[int] = None) -> int:
        """Computes output feature map channel dimension for a given input dimension."""
        cin = in_channels if in_channels is not None else self.in_channels
        if cin is None:
            raise ValueError("in_channels must be provided to determine output channels.")

        if self.aggr in ("mean", "sum"):
            base = cin
        elif self.aggr == "cat_mean_sum":
            base = cin * 2
        else:
            base = cin

        if self.include_density:
            base += 1

        return base

    def forward(
        self,
        x: torch.Tensor,
        grid_coords: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass projecting node embeddings to 2D grid feature maps.

        Args:
            x: Node feature tensor of shape (N, D).
            grid_coords: Discrete grid coordinate tensor of shape (N, 2) [row, col].
            batch: Optional batch assignment tensor of shape (N,) with graph indices.
                   If None, assumes all nodes belong to a single graph (B=1).
            batch_size: Optional explicit batch size B. If None, inferred from batch.max() + 1.

        Returns:
            Pooled 4D tensor of shape (B, C_out, grid_h, grid_w).
        """
        num_nodes, feat_dim = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Resolve Batch Size B
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            b_size = 1 if batch_size is None else batch_size
        else:
            if batch_size is not None:
                b_size = batch_size
            elif num_nodes > 0:
                b_size = int(batch.max().item()) + 1
            else:
                b_size = 1

        # Handle empty graph edge case
        if num_nodes == 0:
            c_out = self.get_out_channels(feat_dim)
            return torch.zeros((b_size, c_out, self.grid_h, self.grid_w), dtype=dtype, device=device)

        # 2. Compute 1D Flattened Spatial Bin Indices
        # grid_coords is [row, col]
        row = grid_coords[:, 0].clamp(0, self.grid_h - 1)
        col = grid_coords[:, 1].clamp(0, self.grid_w - 1)
        spatial_bins_per_graph = self.grid_h * self.grid_w
        total_spatial_bins = b_size * spatial_bins_per_graph

        flat_idx = batch * spatial_bins_per_graph + row * self.grid_w + col

        # 3. Scatter Operations using index_add_ (fully differentiable)
        sum_grid = torch.zeros((total_spatial_bins, feat_dim), dtype=dtype, device=device)
        sum_grid.index_add_(0, flat_idx, x)

        counts = torch.zeros((total_spatial_bins, 1), dtype=dtype, device=device)
        counts.index_add_(0, flat_idx, torch.ones((num_nodes, 1), dtype=dtype, device=device))

        # 4. Aggregation
        if self.aggr == "mean":
            pooled = sum_grid / counts.clamp(min=1.0)
        elif self.aggr == "sum":
            pooled = sum_grid
        elif self.aggr == "cat_mean_sum":
            mean_grid = sum_grid / counts.clamp(min=1.0)
            pooled = torch.cat([mean_grid, sum_grid], dim=-1)
        else:
            pooled = sum_grid

        # 5. Optional Density / Occupancy Channel
        if self.include_density:
            # Normalized density map per graph
            density = counts.clamp(max=10.0) / 10.0
            pooled = torch.cat([pooled, density], dim=-1)

        # 6. Reshape to (B, H, W, C_out) then permute to standard (B, C_out, H, W)
        c_out = pooled.shape[-1]
        grid_4d = pooled.view(b_size, self.grid_h, self.grid_w, c_out).permute(0, 3, 1, 2).contiguous()

        return grid_4d

    def extra_repr(self) -> str:
        return (
            f"grid_size=({self.grid_h}, {self.grid_w}), aggr='{self.aggr}', "
            f"include_density={self.include_density}"
        )
