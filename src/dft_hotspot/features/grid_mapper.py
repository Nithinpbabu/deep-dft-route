"""
Spatial grid discretization and coordinate normalization.
Maps continuous physical layout coordinates (x, y) to a 2D discrete grid (H, W).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from dft_hotspot.features.def_parser import DefDieArea


class GridMapper:
    """
    Maps 2D layout coordinates into discrete grid bins and provides normalized coordinate channels.
    Convention:
      - H = grid rows along Y axis
      - W = grid columns along X axis
      - array[row, col] corresponds to spatial position (y, x)
    """

    def __init__(self, die_area: DefDieArea, grid_size: int | Tuple[int, int] = 64):
        if isinstance(grid_size, int):
            self.grid_h = grid_size
            self.grid_w = grid_size
        else:
            self.grid_h, self.grid_w = grid_size

        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({self.grid_h}, {self.grid_w})")

        self.die_area = die_area
        self.x_min = die_area.x_min
        self.y_min = die_area.y_min
        self.x_max = die_area.x_max
        self.y_max = die_area.y_max

        self.width = max(self.x_max - self.x_min, 1e-6)
        self.height = max(self.y_max - self.y_min, 1e-6)

    def map_coordinate(self, x: float, y: float) -> Tuple[int, int]:
        """
        Map continuous coordinate (x, y) to discrete grid indices (row, col).
        Clips out-of-bounds coordinates to valid grid indices [0..H-1, 0..W-1].
        """
        x_norm = (x - self.x_min) / self.width
        y_norm = (y - self.y_min) / self.height

        col = int(np.clip(np.floor(x_norm * self.grid_w), 0, self.grid_w - 1))
        row = int(np.clip(np.floor(y_norm * self.grid_h), 0, self.grid_h - 1))
        return row, col

    def get_normalized_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates normalized spatial coordinate channels (normalized_x, normalized_y),
        each of shape (H, W), dtype float32, centered at bin midpoints in range [0, 1].
        """
        # Column coordinates (X axis)
        cols = (np.arange(self.grid_w, dtype=np.float32) + 0.5) / float(self.grid_w)
        # Row coordinates (Y axis)
        rows = (np.arange(self.grid_h, dtype=np.float32) + 0.5) / float(self.grid_h)

        norm_x = np.tile(cols, (self.grid_h, 1))
        norm_y = np.tile(rows[:, np.newaxis], (1, self.grid_w))

        return norm_x, norm_y

    def create_empty_grid(self, dtype=np.float32) -> np.ndarray:
        """Create a zero-initialized 2D grid of shape (H, W)."""
        return np.zeros((self.grid_h, self.grid_w), dtype=dtype)
