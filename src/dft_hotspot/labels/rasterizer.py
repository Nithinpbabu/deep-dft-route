"""
Exact length-conserving spatial rasterizer for routed wire segments and vias.
Maps continuous wire segments to a 2D discrete grid (H, W) preserving total length.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from dft_hotspot.features.def_parser import DefDieArea
from dft_hotspot.features.grid_mapper import GridMapper
from dft_hotspot.labels.def_route_parser import DefRouteSegment, DefVia


class SegmentRasterizer:
    """
    Rasterizes Manhattan wire segments and vias onto a discrete 2D grid.
    Guarantees strict conservation of wirelength across crossed grid cell boundaries.
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

        self.dx_cell = self.width / float(self.grid_w)
        self.dy_cell = self.height / float(self.grid_h)

        self.mapper = GridMapper(die_area, (self.grid_h, self.grid_w))

    def create_empty_grid(self, dtype=np.float32) -> np.ndarray:
        """Returns a zero-filled 2D array of shape (H, W)."""
        return np.zeros((self.grid_h, self.grid_w), dtype=dtype)

    def rasterize_segment(self, segment: DefRouteSegment, grid: np.ndarray, value: float) -> None:
        """
        Distribute segment value (e.g. length in microns) across crossed grid cells.
        Guarantees that sum(cells) == value (within float precision).
        """
        if value <= 0.0 or segment.orientation == "zero":
            return

        if segment.is_horizontal:
            self._rasterize_horizontal(segment.x1, segment.y1, segment.x2, segment.y2, value, grid)
        elif segment.is_vertical:
            self._rasterize_vertical(segment.x1, segment.y1, segment.x2, segment.y2, value, grid)
        elif segment.is_diagonal:
            self._rasterize_diagonal(segment.x1, segment.y1, segment.x2, segment.y2, value, grid)

    def _rasterize_horizontal(
        self, x1: float, y1: float, x2: float, y2: float, value: float, grid: np.ndarray
    ) -> None:
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        total_len = x_end - x_start
        if total_len <= 1e-9:
            return

        # Determine row index from y coordinate
        y_norm = (y1 - self.y_min) / self.height
        row = int(np.clip(np.floor(y_norm * self.grid_h), 0, self.grid_h - 1))

        # Determine column range
        c_min = int(np.clip(np.floor((x_start - self.x_min) / self.width * self.grid_w), 0, self.grid_w - 1))
        c_max = int(np.clip(np.floor((x_end - self.x_min) / self.width * self.grid_w), 0, self.grid_w - 1))

        for col in range(c_min, c_max + 1):
            cell_left = self.x_min + col * self.dx_cell
            cell_right = self.x_min + (col + 1) * self.dx_cell if col < self.grid_w - 1 else self.x_max

            overlap = max(0.0, min(x_end, cell_right) - max(x_start, cell_left))
            if overlap > 0.0:
                fraction = overlap / total_len
                grid[row, col] += fraction * value

    def _rasterize_vertical(
        self, x1: float, y1: float, x2: float, y2: float, value: float, grid: np.ndarray
    ) -> None:
        y_start = min(y1, y2)
        y_end = max(y1, y2)
        total_len = y_end - y_start
        if total_len <= 1e-9:
            return

        # Determine col index from x coordinate
        x_norm = (x1 - self.x_min) / self.width
        col = int(np.clip(np.floor(x_norm * self.grid_w), 0, self.grid_w - 1))

        # Determine row range
        r_min = int(np.clip(np.floor((y_start - self.y_min) / self.height * self.grid_h), 0, self.grid_h - 1))
        r_max = int(np.clip(np.floor((y_end - self.y_min) / self.height * self.grid_h), 0, self.grid_h - 1))

        for row in range(r_min, r_max + 1):
            cell_bottom = self.y_min + row * self.dy_cell
            cell_top = self.y_min + (row + 1) * self.dy_cell if row < self.grid_h - 1 else self.y_max

            overlap = max(0.0, min(y_end, cell_top) - max(y_start, cell_bottom))
            if overlap > 0.0:
                fraction = overlap / total_len
                grid[row, col] += fraction * value

    def _rasterize_diagonal(
        self, x1: float, y1: float, x2: float, y2: float, value: float, grid: np.ndarray
    ) -> None:
        """Manhattan decomposition of diagonal segment."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        manhattan_total = dx + dy
        if manhattan_total <= 1e-9:
            return

        val_h = value * (dx / manhattan_total)
        val_v = value * (dy / manhattan_total)

        if dx > 0.0:
            self._rasterize_horizontal(x1, y1, x2, y1, val_h, grid)
        if dy > 0.0:
            self._rasterize_vertical(x2, y1, x2, y2, val_v, grid)

    def rasterize_via(self, via: DefVia, grid: np.ndarray, count: float = 1.0) -> None:
        """Map via point to grid cell and increment count."""
        row, col = self.mapper.map_coordinate(via.x, via.y)
        grid[row, col] += count
