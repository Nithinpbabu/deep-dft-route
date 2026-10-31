"""
Scan-aware spatial feature tensor extraction (Stage 3e).

Maps structured scan-chain metadata (from canonical scan_metadata_v001) onto a
2D discrete spatial grid (C x H x W), producing multi-channel scan-aware feature
representations matching Stage 2 physical layout dimensions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from dft_hotspot.dft.scan_metadata import ScanCell, ScanChain, ScanMetadata
from dft_hotspot.features.def_parser import DefDieArea, DefParser
from dft_hotspot.features.grid_mapper import GridMapper
from dft_hotspot.labels.def_route_parser import DefRouteSegment
from dft_hotspot.labels.rasterizer import SegmentRasterizer

SCAN_CHANNEL_NAMES = [
    "scan_cell_density",
    "scan_chain_endpoint_density",
    "scan_chain_segment_density",
    "scan_chain_order_index",
    "scan_chain_local_wirelength_proxy",
]

NORMALIZATION_METHOD = (
    "channel_max_scaling: raw / max(channel_max, 1.0) for density and wirelength channels; "
    "scan_chain_order_index is inherently bounded in [0.0, 1.0]"
)


class ScanFeatureExtractor:
    """
    Extracts 2D spatial feature tensors from structured scan metadata.
    """

    def __init__(
        self,
        grid_size: int | Tuple[int, int] = 64,
        die_area: Optional[DefDieArea] = None,
    ):
        if isinstance(grid_size, int):
            self.grid_h = grid_size
            self.grid_w = grid_size
        else:
            self.grid_h, self.grid_w = grid_size

        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({self.grid_h}, {self.grid_w})")

        self.die_area = die_area

    def resolve_die_area(
        self,
        scan_metadata: ScanMetadata,
        sample_dir: Optional[Path | str] = None,
        explicit_die_area: Optional[DefDieArea] = None,
    ) -> DefDieArea:
        """
        Resolves DefDieArea from explicit argument, sample metadata.json, DEF file, or cell bounds.
        """
        if explicit_die_area is not None:
            return explicit_die_area
        if self.die_area is not None:
            return self.die_area

        # Check sample_dir / metadata.json
        if sample_dir:
            s_dir = Path(sample_dir)
            sample_meta_path = s_dir / "metadata.json"
            if sample_meta_path.is_file():
                try:
                    with open(sample_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    die_dbu = meta.get("die_area_dbu")
                    dbu = meta.get("dbu_per_micron", 1000)
                    if die_dbu:
                        return DefDieArea(
                            x_min=float(die_dbu["x_min"]),
                            y_min=float(die_dbu["y_min"]),
                            x_max=float(die_dbu["x_max"]),
                            y_max=float(die_dbu["y_max"]),
                            dbu_per_micron=int(dbu),
                        )
                except Exception:
                    pass

        # Check scan_metadata.def_path if file exists
        if scan_metadata.def_path:
            dp = Path(scan_metadata.def_path)
            if dp.is_file():
                try:
                    def_data = DefParser.parse_file(dp)
                    if def_data.die_area:
                        return def_data.die_area
                except Exception:
                    pass

        # Fallback: compute bounding box of scan cells
        all_cells = [c for chain in scan_metadata.chains for c in chain.cells if c.x is not None and c.y is not None]
        if not all_cells:
            return DefDieArea(0.0, 0.0, 100000.0, 100000.0, dbu_per_micron=1000)

        xs = [c.x for c in all_cells if c.x is not None]
        ys = [c.y for c in all_cells if c.y is not None]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_x = max(1000.0, (x_max - x_min) * 0.1)
        pad_y = max(1000.0, (y_max - y_min) * 0.1)
        return DefDieArea(
            x_min=max(0.0, x_min - pad_x),
            y_min=max(0.0, y_min - pad_y),
            x_max=x_max + pad_x,
            y_max=y_max + pad_y,
            dbu_per_micron=1000,
        )

    def extract_from_scan_metadata(
        self,
        scan_metadata: ScanMetadata,
        die_area: Optional[DefDieArea] = None,
        source_scan_metadata_path: Optional[str | Path] = None,
        sample_dir: Optional[str | Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extracts (norm_tensor, raw_tensor, metadata) from ScanMetadata.
        Tensors have shape (5, H, W) and dtype float32.
        """
        resolved_die = self.resolve_die_area(
            scan_metadata=scan_metadata,
            sample_dir=sample_dir,
            explicit_die_area=die_area,
        )
        dbu = float(resolved_die.dbu_per_micron or 1000)

        mapper = GridMapper(resolved_die, (self.grid_h, self.grid_w))
        rasterizer = SegmentRasterizer(resolved_die, (self.grid_h, self.grid_w))

        # Initialize 5 raw channel grids
        grid_cells = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        grid_endpoints = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        grid_segments = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        grid_order = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        grid_local_wl = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        total_segments = 0
        total_scan_wirelength_um = 0.0
        bin_order_indices: Dict[Tuple[int, int], List[float]] = {}

        for chain in scan_metadata.chains:
            placed_cells = [c for c in chain.cells if c.placed and c.x is not None and c.y is not None]
            n_cells = len(placed_cells)
            if n_cells == 0:
                continue

            # 1. Scan cell density & order index collection
            for i, cell in enumerate(placed_cells):
                r, c = mapper.map_coordinate(cell.x, cell.y)
                grid_cells[r, c] += 1.0

                norm_idx = 0.0 if n_cells <= 1 else float(i) / float(n_cells - 1)
                bin_order_indices.setdefault((r, c), []).append(norm_idx)

            # 2. Endpoint density (start -> first placed cell, stop -> last placed cell)
            r_start, c_start = mapper.map_coordinate(placed_cells[0].x, placed_cells[0].y)
            grid_endpoints[r_start, c_start] += 1.0

            r_stop, c_stop = mapper.map_coordinate(placed_cells[-1].x, placed_cells[-1].y)
            grid_endpoints[r_stop, c_stop] += 1.0

            # 3. Scan chain segment density (Manhattan line segments between consecutive cells)
            for i in range(n_cells - 1):
                c1 = placed_cells[i]
                c2 = placed_cells[i + 1]

                dx = abs(c2.x - c1.x)
                dy = abs(c2.y - c1.y)
                dx_um = dx / dbu
                dy_um = dy / dbu
                seg_len_um = dx_um + dy_um

                total_segments += 1
                total_scan_wirelength_um += seg_len_um

                if dx > 0:
                    seg_h = DefRouteSegment(
                        net_name=chain.chain_id,
                        layer="synthetic_scan",
                        x1=c1.x,
                        y1=c1.y,
                        x2=c2.x,
                        y2=c1.y,
                        length_dbu=dx,
                        length_um=dx_um,
                        orientation="horizontal",
                    )
                    rasterizer.rasterize_segment(seg_h, grid_segments, value=dx_um)

                if dy > 0:
                    seg_v = DefRouteSegment(
                        net_name=chain.chain_id,
                        layer="synthetic_scan",
                        x1=c2.x,
                        y1=c1.y,
                        x2=c2.x,
                        y2=c2.y,
                        length_dbu=dy,
                        length_um=dy_um,
                        orientation="vertical",
                    )
                    rasterizer.rasterize_segment(seg_v, grid_segments, value=dy_um)

            # 4. Scan chain local wirelength proxy
            for i in range(n_cells):
                c_curr = placed_cells[i]
                wl_inc = 0.0
                if i > 0:
                    c_prev = placed_cells[i - 1]
                    wl_inc += (abs(c_curr.x - c_prev.x) + abs(c_curr.y - c_prev.y)) / dbu
                if i < n_cells - 1:
                    c_next = placed_cells[i + 1]
                    wl_inc += (abs(c_curr.x - c_next.x) + abs(c_curr.y - c_next.y)) / dbu

                r, c = mapper.map_coordinate(c_curr.x, c_curr.y)
                grid_local_wl[r, c] += wl_inc

        # Compute average order index per bin
        for (r, c), indices in bin_order_indices.items():
            grid_order[r, c] = float(np.mean(indices))

        # Assemble raw tensor
        raw_tensor = np.stack(
            [grid_cells, grid_endpoints, grid_segments, grid_order, grid_local_wl],
            axis=0,
        ).astype(np.float32)

        # Normalize tensor
        norm_tensor = np.zeros_like(raw_tensor)
        for k in range(len(SCAN_CHANNEL_NAMES)):
            if k == 3:
                # scan_chain_order_index is already bounded in [0.0, 1.0]
                norm_tensor[k] = raw_tensor[k]
            else:
                max_val = float(np.max(raw_tensor[k]))
                norm_tensor[k] = raw_tensor[k] / max(max_val, 1.0)

        # Compute channel statistics
        channel_stats = {}
        for k, name in enumerate(SCAN_CHANNEL_NAMES):
            r_plane = raw_tensor[k]
            n_plane = norm_tensor[k]
            channel_stats[name] = {
                "channel_index": k,
                "raw_min": float(np.min(r_plane)),
                "raw_max": float(np.max(r_plane)),
                "raw_sum": float(np.sum(r_plane)),
                "raw_mean": float(np.mean(r_plane)),
                "norm_min": float(np.min(n_plane)),
                "norm_max": float(np.max(n_plane)),
                "norm_sum": float(np.sum(n_plane)),
                "norm_mean": float(np.mean(n_plane)),
                "active_bins": int(np.count_nonzero(r_plane > 0)),
            }

        # Build limitations list
        limitations: List[str] = []
        if scan_metadata.source == "synthetic_proxy":
            limitations.extend([
                "Synthetic proxy scan metadata only; no scan replacement or ATPG was performed.",
                "Scan segment density is a proxy based on deterministic synthetic chain order.",
                "Scan pins are unknown because cells were not replaced with real scan flops.",
                "Use only for scan-chain routing demand proxy experiments.",
            ])
        else:
            limitations.append("Extracted from OpenROAD DEF SCANCHAINS; no ATPG patterns generated.")
            for lim in scan_metadata.limitations:
                if "functional output" in lim and lim not in limitations:
                    limitations.append(lim)

        metadata: Dict[str, Any] = {
            "schema_version": "scan_features_v001",
            "source_scan_metadata": str(source_scan_metadata_path).replace("\\", "/") if source_scan_metadata_path else None,
            "scan_source": scan_metadata.source,
            "design": scan_metadata.design,
            "platform": scan_metadata.platform,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "grid_size": [self.grid_h, self.grid_w],
            "tensor_shape": list(norm_tensor.shape),
            "channel_names": SCAN_CHANNEL_NAMES,
            "wirelength_unit": "microns",
            "normalization_method": NORMALIZATION_METHOD,
            "die_area_um": {
                "x_min": round(resolved_die.x_min / dbu, 4),
                "y_min": round(resolved_die.y_min / dbu, 4),
                "x_max": round(resolved_die.x_max / dbu, 4),
                "y_max": round(resolved_die.y_max / dbu, 4),
                "width": round(resolved_die.width_um, 4),
                "height": round(resolved_die.height_um, 4),
            },
            "die_area_dbu": {
                "x_min": float(resolved_die.x_min),
                "y_min": float(resolved_die.y_min),
                "x_max": float(resolved_die.x_max),
                "y_max": float(resolved_die.y_max),
                "width": float(resolved_die.width),
                "height": float(resolved_die.height),
            },
            "dbu_per_micron": int(dbu),
            "chain_count": scan_metadata.statistics.chain_count,
            "total_scan_cells": scan_metadata.statistics.total_scan_cells,
            "placed_scan_cells": scan_metadata.statistics.placed_scan_cells,
            "segment_count": total_segments,
            "total_scan_wirelength_um": round(total_scan_wirelength_um, 4),
            "channel_statistics": channel_stats,
            "limitations": limitations,
        }

        return norm_tensor, raw_tensor, metadata

    def process_and_save_sample(
        self,
        sample_dir: str | Path,
        scan_metadata_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Path]:
        """
        Loads scan metadata, extracts spatial feature tensors, and writes:
          - features_scan.npz
          - scan_feature_metadata.json
          - scan_feature_preview.png
        """
        s_dir = Path(sample_dir)
        out_dir = Path(output_dir) if output_dir else s_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Locate and load scan_metadata.json
        if scan_metadata_path:
            meta_json = Path(scan_metadata_path)
        else:
            meta_json = s_dir / "scan_metadata.json"

        if not meta_json.is_file():
            raise FileNotFoundError(f"Scan metadata not found at: {meta_json}")

        from dft_hotspot.dft.scan_metadata import ScanMetadata
        scan_meta = ScanMetadata.load_json(meta_json)

        # 2. Extract features
        norm_tensor, raw_tensor, metadata = self.extract_from_scan_metadata(
            scan_metadata=scan_meta,
            source_scan_metadata_path=meta_json,
            sample_dir=s_dir,
        )

        # 3. Save features_scan.npz
        npz_path = out_dir / "features_scan.npz"
        np.savez_compressed(
            npz_path,
            scan_features=norm_tensor,
            raw_scan_features=raw_tensor,
            features=norm_tensor,
            raw_features=raw_tensor,
            channel_names=SCAN_CHANNEL_NAMES,
        )

        # 4. Save scan_feature_metadata.json
        metadata_path = out_dir / "scan_feature_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # 5. Save scan_feature_preview.png
        preview_path = out_dir / "scan_feature_preview.png"
        generate_scan_preview(norm_tensor, metadata, preview_path)

        return {
            "npz": npz_path,
            "metadata": metadata_path,
            "preview": preview_path,
        }


def generate_scan_preview(
    features: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str | Path,
    channel_names: Optional[List[str]] = None,
) -> Path:
    """
    Renders a 5-panel publication-quality visualization of extracted scan-aware feature channels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    names = channel_names or metadata.get("channel_names", SCAN_CHANNEL_NAMES)
    cmaps = [
        "plasma",    # scan_cell_density
        "magma",     # scan_chain_endpoint_density
        "turbo",     # scan_chain_segment_density
        "coolwarm",  # scan_chain_order_index
        "inferno",   # scan_chain_local_wirelength_proxy
    ]

    titles = [
        "Scan Cell Density",
        "Chain Endpoint Density",
        "Chain Segment Density (µm)",
        "Chain Order Index [0, 1]",
        "Local Wirelength Proxy (µm)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.2), facecolor="#181824")

    design = metadata.get("design", "Design")
    platform = metadata.get("platform", "Platform")
    scan_source = metadata.get("scan_source", "unknown")
    grid_size = metadata.get("grid_size", [64, 64])
    die_um = metadata.get("die_area_um", {})
    die_str = f"{die_um.get('width', 100):.1f} x {die_um.get('height', 100):.1f} µm" if die_um else ""

    fig.suptitle(
        f"Scan-Aware Feature Maps — {design} ({platform.upper()}) | Source: {scan_source} | Grid: {grid_size[0]}x{grid_size[1]} | Die: {die_str}",
        fontsize=14,
        fontweight="bold",
        color="#F0F4F8",
        y=0.98,
    )

    stats = metadata.get("channel_statistics", {})

    for i in range(5):
        ax = axes[i]
        plane = features[i]
        ch_name = names[i] if i < len(names) else f"Channel {i}"
        cmap = cmaps[i % len(cmaps)]
        title = titles[i] if i < len(titles) else ch_name

        im = ax.imshow(
            plane,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )

        ch_stat = stats.get(ch_name, {})
        raw_max = ch_stat.get("raw_max", float(np.max(plane)))
        active = ch_stat.get("active_bins", int(np.count_nonzero(plane > 0)))

        ax.set_title(
            f"{title}\nmax: {raw_max:.2f} | active: {active}",
            fontsize=10.5,
            color="#E2E8F0",
            pad=8,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.2)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors="#94A3B8")
        cbar.outline.set_edgecolor("#334155")

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.93])
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return out_file
