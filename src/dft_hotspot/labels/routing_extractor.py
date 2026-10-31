"""
Routing Label Extractor.
Extracts post-route routed wire density labels and via counts from DEF layouts into
standardized 2D grid tensors (8 x H x W), computes comprehensive metadata, and
coordinates sample artifact generation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dft_hotspot.labels.def_route_parser import DefRouteData, DefRouteParser
from dft_hotspot.labels.rasterizer import SegmentRasterizer
from dft_hotspot.labels.visualization import generate_routing_preview

ROUTING_CHANNEL_NAMES = [
    "total_wirelength_density",
    "horizontal_wirelength_density",
    "vertical_wirelength_density",
    "via_count_density",
    "metal1_wirelength_density",
    "metal2_wirelength_density",
    "metal3_wirelength_density",
    "metal4plus_wirelength_density",
]

NORMALIZATION_METHOD = "channel_max_scaling: raw / max(channel_max, 1.0)"


def _classify_layer_bucket(layer_name: str) -> str:
    """
    Classifies a layer string into 'metal1', 'metal2', 'metal3', or 'metal4plus'.
    Handles names like metal1, m1, Metal1, metal2, etc.
    """
    norm = layer_name.lower()
    m = re.search(r"(\d+)", norm)
    if m:
        num = int(m.group(1))
        if num == 1:
            return "metal1"
        elif num == 2:
            return "metal2"
        elif num == 3:
            return "metal3"
        else:
            return "metal4plus"
    return "metal4plus"


class RoutingLabelExtractor:
    """
    Extracts 8-channel post-route wirelength and via density label tensors.
    """

    def __init__(self, grid_size: int | Tuple[int, int] = 64):
        if isinstance(grid_size, int):
            self.grid_h = grid_size
            self.grid_w = grid_size
        else:
            self.grid_h, self.grid_w = grid_size

    def extract_from_route_data(
        self,
        route_data: DefRouteData,
        source_run_dir: Optional[str | Path] = None,
        def_path: Optional[str | Path] = None,
        design_name: Optional[str] = None,
        pdk_name: Optional[str] = "nangate45",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extracts 8-channel label tensors from pre-parsed DefRouteData.

        Returns:
            labels: normalized float32 tensor of shape (8, H, W)
            raw_labels: unnormalized float32 tensor of shape (8, H, W)
            metadata: structured dictionary of metadata
        """
        if route_data.die_area is None:
            raise ValueError("DEF layout missing DIEAREA definition; cannot compute spatial grid.")

        rasterizer = SegmentRasterizer(route_data.die_area, (self.grid_h, self.grid_w))

        # Channel 0: total wirelength
        ch_total = rasterizer.create_empty_grid()
        # Channel 1: horizontal wirelength
        ch_horiz = rasterizer.create_empty_grid()
        # Channel 2: vertical wirelength
        ch_vert = rasterizer.create_empty_grid()
        # Channel 3: via count
        ch_vias = rasterizer.create_empty_grid()
        # Channel 4: metal1 wirelength
        ch_m1 = rasterizer.create_empty_grid()
        # Channel 5: metal2 wirelength
        ch_m2 = rasterizer.create_empty_grid()
        # Channel 6: metal3 wirelength
        ch_m3 = rasterizer.create_empty_grid()
        # Channel 7: metal4+ wirelength
        ch_m4plus = rasterizer.create_empty_grid()

        horiz_seg_count = 0
        vert_seg_count = 0
        diag_seg_count = 0
        zero_seg_count = 0

        # Rasterize wire segments
        for seg in route_data.segments:
            val = seg.length_um

            # Channel 0: All wire segments
            rasterizer.rasterize_segment(seg, ch_total, val)

            # Channel 1 & 2: Orientation
            if seg.is_horizontal:
                rasterizer.rasterize_segment(seg, ch_horiz, val)
                horiz_seg_count += 1
            elif seg.is_vertical:
                rasterizer.rasterize_segment(seg, ch_vert, val)
                vert_seg_count += 1
            elif seg.is_diagonal:
                diag_seg_count += 1
                # Decompose diagonal segment into horiz and vert proportions
                dx = abs(seg.x2 - seg.x1)
                dy = abs(seg.y2 - seg.y1)
                manhattan = dx + dy
                if manhattan > 1e-9:
                    val_h = val * (dx / manhattan)
                    val_v = val * (dy / manhattan)
                    rasterizer._rasterize_horizontal(seg.x1, seg.y1, seg.x2, seg.y1, val_h, ch_horiz)
                    rasterizer._rasterize_vertical(seg.x2, seg.y1, seg.x2, seg.y2, val_v, ch_vert)
            else:
                zero_seg_count += 1

            # Channels 4-7: Layer breakdown
            layer_bucket = _classify_layer_bucket(seg.layer)
            if layer_bucket == "metal1":
                rasterizer.rasterize_segment(seg, ch_m1, val)
            elif layer_bucket == "metal2":
                rasterizer.rasterize_segment(seg, ch_m2, val)
            elif layer_bucket == "metal3":
                rasterizer.rasterize_segment(seg, ch_m3, val)
            else:
                rasterizer.rasterize_segment(seg, ch_m4plus, val)

        # Channel 3: Rasterize vias
        for via in route_data.vias:
            rasterizer.rasterize_via(via, ch_vias, count=1.0)

        # Assemble raw tensor (8, H, W)
        raw_tensor = np.stack(
            [
                ch_total,
                ch_horiz,
                ch_vert,
                ch_vias,
                ch_m1,
                ch_m2,
                ch_m3,
                ch_m4plus,
            ],
            axis=0,
        ).astype(np.float32)

        # Build normalized tensor
        norm_tensor = np.zeros_like(raw_tensor)
        for i in range(len(ROUTING_CHANNEL_NAMES)):
            raw_ch = raw_tensor[i]
            ch_max = float(raw_ch.max())
            denom = max(ch_max, 1.0)
            norm_tensor[i] = raw_ch / denom

        # Strict validation
        if not np.all(np.isfinite(norm_tensor)):
            raise ValueError("Extracted routing label tensor contains NaN or Inf values!")

        # Statistics per channel
        channel_stats: Dict[str, Dict[str, Any]] = {}
        for idx, name in enumerate(ROUTING_CHANNEL_NAMES):
            r_ch = raw_tensor[idx]
            n_ch = norm_tensor[idx]
            channel_stats[name] = {
                "channel_index": idx,
                "raw_min": float(r_ch.min()),
                "raw_max": float(r_ch.max()),
                "raw_sum": float(r_ch.sum()),
                "norm_min": float(n_ch.min()),
                "norm_max": float(n_ch.max()),
                "norm_mean": float(n_ch.mean()),
                "active_bins": int(np.count_nonzero(r_ch)),
            }

        die_um = {
            "x_min": float(route_data.die_area.x_min / route_data.dbu_per_micron),
            "y_min": float(route_data.die_area.y_min / route_data.dbu_per_micron),
            "x_max": float(route_data.die_area.x_max / route_data.dbu_per_micron),
            "y_max": float(route_data.die_area.y_max / route_data.dbu_per_micron),
            "width": float(route_data.die_area.width_um),
            "height": float(route_data.die_area.height_um),
        }

        die_dbu = {
            "x_min": float(route_data.die_area.x_min),
            "y_min": float(route_data.die_area.y_min),
            "x_max": float(route_data.die_area.x_max),
            "y_max": float(route_data.die_area.y_max),
            "width": float(route_data.die_area.width),
            "height": float(route_data.die_area.height),
        }

        metadata: Dict[str, Any] = {
            "design": design_name or "unknown_design",
            "pdk": pdk_name or "nangate45",
            "extraction_version": "stage2b_v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_run_dir": str(source_run_dir) if source_run_dir else None,
            "def_path": str(def_path) if def_path else None,
            "grid_size": [self.grid_h, self.grid_w],
            "tensor_shape": list(norm_tensor.shape),
            "channel_names": ROUTING_CHANNEL_NAMES,
            "wirelength_unit": "microns",
            "normalization_method": NORMALIZATION_METHOD,
            "die_area_um": die_um,
            "die_area_dbu": die_dbu,
            "dbu_per_micron": route_data.dbu_per_micron,
            "routing_statistics": {
                "total_nets": route_data.total_nets_count,
                "routed_nets": route_data.routed_nets_count,
                "unrouted_nets": max(0, route_data.total_nets_count - route_data.routed_nets_count),
                "total_segments": route_data.total_segments,
                "horizontal_segments": horiz_seg_count,
                "vertical_segments": vert_seg_count,
                "diagonal_segments": diag_seg_count,
                "zero_segments": zero_seg_count,
                "total_vias": route_data.total_vias,
                "total_wirelength_um": float(route_data.total_wirelength_um),
                "total_wirelength_dbu": float(route_data.total_wirelength_dbu),
                "layer_wirelengths_um": {k: float(v) for k, v in route_data.layer_wirelengths_um.items()},
                "layer_wirelengths_dbu": {k: float(v) for k, v in route_data.layer_wirelengths_dbu.items()},
                "layer_segment_counts": route_data.layer_segment_counts,
                "layer_via_counts": route_data.layer_via_counts,
            },
            "channel_statistics": channel_stats,
            "skipped_unsupported_syntax_counts": {
                "skipped_diagonal_segments": route_data.skipped_diagonal_count,
                "skipped_syntax_items": route_data.skipped_syntax_count,
            },
            "congestion_data_status": (
                "unavailable_in_reports: OpenROAD reports directory empty; true routing congestion/overflow "
                "channel omitted in accordance with Stage 2b methodology (no false congestion claims)."
            ),
            "known_limitations": [
                "Ground-truth routed wire densities extracted from DEF ROUTED wire segments",
                "Via coordinates mapped as point densities to containing grid cell",
                "Metal4+ density aggregates all metallization layers >= metal4 (e.g., metal4, metal5, metal6)",
                "Segment wirelength distributed across crossed cell boundaries using exact 1D interval intersections",
                "True routing congestion / overflow omitted pending detailed global routing report generation in flow",
            ],
        }

        return norm_tensor, raw_tensor, metadata

    def extract_from_def_file(
        self,
        def_path: str | Path,
        source_run_dir: Optional[str | Path] = None,
        design_name: Optional[str] = None,
        pdk_name: Optional[str] = "nangate45",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Parses DEF file from disk and extracts routing label tensors."""
        path = Path(def_path)
        route_data = DefRouteParser.parse_file(path)
        if design_name is None:
            stem = path.stem.replace("_final", "")
            design_name = stem.split("_")[0]

        return self.extract_from_route_data(
            route_data=route_data,
            source_run_dir=source_run_dir,
            def_path=path,
            design_name=design_name,
            pdk_name=pdk_name,
        )

    def process_and_save_sample(
        self,
        def_path: str | Path,
        output_dir: str | Path,
        source_run_dir: Optional[str | Path] = None,
        design_name: Optional[str] = None,
        pdk_name: Optional[str] = "nangate45",
    ) -> Dict[str, Path]:
        """
        Extracts routing labels and writes sample artifacts to output_dir:
          - labels_routing.npz
          - routing_label_metadata.json
          - routing_preview.png
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        labels, raw_labels, metadata = self.extract_from_def_file(
            def_path=def_path,
            source_run_dir=source_run_dir,
            design_name=design_name,
            pdk_name=pdk_name,
        )

        # 1. Save labels_routing.npz
        npz_path = out_dir / "labels_routing.npz"
        np.savez_compressed(
            npz_path,
            labels=labels,
            raw_labels=raw_labels,
            channel_names=ROUTING_CHANNEL_NAMES,
        )

        # 2. Save routing_label_metadata.json
        meta_path = out_dir / "routing_label_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # 3. Save routing_preview.png
        preview_path = out_dir / "routing_preview.png"
        generate_routing_preview(labels, metadata, preview_path)

        return {
            "npz": npz_path,
            "metadata": meta_path,
            "preview": preview_path,
        }
