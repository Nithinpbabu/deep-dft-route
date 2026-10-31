"""
Physical feature extraction engine.
Transforms DEF physical design layouts into standardized 2D spatial feature tensors (C x H x W),
serializes .npz arrays, generates comprehensive metadata, and creates visual previews.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dft_hotspot.features.cell_classifier import CellClassifier, HEURISTIC_DESCRIPTION, classify_cell
from dft_hotspot.features.def_parser import DefData, DefParser
from dft_hotspot.features.grid_mapper import GridMapper
from dft_hotspot.features.visualization import generate_preview

CHANNEL_NAMES = [
    "cell_density",
    "sequential_cell_density",
    "combinational_cell_density",
    "pin_density",
    "blockage_or_macro_density",
    "normalized_x",
    "normalized_y",
]

NORMALIZATION_METHOD = "channel_max_scaling: raw / max(channel_max, 1.0)"


class PhysicalFeatureExtractor:
    """
    Extracts spatial grid feature tensors from physical design DEF files.
    """

    def __init__(
        self,
        grid_size: int | Tuple[int, int] = 64,
        classifier: Optional[CellClassifier] = None,
    ):
        if isinstance(grid_size, int):
            self.grid_h = grid_size
            self.grid_w = grid_size
        else:
            self.grid_h, self.grid_w = grid_size

        self.classifier = classifier or CellClassifier()

    def extract_from_def_data(
        self,
        def_data: DefData,
        source_run_dir: Optional[str | Path] = None,
        def_path: Optional[str | Path] = None,
        design_name: Optional[str] = None,
        pdk_name: Optional[str] = "nangate45",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extracts spatial features from pre-parsed DefData.

        Returns:
            features: normalized feature tensor of shape (7, H, W), float32
            raw_features: raw count feature tensor of shape (7, H, W), float32
            metadata: dictionary of extraction metadata
        """
        if def_data.die_area is None:
            raise ValueError("DEF data missing DIEAREA definition; cannot compute spatial grid.")

        mapper = GridMapper(def_data.die_area, (self.grid_h, self.grid_w))

        # Initialize raw 2D channels
        raw_cell_density = mapper.create_empty_grid()
        raw_seq_density = mapper.create_empty_grid()
        raw_comb_density = mapper.create_empty_grid()
        raw_pin_density = mapper.create_empty_grid()
        raw_blockage_density = mapper.create_empty_grid()

        # Count instance pins from net connectivity
        inst_pin_counts = def_data.get_instance_pin_counts()

        seq_count = 0
        comb_count = 0

        # Map placed components
        for comp in def_data.components:
            row, col = mapper.map_coordinate(comp.x, comp.y)
            raw_cell_density[row, col] += 1.0

            is_seq = self.classifier.is_sequential(comp.master)
            if is_seq:
                raw_seq_density[row, col] += 1.0
                seq_count += 1
            else:
                raw_comb_density[row, col] += 1.0
                comb_count += 1

            # Pin density contribution from component's connected nets
            pins = float(inst_pin_counts.get(comp.name, 0))
            raw_pin_density[row, col] += pins

        # Map top-level IO pins from PINS section
        for pin in def_data.pins:
            row, col = mapper.map_coordinate(pin.x, pin.y)
            raw_pin_density[row, col] += 1.0

        # Map blockages if present
        blockage_status = "unavailable_in_def: standard cell flat placement without macros/blockages"
        if def_data.blockages:
            blockage_status = f"extracted_{len(def_data.blockages)}_blockages"
            for blk in def_data.blockages:
                r1, c1 = mapper.map_coordinate(blk.x_min, blk.y_min)
                r2, c2 = mapper.map_coordinate(blk.x_max, blk.y_max)
                r_start, r_end = min(r1, r2), max(r1, r2) + 1
                c_start, c_end = min(c1, c2), max(c1, c2) + 1
                raw_blockage_density[r_start:r_end, c_start:c_end] = 1.0

        # Normalized coordinates
        norm_x, norm_y = mapper.get_normalized_coordinates()

        # Assemble raw tensor (7, H, W)
        raw_tensor = np.stack(
            [
                raw_cell_density,
                raw_seq_density,
                raw_comb_density,
                raw_pin_density,
                raw_blockage_density,
                norm_x,
                norm_y,
            ],
            axis=0,
        ).astype(np.float32)

        # Build normalized tensor
        # Density channels (0..4) are normalized by max value
        norm_tensor = np.zeros_like(raw_tensor)
        for i in range(5):
            ch = raw_tensor[i]
            max_v = float(ch.max())
            denom = max(max_v, 1.0)
            norm_tensor[i] = ch / denom

        # Coordinate channels (5, 6) are inherently in [0, 1]
        norm_tensor[5] = norm_x
        norm_tensor[6] = norm_y

        # Validate finiteness
        if not np.all(np.isfinite(norm_tensor)):
            raise ValueError("Extracted feature tensor contains NaN or Inf values!")

        # Channel statistics
        stats: Dict[str, Dict[str, Any]] = {}
        for idx, name in enumerate(CHANNEL_NAMES):
            ch_raw = raw_tensor[idx]
            ch_norm = norm_tensor[idx]
            stats[name] = {
                "channel_index": idx,
                "raw_min": float(ch_raw.min()),
                "raw_max": float(ch_raw.max()),
                "raw_sum": float(ch_raw.sum()),
                "norm_min": float(ch_norm.min()),
                "norm_max": float(ch_norm.max()),
                "norm_mean": float(ch_norm.mean()),
                "active_bins": int(np.count_nonzero(ch_raw)),
            }

        die_um = {
            "x_min": float(def_data.die_area.x_min / def_data.dbu_per_micron),
            "y_min": float(def_data.die_area.y_min / def_data.dbu_per_micron),
            "x_max": float(def_data.die_area.x_max / def_data.dbu_per_micron),
            "y_max": float(def_data.die_area.y_max / def_data.dbu_per_micron),
            "width": float(def_data.die_area.width_um),
            "height": float(def_data.die_area.height_um),
        }

        die_dbu = {
            "x_min": float(def_data.die_area.x_min),
            "y_min": float(def_data.die_area.y_min),
            "x_max": float(def_data.die_area.x_max),
            "y_max": float(def_data.die_area.y_max),
            "width": float(def_data.die_area.width),
            "height": float(def_data.die_area.height),
        }

        metadata: Dict[str, Any] = {
            "design": design_name or "unknown_design",
            "pdk": pdk_name,
            "extraction_version": "stage2a_v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_run_dir": str(source_run_dir) if source_run_dir else None,
            "def_path": str(def_path) if def_path else None,
            "grid_size": [self.grid_h, self.grid_w],
            "tensor_shape": list(norm_tensor.shape),
            "channel_names": CHANNEL_NAMES,
            "normalization_method": NORMALIZATION_METHOD,
            "die_area_um": die_um,
            "die_area_dbu": die_dbu,
            "dbu_per_micron": def_data.dbu_per_micron,
            "design_statistics": {
                "total_components": def_data.total_components,
                "sequential_components": seq_count,
                "combinational_components": comb_count,
                "total_io_pins": def_data.total_pins,
                "total_nets": def_data.total_nets,
                "total_net_pin_connections": sum(inst_pin_counts.values()),
            },
            "channel_statistics": stats,
            "sequential_heuristic": HEURISTIC_DESCRIPTION,
            "pin_density_proxy": (
                "netlist_connectivity_pin_proxy: instance pin connections across DEF NETS "
                "accumulated at instance origin coordinates plus IO pins from DEF PINS."
            ),
            "blockage_status": blockage_status,
            "limitations": [
                "DEF-only text parsing; standard cell internal pin offsets not extracted without LEF geometry",
                "Pin coordinates approximated at instance placement origin",
                "Sequential/combinational classification relies on master name regex heuristics",
                "Blockage channel zeroed out for flat standard cell run without macro blocks",
                "Post-route routing congestion labels deferred to Stage 2b/2c",
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
        """
        Parses DEF file from disk and extracts spatial features.
        """
        path = Path(def_path)
        def_data = DefParser.parse_file(path)
        if design_name is None:
            # Infer from DEF filename or parent
            stem = path.stem.replace("_final", "")
            design_name = stem.split("_")[0]

        return self.extract_from_def_data(
            def_data=def_data,
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
        Extracts features and writes sample artifacts to output_dir:
          - features_spatial.npz
          - metadata.json
          - preview.png
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        features, raw_features, metadata = self.extract_from_def_file(
            def_path=def_path,
            source_run_dir=source_run_dir,
            design_name=design_name,
            pdk_name=pdk_name,
        )

        # 1. Save features_spatial.npz
        npz_path = out_dir / "features_spatial.npz"
        np.savez_compressed(
            npz_path,
            features=features,
            raw_features=raw_features,
            channel_names=CHANNEL_NAMES,
        )

        # 2. Save metadata.json
        meta_path = out_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # 3. Save preview.png
        preview_path = out_dir / "preview.png"
        generate_preview(features, metadata, preview_path)

        return {
            "npz": npz_path,
            "metadata": meta_path,
            "preview": preview_path,
        }

    def extract_net_bounding_box_density(
        self,
        def_data: DefData,
        weighted: bool = False,
    ) -> np.ndarray:
        """
        Extracts 2D net bounding-box density map for this extractor's grid resolution.
        """
        if def_data.die_area is None:
            raise ValueError("DEF data missing DIEAREA definition.")
        mapper = GridMapper(def_data.die_area, (self.grid_h, self.grid_w))
        return compute_net_bounding_box_density(def_data, mapper, weighted=weighted)


def compute_net_bounding_box_density(
    def_data: DefData,
    mapper: GridMapper,
    weighted: bool = False,
) -> np.ndarray:
    """
    Computes 2D net bounding-box density map over the spatial grid.
    For each net in def_data, finds the layout bounding box containing all connected
    instances and pins, and projects that bounding box onto the discrete grid.

    Parameters:
        def_data: Parsed DefData containing components, pins, and nets
        mapper: GridMapper for the design's die area and grid resolution
        weighted: If True, each net's contribution is normalized by the number of
                  grid cells covered (HPWL/area demand proxy).
                  If False, unweighted count (each covered cell increments by 1.0).

    Returns:
        2D numpy array of shape (H, W) with float32 density values.
    """
    grid = mapper.create_empty_grid()
    inst_locs = def_data.get_instance_locations()
    pin_locs = def_data.get_pin_locations()

    for net in def_data.nets:
        xs: List[float] = []
        ys: List[float] = []

        for inst, pin in net.connections:
            if inst == "PIN":
                if pin in pin_locs:
                    px, py = pin_locs[pin]
                    xs.append(px)
                    ys.append(py)
            elif inst in inst_locs:
                ix, iy = inst_locs[inst]
                xs.append(ix)
                ys.append(iy)
            elif inst in pin_locs:
                px, py = pin_locs[inst]
                xs.append(px)
                ys.append(py)

        if len(xs) < 2:
            # Need at least 2 terminals to define a net bounding box
            continue

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        r1, c1 = mapper.map_coordinate(min_x, min_y)
        r2, c2 = mapper.map_coordinate(max_x, max_y)

        r_start, r_end = min(r1, r2), max(r1, r2) + 1
        c_start, c_end = min(c1, c2), max(c1, c2) + 1

        num_cells = (r_end - r_start) * (c_end - c_start)
        weight = (1.0 / float(num_cells)) if (weighted and num_cells > 0) else 1.0

        grid[r_start:r_end, c_start:c_end] += weight

    return grid
