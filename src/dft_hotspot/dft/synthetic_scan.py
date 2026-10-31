"""
Deterministic Synthetic Scan Metadata Fallback for Nangate45 (Stage 3d).

Generates deterministic synthetic scan-chain metadata for designs where standard cell libraries
lack OpenROAD-compatible test_cell() annotations (such as default Nangate45 Liberty).
Extracts placed sequential cells from layout DEF components, orders them deterministically,
and packages them under schema version scan_metadata_v001 with source="synthetic_proxy".
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_hotspot.dft.scan_metadata import (
    ScanCell,
    ScanChain,
    ScanEndpoint,
    ScanMetadata,
    ScanPorts,
    ScanStatistics,
    SyntheticConfig,
)
from dft_hotspot.features.cell_classifier import is_sequential
from dft_hotspot.features.def_parser import DefComponent, DefData, DefParser


class SyntheticScanError(Exception):
    """Base exception for synthetic scan extraction errors."""
    pass


class NoSequentialCellsFoundError(SyntheticScanError, ValueError):
    """Raised when no sequential cells are found in the target DEF file."""
    pass


DEFAULT_SYNTHETIC_LIMITATIONS = [
    "Synthetic proxy scan metadata only; no scan replacement or ATPG was performed.",
    "Generated from placed sequential cells in Nangate45 DEF.",
    "Scan pins are unknown because cells were not replaced with real scan flops.",
    "Use only for scan-chain routing demand proxy experiments.",
]


class SyntheticScanGenerator:
    """
    Generator of deterministic synthetic scan chains from placed sequential cells.
    """

    @classmethod
    def extract_sequential_cells(
        cls,
        def_data: DefData,
        filter_heuristic: str = "sequential_master_heuristic",
    ) -> List[DefComponent]:
        """
        Extracts placed sequential cell components from DefData using cell classifier heuristics.
        Validates that all extracted sequential cells are placed with coordinates.
        """
        seq_cells: List[DefComponent] = []
        for comp in def_data.components:
            if is_sequential(comp.master):
                seq_cells.append(comp)

        if not seq_cells:
            raise NoSequentialCellsFoundError(
                f"No sequential cells found among {len(def_data.components)} DEF components."
            )

        # Validate placement coordinates and master
        for c in seq_cells:
            if not c.name:
                raise ValueError("Sequential component missing instance name.")
            if not c.master:
                raise ValueError(f"Sequential component '{c.name}' missing master cell name.")
            if c.x is None or c.y is None or c.status.upper() not in ("PLACED", "FIXED", "COVER"):
                raise ValueError(
                    f"Sequential component '{c.name}' is unplaced or missing placement coordinates "
                    f"(status='{c.status}', x={c.x}, y={c.y})."
                )

        return seq_cells

    @classmethod
    def order_cells(
        cls,
        cells: List[DefComponent],
        method: str = "row_major_by_y_then_x",
    ) -> List[DefComponent]:
        """
        Orders sequential cells deterministically according to the requested strategy.
        Supported methods:
          - 'row_major_by_y_then_x': sorts by y coordinate, then x coordinate, then instance name.
          - 'nearest_neighbor': greedy nearest-neighbor spatial traversal from lowest (y, x).
        """
        if method == "row_major_by_y_then_x":
            return sorted(cells, key=lambda c: (c.y, c.x, c.name))
        elif method == "nearest_neighbor":
            return cls._order_nearest_neighbor(cells)
        else:
            raise ValueError(
                f"Unsupported ordering method '{method}'. "
                f"Supported options: 'row_major_by_y_then_x', 'nearest_neighbor'."
            )

    @classmethod
    def _order_nearest_neighbor(cls, cells: List[DefComponent]) -> List[DefComponent]:
        """
        Deterministic greedy nearest neighbor ordering.
        Starts at cell with lowest (y, x, name), repeatedly visiting closest unvisited cell.
        Ties are broken deterministically by (y, x, name).
        """
        if not cells:
            return []

        remaining = list(cells)
        # Start at deterministically lowest coordinate
        remaining.sort(key=lambda c: (c.y, c.x, c.name))
        current = remaining.pop(0)
        ordered = [current]

        while remaining:
            best_idx = 0
            best_dist = float("inf")
            best_key = (float("inf"), float("inf"), "")

            for idx, cand in enumerate(remaining):
                dx = cand.x - current.x
                dy = cand.y - current.y
                dist = math.hypot(dx, dy)
                cand_key = (cand.y, cand.x, cand.name)

                if dist < best_dist or (math.isclose(dist, best_dist) and cand_key < best_key):
                    best_dist = dist
                    best_key = cand_key
                    best_idx = idx

            current = remaining.pop(best_idx)
            ordered.append(current)

        return ordered

    @classmethod
    def partition_cells(
        cls,
        cells: List[DefComponent],
        chain_count: int = 1,
        method: str = "contiguous_chunks",
    ) -> List[List[DefComponent]]:
        """
        Partitions ordered cells into chain_count sub-lists deterministically.
        Supported methods:
          - 'contiguous_chunks': contiguous slices of ordered cells, balanced evenly.
          - 'round_robin': cell j assigned to chain j % chain_count.
        """
        if chain_count <= 0:
            raise ValueError(f"chain_count must be at least 1, got {chain_count}")
        if chain_count > len(cells):
            raise ValueError(
                f"chain_count ({chain_count}) cannot exceed total sequential cells ({len(cells)})"
            )

        if chain_count == 1:
            return [list(cells)]

        if method == "contiguous_chunks":
            total = len(cells)
            base_size = total // chain_count
            remainder = total % chain_count
            chains: List[List[DefComponent]] = []
            start_idx = 0
            for i in range(chain_count):
                extra = 1 if i < remainder else 0
                end_idx = start_idx + base_size + extra
                chains.append(cells[start_idx:end_idx])
                start_idx = end_idx
            return chains
        elif method == "round_robin":
            chains = [[] for _ in range(chain_count)]
            for idx, c in enumerate(cells):
                chains[idx % chain_count].append(c)
            return chains
        else:
            raise ValueError(
                f"Unsupported partition method '{method}'. "
                f"Supported options: 'contiguous_chunks', 'round_robin'."
            )

    @classmethod
    def build_chains(
        cls,
        partitioned_cells: List[List[DefComponent]],
        dbu: float = 1000.0,
    ) -> List[ScanChain]:
        """
        Constructs ScanChain and ScanCell objects from partitioned DefComponent groups.
        """
        chains: List[ScanChain] = []
        for chain_idx, group in enumerate(partitioned_cells):
            scan_cells: List[ScanCell] = []
            for cell_idx, comp in enumerate(group):
                x_val = int(comp.x) if comp.x.is_integer() else comp.x
                y_val = int(comp.y) if comp.y.is_integer() else comp.y
                scan_cells.append(
                    ScanCell(
                        index=cell_idx,
                        instance=comp.name,
                        scan_in_pin=None,
                        scan_out_pin=None,
                        master=comp.master,
                        x=x_val,
                        y=y_val,
                        x_um=round(comp.x / dbu, 4),
                        y_um=round(comp.y / dbu, 4),
                        placed=True,
                    )
                )

            chain_id = f"synthetic_chain_{chain_idx}"
            start_ep = ScanEndpoint(type="SYNTHETIC", name=f"synthetic_scan_in_{chain_idx}")
            stop_ep = ScanEndpoint(type="SYNTHETIC", name=f"synthetic_scan_out_{chain_idx}")
            chains.append(
                ScanChain(
                    chain_id=chain_id,
                    start=start_ep,
                    stop=stop_ep,
                    partition="synthetic_default",
                    cell_count=len(scan_cells),
                    cells=scan_cells,
                )
            )

        return chains

    @classmethod
    def generate(
        cls,
        run_dir: Optional[Path | str] = None,
        def_path: Optional[Path | str] = None,
        netlist_path: Optional[Path | str] = None,
        chain_count: int = 1,
        ordering: str = "row_major_by_y_then_x",
        partition_method: str = "contiguous_chunks",
        design: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> ScanMetadata:
        """
        Generates deterministic synthetic scan metadata from a run directory or DEF path.
        """
        # 1. Resolve DEF path
        if def_path:
            actual_def_path = Path(def_path)
            if not actual_def_path.is_file():
                raise FileNotFoundError(f"DEF file not found: {def_path}")
        elif run_dir:
            r_path = Path(run_dir)
            if not r_path.is_dir():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
            candidates = []
            if (r_path / "scripts").is_dir():
                candidates.extend(list((r_path / "scripts").glob("*_final.def")))
            candidates.extend(list(r_path.glob("*_final.def")))
            if (r_path / "scripts").is_dir():
                candidates.extend(list((r_path / "scripts").glob("*.def")))
            candidates.extend(list(r_path.glob("*.def")))
            if not candidates:
                raise FileNotFoundError(f"No DEF file found in run directory: {run_dir}")
            actual_def_path = candidates[0]
        else:
            raise ValueError("Either run_dir or def_path must be provided.")

        # 2. Resolve netlist path
        actual_netlist_path: Optional[Path] = None
        if netlist_path:
            actual_netlist_path = Path(netlist_path) if Path(netlist_path).is_file() else None
        elif run_dir:
            r_path = Path(run_dir)
            v_candidates = []
            if (r_path / "scripts").is_dir():
                v_candidates.extend(list((r_path / "scripts").glob("synth_*.v")))
            v_candidates.extend(list(r_path.glob("synth_*.v")))
            if (r_path / "scripts").is_dir():
                v_candidates.extend(list((r_path / "scripts").glob("*.v")))
            v_candidates.extend(list(r_path.glob("*.v")))
            if v_candidates:
                actual_netlist_path = v_candidates[0]

        # 3. Detect design and platform from manifest or DEF
        detected_design = design
        detected_platform = platform

        if run_dir:
            manifest_file = Path(run_dir) / "manifest.json"
            if manifest_file.is_file():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        if not detected_design:
                            detected_design = m_data.get("design")
                        if not detected_platform:
                            detected_platform = m_data.get("pdk")
                except Exception:
                    pass

        # Parse DEF content
        def_text = actual_def_path.read_text(encoding="utf-8", errors="replace")
        def_data = DefParser.parse_string(def_text)

        if not detected_design:
            design_m = re.search(r"DESIGN\s+([^\s;]+)", def_text)
            if design_m:
                detected_design = design_m.group(1)

        if not detected_platform:
            comb_str = f"{actual_def_path} {run_dir or ''}".lower()
            if "nangate" in comb_str:
                detected_platform = "nangate45"
            elif "sky130" in comb_str:
                detected_platform = "sky130hd"
            else:
                detected_platform = "nangate45"

        dbu = float(def_data.dbu_per_micron or 1000)

        # 4. Extract sequential cells
        seq_cells = cls.extract_sequential_cells(def_data)

        # 5. Order cells
        ordered_cells = cls.order_cells(seq_cells, method=ordering)

        # 6. Partition cells into chains
        partitioned = cls.partition_cells(
            ordered_cells,
            chain_count=chain_count,
            method=partition_method,
        )

        # 7. Build ScanChain objects
        chains = cls.build_chains(partitioned, dbu=dbu)

        # 8. Compute Statistics
        total_cells = sum(c.cell_count for c in chains)
        chain_lens = [c.cell_count for c in chains]
        placed_cells = sum(sum(1 for cell in c.cells if cell.placed) for c in chains)

        stats = ScanStatistics(
            chain_count=len(chains),
            total_scan_cells=total_cells,
            min_chain_length=min(chain_lens) if chain_lens else 0,
            max_chain_length=max(chain_lens) if chain_lens else 0,
            avg_chain_length=round(total_cells / len(chains), 4) if chains else 0.0,
            placed_scan_cells=placed_cells,
        )

        # 9. Validation contracts
        if stats.total_scan_cells != len(seq_cells):
            raise ValueError(
                f"Total scan cells ({stats.total_scan_cells}) does not match "
                f"extracted sequential cell count ({len(seq_cells)})."
            )
        if sum(c.cell_count for c in chains) != stats.total_scan_cells:
            raise ValueError("Sum of chain cell counts does not equal total_scan_cells.")

        # 10. Synthetic Config and Limitations
        notes = [
            f"Synthetic proxy generated because "
            f"{'Nangate45' if 'nangate' in (detected_platform or '').lower() else (detected_platform or 'Target')} "
            f"Liberty lacks OpenROAD-compatible test_cell annotations."
        ]
        if chain_count > 1:
            notes.append(f"Partition method: {partition_method} across {chain_count} chains.")

        synth_config = SyntheticConfig(
            chain_count=chain_count,
            ordering=ordering,
            source_cell_filter="sequential_master_heuristic",
            notes=notes,
            partition_method=partition_method if chain_count > 1 else None,
        )

        plat_name = "Nangate45" if "nangate" in (detected_platform or "").lower() else (detected_platform or "design")
        limitations = [
            "Synthetic proxy scan metadata only; no scan replacement or ATPG was performed.",
            f"Generated from placed sequential cells in {plat_name} DEF.",
            "Scan pins are unknown because cells were not replaced with real scan flops.",
            "Use only for scan-chain routing demand proxy experiments.",
        ]

        scan_ports = ScanPorts(
            scan_enable=[],
            scan_in=[],
            scan_out=[],
        )

        return ScanMetadata(
            schema_version="scan_metadata_v001",
            source="synthetic_proxy",
            design=detected_design or "unknown",
            platform=detected_platform or "unknown",
            source_dir=str(run_dir).replace("\\", "/") if run_dir else "",
            def_path=str(actual_def_path).replace("\\", "/"),
            netlist_path=str(actual_netlist_path).replace("\\", "/") if actual_netlist_path else None,
            synthetic_config=synth_config,
            scan_ports=scan_ports,
            chains=chains,
            statistics=stats,
            limitations=limitations,
        )
