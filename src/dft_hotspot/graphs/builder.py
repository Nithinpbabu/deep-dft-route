"""
Circuit and Scan Graph Builder (Stage 6a).

Parses placed DEF layouts and structured scan metadata into a unified CircuitGraph
combining cell physical node features, netlist functional edges, and scan-chain links.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dft_hotspot.dft.scan_metadata import ScanMetadata
from dft_hotspot.features.cell_classifier import CellClassifier
from dft_hotspot.features.def_parser import DefData, DefDieArea, DefParser
from dft_hotspot.graphs.schema import (
    CircuitGraph,
    EDGE_FEATURE_NAMES,
    EDGE_TYPE_NETLIST,
    EDGE_TYPE_SCAN_FORWARD,
    EDGE_TYPE_SCAN_REVERSE,
    NODE_FEATURE_NAMES,
)


class CircuitGraphBuilder:
    """
    Constructs a CircuitGraph from physical DEF data and optional scan metadata.
    """

    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]] = 64,
        max_net_degree: int = 30,
        include_scan_edges: bool = True,
        bidirectional_scan: bool = True,
        cell_classifier: Optional[CellClassifier] = None,
    ):
        """
        Args:
            grid_size: Spatial grid dimensions (H, W) or single integer H=W for coordinate binning.
            max_net_degree: Maximum fanout / pin connection count for forming netlist cliques.
                            Nets with more pins than this (e.g. global clock/reset) are skipped
                            from dense clique expansion to avoid O(N^2) edge explosion.
            include_scan_edges: Whether to insert explicit scan chain stitch edges.
            bidirectional_scan: If True, adds both forward (i -> i+1) and reverse (i+1 -> i) scan edges.
            cell_classifier: Custom CellClassifier or None for default heuristic.
        """
        if isinstance(grid_size, int):
            self.grid_h = grid_size
            self.grid_w = grid_size
        else:
            self.grid_h, self.grid_w = grid_size

        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({self.grid_h}, {self.grid_w})")

        self.max_net_degree = max_net_degree
        self.include_scan_edges = include_scan_edges
        self.bidirectional_scan = bidirectional_scan
        self.classifier = cell_classifier or CellClassifier()

    def build_from_files(
        self,
        def_path: Union[str, Path],
        scan_metadata_path: Optional[Union[str, Path]] = None,
        design: Optional[str] = None,
        sample_id: Optional[str] = None,
        pdk: Optional[str] = None,
    ) -> CircuitGraph:
        """
        Constructs CircuitGraph directly from DEF file path and optional scan metadata JSON path.
        """
        def_file = Path(def_path)
        if not def_file.is_file():
            raise FileNotFoundError(f"DEF file not found: {def_file}")

        def_data = DefParser.parse_file(def_file)

        scan_meta: Optional[ScanMetadata] = None
        if scan_metadata_path is not None:
            sm_file = Path(scan_metadata_path)
            if sm_file.is_file():
                scan_meta = ScanMetadata.load_json(sm_file)

        return self.build(
            def_data=def_data,
            scan_metadata=scan_meta,
            design=design,
            sample_id=sample_id,
            pdk=pdk,
        )

    def build(
        self,
        def_data: DefData,
        scan_metadata: Optional[ScanMetadata] = None,
        design: Optional[str] = None,
        sample_id: Optional[str] = None,
        pdk: Optional[str] = None,
    ) -> CircuitGraph:
        """
        Builds a CircuitGraph from pre-parsed DefData and optional ScanMetadata objects.
        """
        components = def_data.components
        num_components = len(components)
        if num_components == 0:
            # Handle empty design gracefully
            return CircuitGraph(
                node_features=np.empty((0, len(NODE_FEATURE_NAMES)), dtype=np.float32),
                edge_index=np.empty((2, 0), dtype=np.int64),
                edge_features=np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
                node_names=[],
                node_positions=np.empty((0, 2), dtype=np.float32),
                node_positions_dbu=np.empty((0, 2), dtype=np.float32),
                node_grid_coords=np.empty((0, 2), dtype=np.int64),
                metadata={
                    "design": design or "unknown",
                    "sample_id": sample_id or "unknown",
                    "pdk": pdk or "unknown",
                    "num_nodes": 0,
                    "num_edges": 0,
                },
            )

        # 1. Resolve Die Area Bounds
        die_area = def_data.die_area
        if die_area is not None:
            x_min, y_min = float(die_area.x_min), float(die_area.y_min)
            x_max, y_max = float(die_area.x_max), float(die_area.y_max)
            width = max(float(die_area.width), 1.0)
            height = max(float(die_area.height), 1.0)
            dbu_per_micron = def_data.dbu_per_micron
        else:
            all_x = [c.x for c in components]
            all_y = [c.y for c in components]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            width = max(x_max - x_min, 1.0)
            height = max(y_max - y_min, 1.0)
            dbu_per_micron = def_data.dbu_per_micron

        # 2. Map Scan Metadata
        # Structure: inst_name -> (chain_index, cell_order_index, total_cells_in_chain)
        scan_cell_info: Dict[str, Tuple[int, int, int]] = {}
        total_chains = 0
        if scan_metadata is not None and scan_metadata.chains:
            total_chains = len(scan_metadata.chains)
            for chain_idx, chain in enumerate(scan_metadata.chains):
                chain_len = len(chain.cells)
                for cell in chain.cells:
                    scan_cell_info[cell.instance] = (chain_idx, cell.index, chain_len)

        # 3. Extract Instance Pin Counts and Net Connectivity
        inst_pin_counts = def_data.get_instance_pin_counts()

        # 4. Construct Node Features and Coordinate Mappings
        node_names: List[str] = []
        node_features_list: List[List[float]] = []
        node_positions_list: List[Tuple[float, float]] = []
        node_positions_dbu_list: List[Tuple[float, float]] = []
        node_grid_coords_list: List[Tuple[int, int]] = []
        node_to_idx: Dict[str, int] = {}

        for idx, comp in enumerate(components):
            node_names.append(comp.name)
            node_to_idx[comp.name] = idx

            # Normalized Coordinates [0.0, 1.0]
            norm_x = float(np.clip((comp.x - x_min) / width, 0.0, 1.0))
            norm_y = float(np.clip((comp.y - y_min) / height, 0.0, 1.0))
            node_positions_list.append((norm_x, norm_y))
            node_positions_dbu_list.append((float(comp.x), float(comp.y)))

            # 2D Grid Bin Index (r, c)
            grid_c = int(np.clip(norm_x * self.grid_w, 0, self.grid_w - 1))
            grid_r = int(np.clip(norm_y * self.grid_h, 0, self.grid_h - 1))
            node_grid_coords_list.append((grid_r, grid_c))

            # Cell Classification
            is_seq = 1.0 if self.classifier.is_sequential(comp.master) else 0.0
            is_comb = 1.0 if is_seq == 0.0 else 0.0

            # Scan Cell Information
            if comp.name in scan_cell_info:
                is_scan = 1.0
                c_idx, order_idx, c_len = scan_cell_info[comp.name]
                scan_chain_id = (c_idx + 1.0) / max(1.0, float(total_chains))
                scan_order_index = (
                    order_idx / max(1.0, float(c_len - 1)) if c_len > 1 else 0.0
                )
            else:
                is_scan = 0.0
                scan_chain_id = 0.0
                scan_order_index = 0.0

            # Pin & Fanout Metrics
            pin_count = float(inst_pin_counts.get(comp.name, 0))
            pin_degree = float(np.clip(pin_count / 10.0, 0.0, 1.0))
            relative_cell_area = 1.0
            fanout_degree = pin_degree  # Proxy fanout degree

            node_features_list.append([
                norm_x,
                norm_y,
                is_seq,
                is_comb,
                is_scan,
                scan_chain_id,
                scan_order_index,
                pin_degree,
                relative_cell_area,
                fanout_degree,
            ])

        # 5. Construct Edges
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_attr: List[List[float]] = []

        total_span = width + height
        num_netlist_edges = 0
        num_scan_edges = 0

        # A. Netlist Connectivity Edges (Clique Expansion per Signal Net)
        for net in def_data.nets:
            # Gather unique cell instances on this net
            net_insts = list({
                inst for inst, _ in net.connections
                if inst in node_to_idx
            })
            k = len(net_insts)
            if 2 <= k <= self.max_net_degree:
                weight = 1.0 / float(k - 1)
                for i in range(k):
                    u_name = net_insts[i]
                    u_idx = node_to_idx[u_name]
                    ux, uy = node_positions_dbu_list[u_idx]

                    for j in range(i + 1, k):
                        v_name = net_insts[j]
                        v_idx = node_to_idx[v_name]
                        vx, vy = node_positions_dbu_list[v_idx]

                        # Normalized Manhattan Distance
                        manhattan_dbu = abs(ux - vx) + abs(uy - vy)
                        norm_dist = float(np.clip(manhattan_dbu / total_span, 0.0, 1.0))

                        # Bidirectional Netlist Edge (u -> v and v -> u)
                        edge_src.extend([u_idx, v_idx])
                        edge_dst.extend([v_idx, u_idx])
                        edge_feat = [EDGE_TYPE_NETLIST, 0.0, norm_dist, weight]
                        edge_attr.extend([edge_feat, edge_feat])
                        num_netlist_edges += 2

        # B. Scan Chain Stitch Edges
        if self.include_scan_edges and scan_metadata is not None:
            for chain in scan_metadata.chains:
                cells = chain.cells
                for i in range(len(cells) - 1):
                    u_name = cells[i].instance
                    v_name = cells[i + 1].instance

                    if u_name in node_to_idx and v_name in node_to_idx:
                        u_idx = node_to_idx[u_name]
                        v_idx = node_to_idx[v_name]
                        ux, uy = node_positions_dbu_list[u_idx]
                        vx, vy = node_positions_dbu_list[v_idx]

                        manhattan_dbu = abs(ux - vx) + abs(uy - vy)
                        norm_dist = float(np.clip(manhattan_dbu / total_span, 0.0, 1.0))

                        # Forward Scan Edge (u -> v)
                        edge_src.append(u_idx)
                        edge_dst.append(v_idx)
                        edge_attr.append([EDGE_TYPE_SCAN_FORWARD, 1.0, norm_dist, 1.0])
                        num_scan_edges += 1

                        # Reverse Scan Edge (v -> u)
                        if self.bidirectional_scan:
                            edge_src.append(v_idx)
                            edge_dst.append(u_idx)
                            edge_attr.append([EDGE_TYPE_SCAN_REVERSE, 1.0, norm_dist, 1.0])
                            num_scan_edges += 1

        # 6. Assemble Final Arrays
        node_features = np.asarray(node_features_list, dtype=np.float32)
        node_positions = np.asarray(node_positions_list, dtype=np.float32)
        node_positions_dbu = np.asarray(node_positions_dbu_list, dtype=np.float32)
        node_grid_coords = np.asarray(node_grid_coords_list, dtype=np.int64)

        if edge_src:
            edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
            edge_features = np.asarray(edge_attr, dtype=np.float32)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_features = np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32)

        metadata: Dict[str, Any] = {
            "design": design or "unknown",
            "sample_id": sample_id or "unknown",
            "pdk": pdk or "unknown",
            "num_nodes": num_components,
            "num_edges": edge_index.shape[1],
            "num_netlist_edges": num_netlist_edges,
            "num_scan_edges": num_scan_edges,
            "scan_chains_count": total_chains,
            "scan_cells_count": len(scan_cell_info),
            "die_area_um": {
                "x_min": x_min / dbu_per_micron,
                "y_min": y_min / dbu_per_micron,
                "x_max": x_max / dbu_per_micron,
                "y_max": y_max / dbu_per_micron,
                "width": width / dbu_per_micron,
                "height": height / dbu_per_micron,
            },
            "grid_size": [self.grid_h, self.grid_w],
            "node_feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
        }

        return CircuitGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_names=node_names,
            node_positions=node_positions,
            node_positions_dbu=node_positions_dbu,
            node_grid_coords=node_grid_coords,
            metadata=metadata,
        )
