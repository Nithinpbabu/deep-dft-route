#!/usr/bin/env python3
"""
CLI tool for extracting circuit and scan graphs (Stage 6a).

Extracts node features, netlist connectivity, and scan-chain links from DEF layout
and scan metadata, saving the resulting graph as PyTorch Geometric .pt and compressed .npz.

Usage:
  # Single sample:
  PYTHONPATH=src python3 scripts/extract_circuit_graph.py \
    --sample-dir data/processed/datasets/expanded_v002/samples/dft_counter_nangate45_u45_clk3p0_ar1p0_chains1

  # Entire dataset batch:
  PYTHONPATH=src python3 scripts/extract_circuit_graph.py \
    --dataset-dir data/processed/datasets/expanded_v002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from dft_hotspot.graphs.builder import CircuitGraphBuilder
from dft_hotspot.graphs.schema import CircuitGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract circuit and scan graphs for GNN modeling (Stage 6a)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample-dir",
        type=str,
        help="Path to an individual processed sample directory containing metadata.json.",
    )
    group.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to a processed dataset directory containing a samples/ folder.",
    )

    parser.add_argument(
        "--def-path",
        type=str,
        default=None,
        help="Optional explicit override for DEF file path.",
    )
    parser.add_argument(
        "--scan-metadata",
        type=str,
        default=None,
        help="Optional explicit override for scan_metadata.json path.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Spatial grid dimension (default: 64).",
    )
    parser.add_argument(
        "--max-net-degree",
        type=int,
        default=30,
        help="Maximum net pin degree to form full clique edges (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional destination directory for extracted graph files.",
    )
    parser.add_argument(
        "--format",
        choices=["pt", "npz", "both"],
        default="both",
        help="Output serialization format (default: both).",
    )
    return parser.parse_args()


def process_sample(
    sample_dir: Path,
    builder: CircuitGraphBuilder,
    output_dir: Optional[Path] = None,
    explicit_def: Optional[Path] = None,
    explicit_scan: Optional[Path] = None,
    save_format: str = "both",
) -> CircuitGraph:
    """Extracts and saves circuit graph for a single sample."""
    out_dir = output_dir or sample_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = sample_dir / "metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"metadata.json not found in sample dir: {sample_dir}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Resolve DEF path
    def_path: Optional[Path] = explicit_def
    if def_path is None:
        raw_def = meta.get("def_path")
        if raw_def:
            candidate = Path(raw_def)
            if candidate.is_file():
                def_path = candidate
            else:
                # Handle Windows / WSL cross-path mapping
                alt_str = str(raw_def).replace("/mnt/e/", "E:/")
                candidate_alt = Path(alt_str)
                if candidate_alt.is_file():
                    def_path = candidate_alt

    if def_path is None or not def_path.is_file():
        raise FileNotFoundError(f"DEF file could not be resolved for sample {sample_dir.name}")

    # 2. Resolve scan metadata path
    scan_meta_path: Optional[Path] = explicit_scan
    if scan_meta_path is None:
        candidate_scan = sample_dir / "scan_metadata.json"
        if candidate_scan.is_file():
            scan_meta_path = candidate_scan

    # 3. Extract target routing label for PyG target attribute (if available)
    routing_label_file = sample_dir / "labels_routing.npz"
    y_target = None
    if routing_label_file.is_file():
        try:
            import numpy as np
            with np.load(routing_label_file) as npz:
                if "labels" in npz:
                    y_target = npz["labels"]
        except Exception:
            pass

    # 4. Build Circuit Graph
    graph = builder.build_from_files(
        def_path=def_path,
        scan_metadata_path=scan_meta_path,
        design=meta.get("design", sample_dir.name),
        sample_id=sample_dir.name,
        pdk=meta.get("pdk", "nangate45"),
    )

    # 5. Save Graph Outputs
    if save_format in ("npz", "both"):
        npz_dest = out_dir / "circuit_graph.npz"
        graph.save_npz(npz_dest)

    if save_format in ("pt", "both"):
        pt_dest = out_dir / "circuit_graph.pt"
        try:
            graph.save_pt(pt_dest, y=y_target)
        except ImportError:
            print("[WARNING] PyTorch not available; skipped .pt save", file=sys.stderr)

    # Save summary metadata JSON
    summary_path = out_dir / "circuit_graph_metadata.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(graph.metadata, f, indent=2)

    return graph


def main() -> int:
    args = parse_args()

    builder = CircuitGraphBuilder(
        grid_size=args.grid_size,
        max_net_degree=args.max_net_degree,
        include_scan_edges=True,
        bidirectional_scan=True,
    )

    explicit_def = Path(args.def_path) if args.def_path else None
    explicit_scan = Path(args.scan_metadata) if args.scan_metadata else None
    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
        print(f"[*] Extracting circuit graph for sample: {sample_dir.name}...")
        graph = process_sample(
            sample_dir=sample_dir,
            builder=builder,
            output_dir=out_dir,
            explicit_def=explicit_def,
            explicit_scan=explicit_scan,
            save_format=args.format,
        )
        print(f"[+] Success: {graph.num_nodes} nodes, {graph.num_edges} edges "
              f"({graph.metadata['num_netlist_edges']} netlist, {graph.metadata['num_scan_edges']} scan).")
        return 0

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        samples_root = dataset_dir / "samples"
        if not samples_root.is_dir():
            print(f"[-] Error: samples/ directory not found in dataset: {dataset_dir}", file=sys.stderr)
            return 1

        sample_dirs = sorted([p for p in samples_root.iterdir() if p.is_dir()])
        print(f"[*] Processing batch of {len(sample_dirs)} samples in dataset: {dataset_dir.name}...")

        success_count = 0
        total_nodes = 0
        total_edges = 0

        for s_dir in sample_dirs:
            try:
                g = process_sample(
                    sample_dir=s_dir,
                    builder=builder,
                    save_format=args.format,
                )
                success_count += 1
                total_nodes += g.num_nodes
                total_edges += g.num_edges
                print(f"  [OK] {s_dir.name}: {g.num_nodes} nodes, {g.num_edges} edges")
            except Exception as e:
                print(f"  [FAIL] {s_dir.name}: {e}", file=sys.stderr)

        print(f"\n[+] Batch extraction complete: {success_count}/{len(sample_dirs)} samples extracted.")
        print(f"    Total graph nodes: {total_nodes:,} | Total graph edges: {total_edges:,}")
        return 0 if success_count == len(sample_dirs) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
