#!/usr/bin/env python3
"""
CLI Script: Extract Post-Route Routed Wire Density Labels From DEF Layout Run.
Stage 2b of the DFT-Aware Physical Design Hotspot Prediction framework.

Usage:
    python scripts/extract_routing_labels.py \
        --run-dir runs/uart_tx_nangate45_20261011_064013 \
        --grid-size 64 \
        --output-dir data/processed/samples/uart_tx_nangate45_20261011_064013
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure src is on Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dft_hotspot.labels.routing_extractor import RoutingLabelExtractor


def find_def_file(run_dir: Path, explicit_def: Optional[str] = None) -> Path:
    """
    Locates the target routed DEF file from run directory or explicit path.
    """
    if explicit_def:
        path = Path(explicit_def)
        if not path.is_file():
            path = run_dir / explicit_def
        if path.is_file():
            return path
        raise FileNotFoundError(f"Specified DEF file not found: {explicit_def}")

    # Standard candidate locations in run directory
    candidates = [
        run_dir / "scripts" / "uart_tx_final.def",
        run_dir / "uart_tx_final.def",
    ]

    for c in candidates:
        if c.is_file():
            return c

    # Search for any *_final.def or *.def
    final_defs = list(run_dir.rglob("*_final.def"))
    if final_defs:
        return final_defs[0]

    all_defs = list(run_dir.rglob("*.def"))
    if all_defs:
        return all_defs[0]

    raise FileNotFoundError(f"No routed .def file found within run directory: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract post-route routed wire density labels and via counts into 2D grid tensors."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to physical design flow run directory (e.g. runs/uart_tx_nangate45_20261011_064013)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to write processed sample artifacts (e.g. data/processed/samples/uart_tx_nangate45_20261011_064013)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Grid resolution (default: 64 for 64x64 grid)",
    )
    parser.add_argument(
        "--def-path",
        type=str,
        default=None,
        help="Optional explicit path to DEF file",
    )
    parser.add_argument(
        "--pdk",
        type=str,
        default=None,
        help="PDK identifier (default: inferred from manifest or 'nangate45')",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[ERROR] Run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()

    # Locate DEF
    try:
        def_path = find_def_file(run_dir, args.def_path)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1

    print("=" * 75)
    print("STAGE 2b: POST-ROUTE ROUTED WIRE & VIA LABEL EXTRACTION")
    print("=" * 75)
    print(f"Run Directory  : {run_dir}")
    print(f"DEF Layout File: {def_path}")
    print(f"Grid Resolution: {args.grid_size} x {args.grid_size}")
    print(f"Output Target  : {output_dir}")

    # Inspect manifest if present
    design_name = None
    pdk_name = args.pdk
    manifest_file = run_dir / "manifest.json"
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            design_name = manifest.get("design")
            if pdk_name is None:
                pdk_name = manifest.get("pdk", "nangate45")
            print(f"Manifest Status: {manifest.get('status')} (Design: {design_name}, PDK: {pdk_name})")
        except Exception as err:
            print(f"[WARNING] Could not parse manifest.json: {err}")

    if pdk_name is None:
        pdk_name = "nangate45"

    # Instantiate extractor
    extractor = RoutingLabelExtractor(grid_size=args.grid_size)

    # Process and write sample
    print("\nParsing DEF routed nets and computing spatial label channels...")
    artifacts = extractor.process_and_save_sample(
        def_path=def_path,
        output_dir=output_dir,
        source_run_dir=run_dir,
        design_name=design_name,
        pdk_name=pdk_name,
    )

    # Read back metadata to display summary
    meta = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
    stats = meta["channel_statistics"]
    routing_stats = meta["routing_statistics"]
    die = meta["die_area_um"]

    print("\n--- Physical Routing Metrics ---")
    print(f"Die Dimensions     : {die['width']:.1f} x {die['height']:.1f} µm (Area: {die['width']*die['height']:.1f} µm²)")
    print(f"Total Nets         : {routing_stats['total_nets']}")
    print(f"Routed Nets        : {routing_stats['routed_nets']} ({routing_stats['routed_nets']/max(routing_stats['total_nets'], 1)*100:.1f}%)")
    print(f"Total Segments     : {routing_stats['total_segments']} (Horiz: {routing_stats['horizontal_segments']}, Vert: {routing_stats['vertical_segments']})")
    print(f"Total Vias         : {routing_stats['total_vias']}")
    print(f"Total Wirelength   : {routing_stats['total_wirelength_um']:.2f} µm ({routing_stats['total_wirelength_dbu']:.0f} DBU)")

    print("\n--- Layer Wirelength Breakdown ---")
    layer_wls = routing_stats.get("layer_wirelengths_um", {})
    layer_counts = routing_stats.get("layer_segment_counts", {})
    layer_vias = routing_stats.get("layer_via_counts", {})
    for lyr in sorted(set(list(layer_wls.keys()) + list(layer_counts.keys()))):
        wl = layer_wls.get(lyr, 0.0)
        sc = layer_counts.get(lyr, 0)
        vc = layer_vias.get(lyr, 0)
        print(f"  {lyr:<10}: {wl:>8.2f} µm ({sc:>4} segments, {vc:>4} vias)")

    print("\n--- Spatial Label Channel Summary (Tensor Shape: {}) ---".format(meta["tensor_shape"]))
    header = f"{'Idx':<4} {'Channel Name':<34} {'Raw Max':<10} {'Norm Max':<10} {'Active Bins':<12} {'Raw Sum':<12}"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        print(f"{s['channel_index']:<4} {name:<34} {s['raw_max']:<10.2f} {s['norm_max']:<10.3f} {s['active_bins']:<12} {s['raw_sum']:<12.1f}")

    print("\n--- Generated Sample Artifacts ---")
    for k, p in artifacts.items():
        print(f"[{k.upper():<8}] {p}")

    print("\n[SUCCESS] Stage 2b post-route routing label extraction completed successfully.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
