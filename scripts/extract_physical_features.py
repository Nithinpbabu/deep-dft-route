#!/usr/bin/env python3
"""
CLI Script: Extract Basic Physical Features from DEF Layout Run.
Stage 2a of the DFT-Aware Physical Design Hotspot Prediction framework.

Usage:
    python scripts/extract_physical_features.py \
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

from dft_hotspot.features.extractor import PhysicalFeatureExtractor


def find_def_file(run_dir: Path, explicit_def: Optional[str] = None) -> Path:
    """
    Locates the target DEF file from run directory or explicit path.
    """
    if explicit_def:
        path = Path(explicit_def)
        if not path.is_file():
            # Try relative to run_dir
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

    raise FileNotFoundError(f"No .def file found within run directory: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract spatial physical features into 2D tensor and sample metadata."
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

    print("=" * 70)
    print("STAGE 2a: BASIC PHYSICAL FEATURE EXTRACTION")
    print("=" * 70)
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
    extractor = PhysicalFeatureExtractor(grid_size=args.grid_size)

    # Process and write sample
    print("\nParsing DEF and computing spatial feature channels...")
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
    design_stats = meta["design_statistics"]
    die = meta["die_area_um"]

    print("\n--- Physical Layout Metrics ---")
    print(f"Die Dimensions   : {die['width']:.1f} x {die['height']:.1f} µm (Area: {die['width']*die['height']:.1f} µm²)")
    print(f"Total Components : {design_stats['total_components']}")
    print(f"Sequential Cells : {design_stats['sequential_components']} ({design_stats['sequential_components']/max(design_stats['total_components'],1)*100:.1f}%)")
    print(f"Combinational    : {design_stats['combinational_components']}")
    print(f"Top-Level IO Pins: {design_stats['total_io_pins']}")
    print(f"Nets             : {design_stats['total_nets']}")
    print(f"Net Pin Conns    : {design_stats['total_net_pin_connections']}")

    print("\n--- Spatial Channel Summary (Tensor Shape: {}) ---".format(meta["tensor_shape"]))
    header = f"{'Idx':<4} {'Channel Name':<28} {'Raw Max':<10} {'Norm Max':<10} {'Active Bins':<12}"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        print(f"{s['channel_index']:<4} {name:<28} {s['raw_max']:<10.1f} {s['norm_max']:<10.3f} {s['active_bins']:<12}")

    print("\n--- Generated Sample Artifacts ---")
    for k, p in artifacts.items():
        print(f"[{k.upper():<8}] {p}")

    print("\n[SUCCESS] Stage 2a physical feature extraction completed successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
