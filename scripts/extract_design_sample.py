#!/usr/bin/env python3
"""
CLI Script: Extract Unified Paired Design Sample (Stage 2a + Stage 2b).
Orchestrates physical layout feature extraction (intended to be pre-route-compatible)
and post-route routing label extraction for a given physical design run, producing a validated paired sample.

Usage:
    python scripts/extract_design_sample.py \
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

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.features.extractor import PhysicalFeatureExtractor
from dft_hotspot.labels.routing_extractor import RoutingLabelExtractor

# =============================================================================
# TOP-LEVEL CONFIGURATION VARIABLE
# =============================================================================
# FILTER: Set to True to filter out non-functional dummy cells (FDCAP, FILL, TAPCELL)
# from density and graph tensors. Controllable via CLI flag '--filter' or '--no-filter'.
FILTER = True


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

    candidates = [
        run_dir / "scripts" / "uart_tx_final.def",
        run_dir / "uart_tx_final.def",
    ]
    for c in candidates:
        if c.is_file():
            return c

    final_defs = list(run_dir.rglob("*_final.def"))
    if final_defs:
        return final_defs[0]

    all_defs = list(run_dir.rglob("*.def"))
    if all_defs:
        return all_defs[0]

    raise FileNotFoundError(f"No .def file found within run directory: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified builder: Extract paired physical layout features (Stage 2a) and routing labels (Stage 2b)."
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
        help="Path to write paired sample artifacts (e.g. data/processed/samples/uart_tx_nangate45_20261011_064013)",
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
    # --filter flag (controllable via variable FILTER = True or CLI argument)
    parser.add_argument(
        "--filter",
        dest="filter_dummy_cells",
        action=argparse.BooleanOptionalAction,
        default=FILTER,
        help="Filter non-functional dummy cells (FDCAP, FILL, TAPCELL) from density tensors (default: True)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[ERROR] Run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        def_path = find_def_file(run_dir, args.def_path)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1

    print("=" * 75)
    print("STAGE 2c: UNIFIED PAIRED DESIGN SAMPLE BUILDER")
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

    # 1. Stage 2a Feature Extraction
    print("\n--- [1/3] Extracting Stage 2a Physical Layout Feature Tensors (Pre-Route-Compatible Target) ---")
    feat_extractor = PhysicalFeatureExtractor(
        grid_size=args.grid_size,
        filter_dummy_cells=args.filter_dummy_cells,
    )
    feat_artifacts = feat_extractor.process_and_save_sample(
        def_path=def_path,
        output_dir=output_dir,
        source_run_dir=run_dir,
        design_name=design_name,
        pdk_name=pdk_name,
    )
    for k, p in feat_artifacts.items():
        print(f"  Generated [feat_{k}]: {p.name}")

    # 2. Stage 2b Label Extraction
    print("\n--- [2/3] Extracting Stage 2b Post-Route Routing Labels ---")
    label_extractor = RoutingLabelExtractor(grid_size=args.grid_size)
    label_artifacts = label_extractor.process_and_save_sample(
        def_path=def_path,
        output_dir=output_dir,
        source_run_dir=run_dir,
        design_name=design_name,
        pdk_name=pdk_name,
    )
    for k, p in label_artifacts.items():
        print(f"  Generated [lbl_{k}]: {p.name}")

    # 3. Stage 2c Paired Validation
    print("\n--- [3/3] Validating Paired Dataset Sample ---")
    try:
        sample = PhysicalDesignSample.load(sample_dir=output_dir, validate=True)
    except Exception as err:
        print(f"[ERROR] Paired sample validation failed: {err}", file=sys.stderr)
        return 1

    print(f"  Sample ID        : {sample.sample_id}")
    print(f"  Feature Tensor   : {sample.feature_shape} (Normalized float32, {sample.num_feature_channels} ch)")
    print(f"  Label Tensor     : {sample.label_shape} (Normalized float32, {sample.num_label_channels} ch)")
    print(f"  Grid Alignment   : PASS ({args.grid_size} x {args.grid_size})")
    print(f"  Finite Values    : PASS (np.isfinite == True)")

    print("\n[SUCCESS] Unified paired design sample extracted and validated successfully.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
