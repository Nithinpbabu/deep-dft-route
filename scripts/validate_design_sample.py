#!/usr/bin/env python3
"""
CLI Script: Validate Paired Physical Design Sample.
Stage 2c of the DFT-Aware Physical Design Hotspot Prediction framework.

Verifies:
1. Presence of all 4 required artifacts (features_spatial.npz, metadata.json,
   labels_routing.npz, routing_label_metadata.json).
2. Spatial grid alignment between features and labels (H x W).
3. Tensor dimensionality (C x H x W) and channel name count matching.
4. Finite value verification (no NaN or Inf).
5. Source run directory and design consistency across metadatas.

Usage:
    python scripts/validate_design_sample.py \
        --sample-dir data/processed/samples/uart_tx_nangate45_20261011_064013
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dft_hotspot.data.dataset import PhysicalDesignSample


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a paired physical design sample (Stage 2a features + Stage 2b labels + optional Stage 3e scan features)."
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        required=True,
        help="Path to processed sample directory (e.g. data/processed/samples/uart_tx_nangate45_20261011_064013)",
    )
    parser.add_argument(
        "--require-scan",
        action="store_true",
        default=False,
        help="Enforce that scan features (features_scan.npz and scan_feature_metadata.json) are present.",
    )

    args = parser.parse_args()
    sample_dir = Path(args.sample_dir).resolve()

    print("=" * 75)
    print("STAGE 2c/3f: PAIRED PHYSICAL + SCAN-AWARE DESIGN SAMPLE VALIDATION")
    print("=" * 75)
    print(f"Target Sample Directory : {sample_dir}")

    if not sample_dir.is_dir():
        print(f"\n[ERROR] Sample directory does not exist: {sample_dir}", file=sys.stderr)
        print("\n[RESULT] FAIL")
        print("=" * 75)
        return 1

    try:
        sample = PhysicalDesignSample.load(sample_dir=sample_dir, validate=True)
    except FileNotFoundError as err:
        print(f"\n[ERROR] Missing required sample file(s): {err}", file=sys.stderr)
        print("\n[RESULT] FAIL")
        print("=" * 75)
        return 1
    except ValueError as err:
        print(f"\n[ERROR] Sample validation failed: {err}", file=sys.stderr)
        print("\n[RESULT] FAIL")
        print("=" * 75)
        return 1
    except Exception as err:
        print(f"\n[ERROR] Unexpected error while loading sample: {err}", file=sys.stderr)
        print("\n[RESULT] FAIL")
        print("=" * 75)
        return 1

    if args.require_scan and not sample.has_scan_features:
        print("\n[ERROR] Validation failed: --require-scan specified, but scan features are missing from sample.", file=sys.stderr)
        print("\n[RESULT] FAIL")
        print("=" * 75)
        return 1

    summary = sample.summary()
    grid_h, grid_w = sample.grid_shape

    print(f"Sample ID               : {sample.sample_id}")
    print(f"Design / PDK            : {sample.design or 'N/A'} / {sample.pdk or 'N/A'}")
    print(f"Source Run Directory    : {sample.source_run_dir or 'N/A'}")
    print(f"Spatial Grid Resolution : {grid_h} x {grid_w}")
    print(f"Feature Tensor Shape    : {sample.feature_shape} (Normalized float32)")
    print(f"Label Tensor Shape      : {sample.label_shape} (Normalized float32)")

    if sample.has_scan_features:
        print(f"Scan Tensor Shape       : {sample.scan_feature_shape} (Normalized float32, source: {sample.scan_source or 'unknown'})")
        print(f"Fused Input Shape       : {sample.get_fused_input_tensor().shape} (Physical + Scan concatenated)")
    else:
        print(f"Scan Feature Status     : Not Present (Physical-only baseline)")
        print(f"Fused Input Shape       : {sample.get_fused_input_tensor().shape} (Physical-only)")

    print("\n--- Feature Channels (Input: Physical Layout Features, Pre-Route Compatible) ---")
    for idx, name in enumerate(sample.feature_channel_names):
        print(f"  [{idx}] {name}")

    if sample.has_scan_features:
        print(f"\n--- Scan Channels (Input: Scan-Aware Layout Features, Source: {sample.scan_source or 'unknown'}) ---")
        for idx, name in enumerate(sample.scan_feature_channel_names):
            print(f"  [{idx}] {name}")

    print("\n--- Label Channels (Ground Truth) ---")
    for idx, name in enumerate(sample.label_channel_names):
        print(f"  [{idx}] {name}")

    print("\n--- Validation Breakdown ---")
    print(f"  [1] Artifact Completeness : PASS (All required NPZ/JSON files present)")
    scan_grid_str = f" == Scan {grid_h}x{grid_w}" if sample.has_scan_features else ""
    print(f"  [2] Grid Alignment        : PASS (Feature {grid_h}x{grid_w} == Label {grid_h}x{grid_w}{scan_grid_str})")
    print(f"  [3] Finite Values         : PASS (All values finite across all tensors; np.isfinite == True)")
    scan_ch_str = f", Scan: {sample.num_scan_channels} ch" if sample.has_scan_features else ""
    print(f"  [4] Channel Matching      : PASS (Features: {sample.num_feature_channels} ch, Labels: {sample.num_label_channels} ch{scan_ch_str})")
    print(f"  [5] Metadata Consistency  : PASS (Grid & source run IDs congruent)")
    if sample.has_scan_features:
        print(f"  [6] Scan Feature Contract : PASS (Scan source: {sample.scan_source}, Fused shape: {sample.get_fused_input_tensor().shape})")
    else:
        print(f"  [6] Scan Feature Status   : OPTIONAL (Absent; Physical-only baseline valid)")

    die = sample.die_area_um
    if die:
        print(f"\n--- Physical Layout Context ---")
        print(f"  Die Dimensions : {die.get('width', 0.0):.1f} x {die.get('height', 0.0):.1f} µm")
        d_stats = summary.get("design_statistics", {})
        r_stats = summary.get("routing_statistics", {})
        if d_stats:
            print(f"  Components     : {d_stats.get('total_components')} (Seq: {d_stats.get('sequential_components')}, Comb: {d_stats.get('combinational_components')})")
        if r_stats:
            print(f"  Routing        : {r_stats.get('routed_nets')}/{r_stats.get('total_nets')} nets routed, {r_stats.get('total_wirelength_um', 0.0):.1f} µm wirelength, {r_stats.get('total_vias')} vias")

    print("\n[RESULT] PASS")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
