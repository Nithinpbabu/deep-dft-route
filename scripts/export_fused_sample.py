#!/usr/bin/env python3
"""
CLI Script: Export Fused Physical + Scan-Aware Input Sample.
Stage 3f of the DFT-Aware Physical Design Hotspot Prediction framework.

Exports a unified baseline model input tensor file (fused_input.npz)
concatenating physical layout features and scan-aware layout features:
  - fused_features: (12, H, W) normalized float32
  - raw_fused_features: (12, H, W) raw counts float32
  - channel_names: 12 prefixed names (physical/* and scan/*)
and an accompanying provenance metadata file (fused_input_metadata.json).

Usage:
    python scripts/export_fused_sample.py \\
        --sample-dir data/processed/samples/uart_tx_nangate45_20261011_064013 \\
        --output data/processed/samples/uart_tx_nangate45_20261011_064013/fused_input.npz
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# Ensure src is on Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from dft_hotspot.data.dataset import (
    FEATURE_NPZ_NAME,
    SCAN_NPZ_NAME,
    PhysicalDesignSample,
)


def export_fused_sample(
    sample_dir: Path,
    output_npz_path: Path,
    include_scan: bool = True,
) -> Path:
    """
    Loads sample and exports fused input NPZ and provenance metadata JSON.
    """
    sample = PhysicalDesignSample.load(sample_dir, validate=True, load_scan=include_scan)

    fused_features = sample.get_fused_input_tensor(include_scan=include_scan)
    channel_names = sample.get_fused_channel_names(include_scan=include_scan)

    if include_scan and sample.has_scan_features:
        assert sample.raw_scan_feature_tensor is not None
        raw_fused_features = np.concatenate(
            [sample.raw_feature_tensor, sample.raw_scan_feature_tensor],
            axis=0,
        ).astype(np.float32)
    else:
        raw_fused_features = sample.raw_feature_tensor.copy().astype(np.float32)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Save NPZ
    np.savez_compressed(
        output_npz_path,
        fused_features=fused_features,
        raw_fused_features=raw_fused_features,
        channel_names=channel_names,
    )

    # Save provenance metadata JSON
    meta_path = output_npz_path.with_name(f"{output_npz_path.stem}_metadata.json")
    metadata = {
        "schema_version": "fused_input_v001",
        "sample_id": sample.sample_id,
        "design": sample.design,
        "pdk": sample.pdk,
        "has_scan_features": sample.has_scan_features if include_scan else False,
        "scan_source": sample.scan_source if include_scan else None,
        "tensor_shape": list(fused_features.shape),
        "channel_count": len(channel_names),
        "channel_names": channel_names,
        "source_sample_dir": str(sample.sample_dir),
        "physical_feature_path": str(sample.sample_dir / FEATURE_NPZ_NAME),
        "scan_feature_path": str(sample.sample_dir / SCAN_NPZ_NAME) if (include_scan and sample.has_scan_features) else None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return output_npz_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export fused physical + scan-aware baseline input tensor (Stage 3f)."
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        required=True,
        help="Path to processed sample directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output NPZ path (default: <sample-dir>/fused_input.npz).",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        default=False,
        help="Exclude scan features even if present (export physical features only).",
    )

    args = parser.parse_args()
    sample_dir = Path(args.sample_dir).resolve()

    if not sample_dir.is_dir():
        print(f"[ERROR] Sample directory not found: {sample_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.output).resolve() if args.output else (sample_dir / "fused_input.npz")
    include_scan = not args.no_scan

    print("=" * 75)
    print("STAGE 3f: EXPORT FUSED INPUT TENSOR")
    print("=" * 75)
    print(f"Source Sample Directory : {sample_dir}")
    print(f"Output NPZ Path         : {out_path}")
    print(f"Include Scan Features   : {include_scan}")

    try:
        exported_path = export_fused_sample(sample_dir, out_path, include_scan=include_scan)
        meta_path = exported_path.with_name(f"{exported_path.stem}_metadata.json")
    except Exception as err:
        print(f"\n[ERROR] Export failed: {err}", file=sys.stderr)
        return 1

    print("\n--- Export Summary ---")
    print(f"  Fused NPZ File  : {exported_path}")
    print(f"  Metadata File   : {meta_path}")

    # Inspect exported file
    with np.load(exported_path, allow_pickle=False) as npz:
        fused = npz["fused_features"]
        channels = [str(c) for c in npz["channel_names"]]
        print(f"  Tensor Shape    : {fused.shape} (dtype: {fused.dtype})")
        print(f"  Channel Count   : {len(channels)}")
        for idx, ch in enumerate(channels):
            print(f"    [{idx:02d}] {ch}")

    print("\n[RESULT] SUCCESS")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
