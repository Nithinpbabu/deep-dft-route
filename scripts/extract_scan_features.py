#!/usr/bin/env python3
"""
CLI tool for extracting scan-aware spatial feature tensors (Stage 3e).

Usage:
  PYTHONPATH=src python3 scripts/extract_scan_features.py \
    --sample-dir data/processed/samples/uart_tx_nangate45_20261011_064013 \
    --grid-size 64
"""

import argparse
import sys
from pathlib import Path

from dft_hotspot.features.scan_features import ScanFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2D scan-aware spatial feature tensors from scan metadata."
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        required=True,
        help="Path to processed sample directory containing scan_metadata.json and metadata.json.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Spatial grid dimension (H=W, default: 64).",
    )
    parser.add_argument(
        "--scan-metadata",
        type=str,
        default=None,
        help="Optional explicit path to scan_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional destination directory (defaults to sample-dir).",
    )
    parser.add_argument(
        "--summary-text",
        nargs="?",
        const="",
        default=None,
        help="Optionally write human-readable summary text file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    out_dir = Path(args.output_dir) if args.output_dir else sample_dir

    if not sample_dir.is_dir():
        print(f"ERROR: Sample directory does not exist: {sample_dir}", file=sys.stderr)
        return 1

    scan_meta_path = Path(args.scan_metadata) if args.scan_metadata else sample_dir / "scan_metadata.json"
    if not scan_meta_path.is_file():
        print(f"ERROR: Scan metadata file not found: {scan_meta_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(" Scan-Aware Spatial Feature Extractor (Stage 3e)")
    print("=" * 60)

    try:
        extractor = ScanFeatureExtractor(grid_size=args.grid_size)
        results = extractor.process_and_save_sample(
            sample_dir=sample_dir,
            scan_metadata_path=scan_meta_path,
            output_dir=out_dir,
        )

        import json
        with open(results["metadata"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        scan_source = meta.get("scan_source", "unknown")
        if scan_source == "synthetic_proxy":
            print("[WARNING] Scan metadata is a SYNTHETIC PROXY only.")
            print("          No scan replacement or ATPG was performed.")
            print("          Use only for scan-chain routing demand proxy experiments.")
            print("-" * 60)

        print(f"Scan Metadata Path:      {scan_meta_path}")
        print(f"Scan Source:             {scan_source}")
        print(f"Design:                  {meta.get('design', 'unknown')}")
        print(f"Platform:                {meta.get('platform', 'unknown')}")
        print(f"Chain Count:             {meta.get('chain_count', 0)}")
        print(f"Total Scan Cells:        {meta.get('total_scan_cells', 0)}")
        print(f"Placed Scan Cells:       {meta.get('placed_scan_cells', 0)}")
        print(f"Segment Count:           {meta.get('segment_count', 0)}")
        print(f"Total Scan Wirelength:   {meta.get('total_scan_wirelength_um', 0.0):.2f} µm")
        print(f"Tensor Shape:            {meta.get('tensor_shape', [])}")
        print(f"Grid Size:               {args.grid_size} x {args.grid_size}")
        print(f"Feature Tensor Output:   {results['npz']}")
        print(f"Metadata Output:         {results['metadata']}")
        print(f"Preview Output:          {results['preview']}")

        # Optional summary text
        if args.summary_text is not None:
            summary_path = (
                Path(args.summary_text)
                if args.summary_text.strip()
                else out_dir / "scan_feature_summary.txt"
            )
            lines = [
                "=" * 60,
                " SCAN-AWARE FEATURE EXTRACTION SUMMARY",
                "=" * 60,
                f"Schema Version:          {meta.get('schema_version')}",
                f"Source Scan Metadata:    {meta.get('source_scan_metadata')}",
                f"Scan Source:             {scan_source}",
                f"Design:                  {meta.get('design')}",
                f"Platform:                {meta.get('platform')}",
                f"Grid Size:               {meta.get('grid_size')}",
                f"Tensor Shape:            {meta.get('tensor_shape')}",
                f"Chain Count:             {meta.get('chain_count')}",
                f"Total Scan Cells:        {meta.get('total_scan_cells')}",
                f"Segment Count:           {meta.get('segment_count')}",
                f"Total Scan Wirelength:   {meta.get('total_scan_wirelength_um'):.2f} µm",
                "-" * 60,
                "Channel Statistics:",
            ]
            for ch_name, ch_stat in meta.get("channel_statistics", {}).items():
                lines.append(
                    f"  - [{ch_stat.get('channel_index')}] {ch_name:35s}: "
                    f"raw_max={ch_stat.get('raw_max', 0):.2f}, "
                    f"raw_sum={ch_stat.get('raw_sum', 0):.2f}, "
                    f"active_bins={ch_stat.get('active_bins', 0)}"
                )
            if meta.get("limitations"):
                lines.append("-" * 60)
                lines.append("Limitations:")
                for lim in meta["limitations"]:
                    lines.append(f"  * {lim}")
            lines.append("=" * 60)
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Summary Text Path:       {summary_path}")

        print("-" * 60)
        print("Status:                  PASS")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("-" * 60)
        print("Status:                  FAIL")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
