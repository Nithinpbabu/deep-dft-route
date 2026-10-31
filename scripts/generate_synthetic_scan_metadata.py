#!/usr/bin/env python3
"""
CLI tool for generating deterministic synthetic scan metadata (Stage 3d Nangate45 fallback).

Usage:
  PYTHONPATH=src python3 scripts/generate_synthetic_scan_metadata.py \
    --run-dir runs/uart_tx_nangate45_20261011_064013 \
    --output data/processed/samples/uart_tx_nangate45_20261011_064013/scan_metadata.json \
    --chain-count 1 \
    --ordering row_major_by_y_then_x \
    --summary-text
"""

import argparse
import sys
from pathlib import Path

from dft_hotspot.dft.synthetic_scan import (
    NoSequentialCellsFoundError,
    SyntheticScanGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic scan metadata from placed sequential cells."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to physical flow run directory (e.g. runs/uart_tx_nangate45_...).",
    )
    parser.add_argument(
        "--def-path",
        type=str,
        default=None,
        help="Optional explicit path to target DEF file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination path for generated scan_metadata.json.",
    )
    parser.add_argument(
        "--chain-count",
        type=int,
        default=1,
        help="Number of synthetic scan chains to construct (default: 1).",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        choices=["row_major_by_y_then_x", "nearest_neighbor"],
        default="row_major_by_y_then_x",
        help="Deterministic sequential cell ordering method (default: row_major_by_y_then_x).",
    )
    parser.add_argument(
        "--partition-method",
        type=str,
        choices=["contiguous_chunks", "round_robin"],
        default="contiguous_chunks",
        help="Deterministic partition method across chains (default: contiguous_chunks).",
    )
    parser.add_argument(
        "--netlist-path",
        type=str,
        default=None,
        help="Optional explicit path to target netlist file.",
    )
    parser.add_argument(
        "--design",
        type=str,
        default=None,
        help="Optional explicit design name override.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional explicit platform/PDK name override.",
    )
    parser.add_argument(
        "--summary-text",
        nargs="?",
        const="",
        default=None,
        help="Optionally write human-readable summary text file (adjacent to output JSON or custom path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.run_dir and not args.def_path:
        print("ERROR: Either --run-dir or --def-path must be specified.", file=sys.stderr)
        return 1

    out_path = Path(args.output)

    print("=" * 60)
    print(" Synthetic Scan Metadata Generator (Nangate45 Fallback)")
    print("=" * 60)
    print("[WARNING] Metadata is a SYNTHETIC PROXY only.")
    print("          No scan replacement or ATPG was performed.")
    print("          Use only for scan-chain routing demand proxy experiments.")
    print("-" * 60)

    try:
        metadata = SyntheticScanGenerator.generate(
            run_dir=args.run_dir,
            def_path=args.def_path,
            netlist_path=args.netlist_path,
            chain_count=args.chain_count,
            ordering=args.ordering,
            partition_method=args.partition_method,
            design=args.design,
            platform=args.platform,
        )

        saved_json = metadata.save_json(out_path)

        summary_out_path = None
        if args.summary_text is not None:
            if args.summary_text.strip():
                summary_out_path = Path(args.summary_text)
            else:
                summary_out_path = out_path.parent / (out_path.stem + "_summary.txt")
            summary_out_path.parent.mkdir(parents=True, exist_ok=True)
            summary_out_path.write_text(metadata.to_summary_text(), encoding="utf-8")

        print(f"Source DEF Path:         {metadata.def_path}")
        print(f"Design:                  {metadata.design}")
        print(f"Platform:                {metadata.platform}")
        print(f"Sequential Cells Found:  {metadata.statistics.total_scan_cells}")
        print(f"Chain Count:             {metadata.statistics.chain_count}")
        print(f"Ordering Method:         {metadata.synthetic_config.ordering if metadata.synthetic_config else 'N/A'}")
        print(f"Total Scan Cells:        {metadata.statistics.total_scan_cells}")
        print(f"Placed Scan Cells:       {metadata.statistics.placed_scan_cells}")
        print(f"JSON Output Path:        {saved_json}")
        if summary_out_path:
            print(f"Summary Text Path:       {summary_out_path}")
        print("-" * 60)
        print("Status:                  PASS")
        print("=" * 60)
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("-" * 60)
        print("Status:                  FAIL")
        print("=" * 60)
        return 1
    except NoSequentialCellsFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("-" * 60)
        print("Status:                  FAIL")
        print("=" * 60)
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected exception during synthetic scan generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("-" * 60)
        print("Status:                  FAIL")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
