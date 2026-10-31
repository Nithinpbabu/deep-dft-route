#!/usr/bin/env python3
"""
CLI tool for extracting structured scan metadata from OpenROAD native DFT outputs.

Usage:
  PYTHONPATH=src python3 scripts/extract_scan_metadata.py \
    --dft-dir data/dft_validation/dft_smoke_sky130hd_20261011_185057 \
    --output data/dft_validation/dft_smoke_sky130hd_20261011_185057/scan_metadata.json \
    --summary-text
"""

import argparse
import sys
from pathlib import Path

from dft_hotspot.dft.scan_metadata import (
    ScanMetadataExtractor,
    NoScanChainsFoundError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured scan metadata from OpenROAD DFT outputs."
    )
    parser.add_argument(
        "--dft-dir",
        type=str,
        required=True,
        help="Path to DFT validation artifact directory (containing DEF, Verilog, logs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where scan_metadata.json will be saved.",
    )
    parser.add_argument(
        "--def-path",
        type=str,
        default=None,
        help="Optional explicit path to target DEF file.",
    )
    parser.add_argument(
        "--netlist-path",
        type=str,
        default=None,
        help="Optional explicit path to target Verilog netlist file.",
    )
    parser.add_argument(
        "--allow-missing-scanchains",
        action="store_true",
        default=False,
        help="Allow missing SCANCHAINS block without error (useful for negative control reporting).",
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
    dft_dir = Path(args.dft-dir if hasattr(args, "dft-dir") else args.dft_dir)
    out_path = Path(args.output)

    print("=" * 60)
    print(" OpenROAD DFT Scan Metadata Extractor")
    print("=" * 60)
    print(f"Target Directory: {dft_dir}")
    print(f"Output Target:    {out_path}")

    try:
        metadata = ScanMetadataExtractor.extract_from_directory(
            dft_dir=dft_dir,
            def_path=args.def_path,
            netlist_path=args.netlist_path,
            allow_missing_scanchains=args.allow_missing_scanchains,
        )

        # Save JSON output
        saved_json = metadata.save_json(out_path)

        # Optionally save summary text
        summary_out_path = None
        if args.summary_text is not None:
            if args.summary_text.strip():
                summary_out_path = Path(args.summary_text)
            else:
                summary_out_path = out_path.parent / (out_path.stem + "_summary.txt")
            summary_out_path.write_text(metadata.to_summary_text(), encoding="utf-8")

        # Required console report
        print(f"Source DEF Path:      {metadata.def_path}")
        print(f"Number of Chains:     {metadata.statistics.chain_count}")
        print(f"Total Scan Cells:     {metadata.statistics.total_scan_cells}")
        print(f"Placed Scan Cells:    {metadata.statistics.placed_scan_cells}")
        print(
            f"Scan Ports Detected:  enable={metadata.scan_ports.scan_enable}, "
            f"in={metadata.scan_ports.scan_in}, out={metadata.scan_ports.scan_out}"
        )
        print(f"JSON Output Path:     {saved_json}")
        if summary_out_path:
            print(f"Summary Text Path:    {summary_out_path}")
        print("-" * 60)
        print("Status:               PASS")
        print("=" * 60)
        return 0

    except NoScanChainsFoundError as e:
        print(f"ERROR: {e}")
        print("Hint: Use --allow-missing-scanchains if processing a negative control run.")
        print("-" * 60)
        print("Status:               FAIL")
        print("=" * 60)
        return 1
    except Exception as e:
        print(f"ERROR: Unhandled exception during extraction: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 60)
        print("Status:               FAIL")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
