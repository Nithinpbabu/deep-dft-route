#!/usr/bin/env python3
"""
CLI Runner for EDA Flow Execution and Synthesis/P&R Orchestration.
Usage:
  python scripts/run_flow.py --config configs/flows/uart_nangate45.yaml [--dry-run]
"""

import sys
import argparse
import json
from pathlib import Path

# Add src/ to sys.path to ensure module loading when executed directly
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dft_hotspot.eda import FlowRunner
from dft_hotspot.utils import setup_logger

logger = setup_logger("run_flow_cli")

def main():
    parser = argparse.ArgumentParser(
        description="Run EDA Physical Design Flow for DFT-Aware Hotspot Prediction."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML flow configuration file."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry-run only: copy RTL, render Tcl/Makefile scripts, and output manifest without executing tools."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom root directory for flow runs."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for run directory (e.g. uart_tx_nangate45_u45_clk3p0)."
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f" Starting EDA Flow Run: {config_path.name}")
    print(f" Dry Run Mode: {args.dry_run}")
    print(f"{'=' * 60}\n")

    try:
        manifest = FlowRunner.run_flow(
            config_path=config_path,
            dry_run=args.dry_run,
            output_root=args.output_dir,
            run_name=args.run_name,
        )

        print("\nFlow Execution Summary:")
        print(f"  Design Name      : {manifest['design']}")
        print(f"  PDK Target       : {manifest['pdk']}")
        print(f"  Backend          : {manifest['flow_backend']}")
        print(f"  Run Directory    : {manifest['run_dir']}")
        print(f"  Status           : {manifest['status']}")
        print(f"  Generated Scripts: {', '.join(manifest['generated_scripts'])}")
        print(f"  Copied RTL       : {', '.join(manifest['copied_rtl'])}")
        
        if manifest["failure_reason"]:
            print(f"  Notice/Reason    : {manifest['failure_reason']}")

        print(f"\nManifest written to: {Path(manifest['run_dir']) / 'manifest.json'}")
        print(f"{'=' * 60}\n")

        if manifest["status"] in ["dry_run", "success", "blocked_missing_eda_env", "blocked_missing_pdk_assets"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error executing flow runner: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
