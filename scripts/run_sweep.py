#!/usr/bin/env python3
"""
CLI Script: Automated Parameter Sweep Runner (Stage 4a).
Orchestrates multi-run physical design flows, feature extraction,
label extraction, synthetic scan metadata generation, scan feature extraction,
sample validation, and dataset manifest serialization.

Usage:
    PYTHONPATH=src python3 scripts/run_sweep.py \\
        --config configs/sweeps/uart_nangate45_smoke_v001.yaml \\
        --dry-run-plan

    PYTHONPATH=src python3 scripts/run_sweep.py \\
        --config configs/sweeps/uart_nangate45_smoke_v001.yaml \\
        --limit 2 \\
        --continue-on-error
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

from dft_hotspot.sweep import SweepConfig, SweepRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run automated multi-configuration parameter sweep to generate paired physical design dataset (Stage 4a)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML sweep configuration file (e.g. configs/sweeps/uart_nangate45_smoke_v001.yaml).",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        default=False,
        help="Display planned parameter combinations and exit without running flows or creating artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sweep points to execute (useful for smoke testing).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip sweep points if sample already exists and passes validation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue executing remaining sweep points if an individual run fails.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite existing sample directories.",
    )

    args = parser.parse_args()
    config_file = Path(args.config).resolve()

    if not config_file.is_file():
        print(f"[ERROR] Sweep configuration file does not exist: {config_file}", file=sys.stderr)
        return 1

    try:
        sweep_cfg = SweepConfig.load(config_file)
        runner = SweepRunner(
            config=sweep_cfg,
            dry_run_plan=args.dry_run_plan,
            limit=args.limit,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
            force=args.force,
        )
        res = runner.run_sweep()

        if args.dry_run_plan:
            return 0

        success_count = res.get("success_count", 0)
        failure_count = res.get("failure_count", 0)

        if success_count > 0 and failure_count == 0:
            return 0
        elif success_count > 0 and failure_count > 0:
            print(f"[WARNING] Sweep completed with partial successes: {success_count} succeeded, {failure_count} failed.")
            return 0
        else:
            print(f"[ERROR] Sweep failed: 0 succeeded, {failure_count} failed.")
            return 1

    except Exception as err:
        print(f"\n[ERROR] Fatal sweep runner error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
