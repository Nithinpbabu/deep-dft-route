#!/usr/bin/env python3
"""CLI Script: Generate Stage 4 Smoke Benchmark Report and Summary.

Synthesizes results from Stage 4a (sweep), Stage 4b (dataset introspection & duplicates),
Stage 4c (heuristics), and Stage 4d (classical ML baselines).

Usage:
    PYTHONPATH=src python3 scripts/generate_stage4_report.py \\
        --dataset-dir data/processed/datasets/uart_nangate45_smoke_v001 \\
        --heuristics-dir experiments/heuristics/uart_nangate45_smoke_v001 \\
        --classical-dir experiments/classical/uart_nangate45_smoke_v001 \\
        --output reports/stage4_smoke_benchmark.md \\
        --summary-json reports/stage4_smoke_benchmark_summary.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dft_hotspot.reporting.stage4_benchmark import generate_stage4_benchmark_report


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage 4 Smoke Benchmark Report and Readiness Review."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/datasets/uart_nangate45_smoke_v001",
        help="Path to processed dataset directory (default: data/processed/datasets/uart_nangate45_smoke_v001)",
    )
    parser.add_argument(
        "--heuristics-dir",
        type=str,
        default="experiments/heuristics/uart_nangate45_smoke_v001",
        help="Path to heuristics experiment directory (default: experiments/heuristics/uart_nangate45_smoke_v001)",
    )
    parser.add_argument(
        "--classical-dir",
        type=str,
        default="experiments/classical/uart_nangate45_smoke_v001",
        help="Path to classical ML experiment directory (default: experiments/classical/uart_nangate45_smoke_v001)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/stage4_smoke_benchmark.md",
        help="Target path for generated Markdown report (default: reports/stage4_smoke_benchmark.md)",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="reports/stage4_smoke_benchmark_summary.json",
        help="Target path for generated JSON summary (default: reports/stage4_smoke_benchmark_summary.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    dataset_dir = ROOT_DIR / args.dataset_dir if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    heuristics_dir = ROOT_DIR / args.heuristics_dir if not Path(args.heuristics_dir).is_absolute() else Path(args.heuristics_dir)
    classical_dir = ROOT_DIR / args.classical_dir if not Path(args.classical_dir).is_absolute() else Path(args.classical_dir)
    output_md = ROOT_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)
    summary_json = ROOT_DIR / args.summary_json if (args.summary_json and not Path(args.summary_json).is_absolute()) else (Path(args.summary_json) if args.summary_json else None)

    print("=" * 75)
    print("STAGE 4e: GENERATING FINAL SMOKE BENCHMARK REPORT")
    print("=" * 75)
    print(f"Dataset Dir     : {dataset_dir}")
    print(f"Heuristics Dir  : {heuristics_dir}")
    print(f"Classical Dir   : {classical_dir}")
    print(f"Output Report   : {output_md}")
    print(f"Summary JSON    : {summary_json}")
    print("=" * 75)

    out_md, out_json = generate_stage4_benchmark_report(
        dataset_dir=dataset_dir,
        heuristics_dir=heuristics_dir,
        classical_dir=classical_dir,
        output_md_path=output_md,
        summary_json_path=summary_json,
    )

    print("\nReport generation completed successfully:")
    print(f"  Markdown Report : {out_md}")
    if out_json:
        print(f"  Summary JSON    : {out_json}")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
