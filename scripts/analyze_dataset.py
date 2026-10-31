#!/usr/bin/env python3
"""
CLI Script: Analyze Physical Design Dataset, Compute Statistics, Detect Duplicates, and Generate Splits.
Stage 4b of the DFT-Aware Physical Design Hotspot Prediction framework.

Usage:
    PYTHONPATH=src python3 scripts/analyze_dataset.py \\
        --manifest data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json \\
        --output-dir data/processed/datasets/uart_nangate45_smoke_v001 \\
        --write-splits
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

from dft_hotspot.data.dataset_collection import PhysicalDesignDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze physical design dataset, compute statistics, detect duplicates, and generate splits."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to dataset_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis artifacts (defaults to manifest directory)",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="Generate and export splits_v001.json file",
    )
    parser.add_argument(
        "--split-policy",
        type=str,
        default="smoke_config_holdout",
        help="Split policy strategy (default: 'smoke_config_holdout')",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum absolute difference tolerance for duplicate detection (default: 1e-6)",
    )
    parser.add_argument(
        "--no-require-scan",
        action="store_true",
        help="Do not enforce presence of scan-aware feature tensors during validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest_path.parent
    require_scan = not args.no_require_scan

    print("=" * 75)
    print("STAGE 4b: DATASET INTROSPECTION & SPLIT GENERATION")
    print("=" * 75)
    print(f"Manifest Path       : {manifest_path}")
    print(f"Output Directory    : {output_dir}")
    print(f"Require Scan        : {require_scan}")
    print(f"Split Policy        : {args.split_policy}")

    # 1. Load dataset
    try:
        dataset = PhysicalDesignDataset.load(manifest_path, load_tensors=True, load_scan=require_scan)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        return 1

    print(f"Dataset ID          : {dataset.dataset_id}")
    print(f"Design / Platform   : {dataset.design} / {dataset.platform}")
    print(f"Total Sample Count  : {len(dataset)}")

    # 2. Validate dataset
    try:
        val_report = dataset.validate(require_scan=require_scan, require_fused=True, raise_on_error=False)
        val_status = "ALL PASS" if val_report["status"] == "pass" else f"FAILED ({val_report['failed_samples']} errors)"
    except Exception as e:
        val_status = f"FAILED: {e}"

    print(f"Validation Status   : {val_status}")

    # 3. Tensor shapes summary
    summary = dataset.summary()
    shapes = summary.get("tensor_shapes", {})
    print("\nTensor Shape Summary:")
    print(f"  Physical Features : {shapes.get('physical')}")
    print(f"  Scan Features     : {shapes.get('scan')}")
    print(f"  Fused Input       : {shapes.get('fused')}")
    print(f"  Routing Labels    : {shapes.get('labels')}")

    # 4. Duplicate layout detection
    dup_report = dataset.detect_duplicates(tolerance=args.tolerance)
    print("\nDuplicate Detection Summary:")
    print(f"  Total Samples          : {dup_report['total_samples']}")
    print(f"  Unique Fused Tensors   : {dup_report.get('unique_fused_inputs', dup_report['unique_layouts'])}")
    print(f"  Unique Routing Layouts : {dup_report.get('unique_routing_layouts', dup_report['unique_layouts'])}")
    print(f"  Exact Duplicate Groups : {dup_report['duplicate_group_count']} groups")
    for g in dup_report.get("duplicate_groups", []):
        sids = g["sample_ids"]
        shared = ", ".join(f"{k}={v}" for k, v in g["shared_config"].items())
        diff_info = f"max_diff_fused={g['max_abs_diff_fused']:.1e}, max_diff_lbl={g['max_abs_diff_labels']:.1e}"
        print(f"    Group {g['group_id']}: {sids} ({shared}) [{diff_info}]")
    if dup_report.get("routing_duplicate_group_count", 0) > 0 and dup_report.get("duplicate_group_count", 0) == 0:
        print(f"  Routing Duplicate Groups: {dup_report['routing_duplicate_group_count']} groups (identical routing labels across chain counts)")

    # 5. Split summary
    splits_dict = dataset.create_splits(policy=args.split_policy)
    train_ids = splits_dict.get("train", [])
    val_ids = splits_dict.get("val", [])
    test_ids = splits_dict.get("test", [])
    print(f"\nSplit Allocation Summary ({args.split_policy}):")
    print(f"  Train ({len(train_ids)} samples) : {', '.join(train_ids)}")
    print(f"  Val   ({len(val_ids)} sample)  : {', '.join(val_ids)}")
    print(f"  Test  ({len(test_ids)} sample)  : {', '.join(test_ids)}")

    # Check for val/test duplicate warning
    for lim in splits_dict.get("limitations", []):
        if "Duplicate layout alert" in lim:
            print(f"  * WARNING: {lim}")

    # 6. Export analysis artifacts
    written = dataset.export_dataset_analysis(
        output_dir=output_dir,
        write_splits=args.write_splits,
        split_policy=args.split_policy,
    )

    print("\nWritten Artifacts:")
    for name, p in written.items():
        print(f"  - {name}: {p}")

    print("=" * 75)
    print("Stage 4b Dataset Introspection Complete.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
