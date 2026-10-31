#!/usr/bin/env python3
"""
CLI Script: Evaluate Non-Learning Heuristic Baselines on Smoke Dataset.
Stage 4c of the DFT-Aware Physical Design Hotspot Prediction framework.

Usage:
    PYTHONPATH=src python3 scripts/evaluate_heuristics.py \\
        --manifest data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json \\
        --splits data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json \\
        --output-dir experiments/heuristics/uart_nangate45_smoke_v001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.evaluation.heuristics import (
    build_heuristic_predictions,
    evaluate_dataset_heuristics,
    export_metrics_by_split_csv,
    export_metrics_per_sample_csv,
    get_standard_heuristics,
    render_limitations_markdown,
)
from dft_hotspot.evaluation.metrics import sanitize_metrics_for_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate non-learning heuristic baselines on physical design dataset."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to dataset_manifest.json",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Path to splits_v001.json (optional, defaults to manifest dir/splits_v001.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/heuristics/uart_nangate45_smoke_v001",
        help="Output directory for heuristic evaluation results",
    )
    parser.add_argument(
        "--target-channel",
        type=str,
        default="total_wirelength_density",
        help="Target routing label channel to predict (default: 'total_wirelength_density')",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Disable generating PNG heatmap comparison previews",
    )
    return parser.parse_args()


def generate_comparison_preview(
    sample_id: str,
    target_2d: np.ndarray,
    target_channel: str,
    predictions: dict,
    output_png: Path,
) -> None:
    """Renders a 4-panel comparison preview figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    panels = [
        ("Ground Truth Target", target_channel, target_2d, "viridis"),
        ("Heuristic 1: Pin Density", "physical/pin_density", predictions.get("pin_density"), "plasma"),
        ("Heuristic 5: Scan Segment", "scan/scan_chain_segment_density", predictions.get("scan_segment_density"), "inferno"),
        ("Heuristic 7: Weighted Sum", "0.35 pin + 0.25 cell + 0.25 scan + 0.15 wl", predictions.get("simple_weighted_sum"), "magma"),
    ]

    for ax, (title, subtitle, arr, cmap) in zip(axes, panels):
        if arr is not None:
            im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0)
            ax.set_title(f"{title}\n({subtitle})", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f"{title}\n(N/A)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Stage 4c Heuristic Baseline Comparison: {sample_id}", fontsize=13, weight="bold")
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = Path(args.splits).resolve() if args.splits else manifest_path.parent / "splits_v001.json"
    splits_dict = None
    if splits_path.is_file():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_dict = json.load(f)

    print("=" * 75)
    print("STAGE 4c: NON-LEARNING HEURISTIC BASELINE EVALUATION")
    print("=" * 75)
    print(f"Manifest Path       : {manifest_path}")
    print(f"Splits Path         : {splits_path if splits_path.is_file() else 'None (evaluating all)'}")
    print(f"Output Directory    : {output_dir}")
    print(f"Target Channel      : {args.target_channel}")

    # 1. Load dataset
    try:
        dataset = PhysicalDesignDataset.load(manifest_path, load_tensors=True, load_scan=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        return 1

    print(f"Dataset ID          : {dataset.dataset_id} ({len(dataset)} samples)")
    print(f"Design / Platform   : {dataset.design} / {dataset.platform}")

    # 2. Get standard baselines
    baselines = get_standard_heuristics()
    print(f"\nHeuristic Baselines ({len(baselines)}):")
    for idx, b in enumerate(baselines):
        print(f"  [{idx + 1}] {b.name:<28}: {b.description}")

    # 3. Save heuristic configuration
    cfg_path = output_dir / "heuristic_config.json"
    cfg_data = {
        "schema_version": "heuristic_config_v001",
        "target_channel": args.target_channel,
        "normalization": "per_sample_max",
        "baselines": [
            {
                "name": b.name,
                "description": b.description,
                "channel_name": b.channel_name,
                "weights": b.weights,
                "normalize": b.normalize,
            }
            for b in baselines
        ],
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    # 4. Run evaluation across dataset
    eval_report = evaluate_dataset_heuristics(
        dataset=dataset,
        splits=splits_dict,
        target_channel=args.target_channel,
        baselines=baselines,
    )

    # 5. Export CSV and JSON summaries
    per_sample_csv = output_dir / "metrics_per_sample.csv"
    export_metrics_per_sample_csv(eval_report, per_sample_csv)

    by_split_csv = output_dir / "metrics_by_split.csv"
    export_metrics_by_split_csv(eval_report, by_split_csv)

    summary_json_path = output_dir / "metrics_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_metrics_for_json(eval_report), f, indent=2)

    limitations_md_path = output_dir / "limitations.md"
    limitations_content = render_limitations_markdown(eval_report)
    with open(limitations_md_path, "w", encoding="utf-8") as f:
        f.write(limitations_content)

    # 6. Generate comparison previews
    if not args.no_previews:
        previews_dir = output_dir / "previews"
        for sid, rec in dataset.items():
            if rec.sample is None:
                continue
            t_idx = rec.sample.label_channel_names.index(args.target_channel)
            target_2d = rec.sample.label_tensor[t_idx]
            preds = build_heuristic_predictions(rec.sample, baselines=baselines)
            img_path = previews_dir / f"{sid}_heuristics_preview.png"
            generate_comparison_preview(sid, target_2d, args.target_channel, preds, img_path)

    # 7. Print summary tables to console
    summary = eval_report.get("summary", {})
    by_split = eval_report.get("by_split", {})
    eval_partition = eval_report.get("evaluation_partition_used", "all")

    print("\n" + "=" * 75)
    print(f"EVALUATION SUMMARY: Partition '{eval_partition.upper()}'")
    print("=" * 75)
    print(f"Target Label Channel : {args.target_channel}")
    print(f"Best Baseline by MAE : {summary.get('best_baseline_by_mae')} (MAE: {summary.get('best_mae'):.4f})")
    print(f"Worst Baseline by MAE: {summary.get('worst_baseline_by_mae')} (MAE: {summary.get('worst_mae'):.4f})")
    print(f"Best Recall@5%       : {summary.get('best_baseline_by_recall_5pct')} (Recall: {summary.get('best_recall_5pct'):.2%})")

    print(f"\n{eval_partition.upper()} Split Metrics Table:")
    print(f"{'Baseline':<28} | {'MAE':<7} | {'RMSE':<7} | {'Pearson':<8} | {'Recall@5%':<9} | {'IoU@5%':<7}")
    print("-" * 75)
    part_stats = by_split.get(eval_partition, {})
    for b in baselines:
        s = part_stats.get(b.name, {})
        mae_v = f"{s.get('mae', {}).get('mean', 0.0):.4f}" if s.get('mae', {}).get('mean') is not None else "N/A"
        rmse_v = f"{s.get('rmse', {}).get('mean', 0.0):.4f}" if s.get('rmse', {}).get('mean') is not None else "N/A"
        r_v = f"{s.get('pearson', {}).get('mean', 0.0):.4f}" if s.get('pearson', {}).get('mean') is not None else "N/A"
        rec_v = f"{s.get('recall_5pct', {}).get('mean', 0.0):.2%}" if s.get('recall_5pct', {}).get('mean') is not None else "N/A"
        iou_v = f"{s.get('iou_5pct', {}).get('mean', 0.0):.4f}" if s.get('iou_5pct', {}).get('mean') is not None else "N/A"
        print(f"{b.name:<28} | {mae_v:<7} | {rmse_v:<7} | {r_v:<8} | {rec_v:<9} | {iou_v:<7}")

    print("\nWritten Artifacts:")
    print(f"  - Per-Sample CSV     : {per_sample_csv}")
    print(f"  - By-Split CSV       : {by_split_csv}")
    print(f"  - Metrics Summary    : {summary_json_path}")
    print(f"  - Baseline Config    : {cfg_path}")
    print(f"  - Limitations Doc    : {limitations_md_path}")
    if not args.no_previews:
        print(f"  - Visual Previews    : {output_dir / 'previews'}")

    print("=" * 75)
    print("Stage 4c Heuristic Baseline Evaluation Complete.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
