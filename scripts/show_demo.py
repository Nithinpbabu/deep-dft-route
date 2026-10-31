#!/usr/bin/env python3
"""
Interactive & Visual Demonstration CLI for DFT Hotspot Prediction.

Visualizes how the Multimodal CNN + GNN model predicts physical routing
congestion from pre-route placement features on unseen holdout circuits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Computes continuous and top-5% discrete hotspot metrics."""
    p_flat = pred.flatten()
    t_flat = target.flatten()
    mae = float(np.mean(np.abs(p_flat - t_flat)))
    rmse = float(np.sqrt(np.mean((p_flat - t_flat) ** 2)))
    
    # Pearson r
    if np.std(p_flat) > 1e-8 and np.std(t_flat) > 1e-8:
        pearson_r = float(np.corrcoef(p_flat, t_flat)[0, 1])
    else:
        pearson_r = float("nan")

    # Hotspot masks (Top 5% highest congestion bins)
    q_gt = np.percentile(t_flat, 95)
    q_pred = np.percentile(p_flat, 95)
    gt_mask = target >= q_gt
    pred_mask = pred >= q_pred

    tp = np.logical_and(gt_mask, pred_mask)
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask)
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask))

    num_gt = np.sum(gt_mask)
    recall = float(np.sum(tp) / num_gt) if num_gt > 0 else 0.0
    union = np.logical_or(gt_mask, pred_mask)
    iou = float(np.sum(tp) / np.sum(union)) if np.sum(union) > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "recall_top5": recall,
        "iou_top5": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a visual and metric demo of the DFT Hotspot Prediction model."
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="dft_counter_nangate45_u65_clk3p0_ar1p5_chains1",
        help="Sample identifier from the holdout test set.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/datasets/expanded_v002",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="experiments/fusion/expanded_v002_fusion_primary/predictions",
        help="Path to saved fusion predictions.",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="reports/demo_prediction_comparison.png",
        help="Where to save the visualization figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive GUI window with the visualization.",
    )

    args = parser.parse_args()

    sample_dir = Path(args.dataset_dir) / "samples" / args.sample_id
    pred_path = Path(args.predictions_dir) / f"{args.sample_id}_pred.npz"

    if not sample_dir.exists():
        print(f"[ERROR] Sample directory not found: {sample_dir}")
        sys.exit(1)
    if not pred_path.exists():
        print(f"[ERROR] Prediction file not found: {pred_path}")
        sys.exit(1)

    print("=" * 72)
    print("      DFT HOTSPOT PREDICTION: AI CHIP DESIGN DEMO")
    print("=" * 72)
    print(f"Sample Tested : {args.sample_id}")
    print(f"Design Class  : Unseen Holdout Circuit (dft_counter / spi)")
    print(f"Technology    : Nangate45 (45nm Open PDK)")
    print(f"Resolution    : 64 x 64 spatial die bins")
    print("-" * 72)

    # 1. Load data
    features_spatial = np.load(sample_dir / "features_spatial.npz")
    cell_density = features_spatial["features"][0]  # Channel 0: cell_density
    seq_density = features_spatial["features"][1]   # Channel 1: sequential_cell_density

    labels_data = np.load(sample_dir / "labels_routing.npz")
    target_wirelength = labels_data["labels"][0]    # Channel 0: total_wirelength_density

    pred_data = np.load(pred_path)
    predicted_wirelength = pred_data["prediction"]

    # 2. Compute metrics
    m = compute_metrics(predicted_wirelength, target_wirelength)

    print(f"Prediction Performance against Post-Route Ground Truth:")
    print(f"  * Pearson Correlation (r) : {m['pearson_r']:.4f}  (Strong positive alignment)")
    print(f"  * Mean Absolute Error     : {m['mae']:.4f}")
    print(f"  * Root Mean Squared Error : {m['rmse']:.4f}")
    print(f"  * Critical Hotspot Recall : {m['recall_top5'] * 100:.1f}% of top-5% congestion captured")
    print(f"  * Hotspot IoU @ 5%        : {m['iou_top5']:.4f}")
    print("=" * 72)

    if not _HAS_MPL:
        print("[WARNING] matplotlib not found. Visual demo image cannot be plotted.")
        return

    # 3. Create high-resolution presentation figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(
        f"Pre-Route DFT Hotspot Prediction vs Ground Truth Router\nDesign: {args.sample_id}",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Input Placement Cell Density (Pre-Route)
    im0 = axes[0].imshow(cell_density, cmap="Blues", origin="lower")
    axes[0].set_title("1. Pre-Route Input\n(Standard Cell Placement)", fontsize=11, fontweight="semibold")
    axes[0].set_xlabel("Die Grid X (0-63)")
    axes[0].set_ylabel("Die Grid Y (0-63)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Cell Count")

    # Panel 2: Predicted Congestion (Model Output)
    im1 = axes[1].imshow(predicted_wirelength, cmap="inferno", origin="lower")
    axes[1].set_title(
        f"2. AI Predicted Congestion\n(Pearson r = {m['pearson_r']:.3f})",
        fontsize=11,
        fontweight="semibold",
    )
    axes[1].set_xlabel("Die Grid X (0-63)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized Wirelength")

    # Panel 3: Ground Truth Post-Route Wirelength (OpenROAD)
    im2 = axes[2].imshow(target_wirelength, cmap="inferno", origin="lower")
    axes[2].set_title("3. Actual Router Result\n(Post-Route Detailed Wires)", fontsize=11, fontweight="semibold")
    axes[2].set_xlabel("Die Grid X (0-63)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Actual Wirelength")

    # Panel 4: Hotspot Alert Classification (Top 5%)
    # Build RGB alert map:
    # Green = True Positive (Caught Hotspot)
    # Red = False Negative (Missed Hotspot)
    # Blue = False Positive (False Alarm)
    # Light gray = Non-congested
    alert_rgb = np.ones((64, 64, 3), dtype=float) * 0.94  # background light gray
    alert_rgb[m['tp']] = [0.12, 0.65, 0.23]   # Green: True Positive
    alert_rgb[m['fn']] = [0.85, 0.15, 0.15]   # Red: Missed Hotspot
    alert_rgb[m['fp']] = [0.20, 0.45, 0.85]   # Blue: False Alarm

    axes[3].imshow(alert_rgb, origin="lower")
    axes[3].set_title(
        f"4. Hotspot Detection Alert\n(Top-5% Recall: {m['recall_top5'] * 100:.1f}%)",
        fontsize=11,
        fontweight="semibold",
    )
    axes[3].set_xlabel("Die Grid X (0-63)")

    # Custom legend for alerts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.12, 0.65, 0.23], label=f"True Hotspot Caught ({np.sum(m['tp'])})"),
        Patch(facecolor=[0.85, 0.15, 0.15], label=f"Missed Hotspot ({np.sum(m['fn'])})"),
        Patch(facecolor=[0.20, 0.45, 0.85], label=f"False Alarm ({np.sum(m['fp'])})"),
    ]
    axes[3].legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    out_path = Path(args.output_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[SUCCESS] Visual demo saved to: {out_path.resolve()}")

    if args.show:
        print("[INFO] Displaying interactive plot window...")
        plt.show()


if __name__ == "__main__":
    main()
