#!/usr/bin/env python3
"""
CLI Script: Train and Evaluate Classical Tabular ML Baselines on Smoke Dataset.
Stage 4d of the DFT-Aware Physical Design Hotspot Prediction framework.

Usage:
    PYTHONPATH=src python3 scripts/train_classical_baselines.py \\
        --manifest data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json \\
        --splits data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json \\
        --output-dir experiments/classical/uart_nangate45_smoke_v001 \\
        --target-channel total_wirelength_density
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

from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.evaluation.metrics import sanitize_metrics_for_json
from dft_hotspot.ml.classical_baselines import (
    export_coefficients_csv,
    export_feature_importance_csv,
    export_metrics_by_split_csv,
    export_metrics_per_model_csv,
    export_metrics_per_sample_csv,
    render_classical_limitations_markdown,
    train_classical_baselines,
)
from dft_hotspot.ml.tabular_dataset import load_tabular_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate classical ML baselines on physical design dataset."
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
        default="experiments/classical/uart_nangate45_smoke_v001",
        help="Output directory for classical ML experiment results",
    )
    parser.add_argument(
        "--target-channel",
        type=str,
        default="total_wirelength_density",
        help="Target routing label channel to predict (default: 'total_wirelength_density')",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Disable generating PNG heatmap comparison previews",
    )
    return parser.parse_args()


def generate_best_model_preview(
    sample_id: str,
    target_2d: np.ndarray,
    pred_2d: np.ndarray,
    model_name: str,
    output_png: Path,
) -> None:
    """Renders a 3-panel comparison figure: Target vs Prediction vs Residual."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    residual = np.abs(pred_2d - target_2d)

    panels = [
        ("Ground Truth Target", target_2d, "viridis", (0.0, 1.0)),
        (f"Prediction: {model_name}", pred_2d, "plasma", (0.0, 1.0)),
        ("Absolute Residual (|pred - target|)", residual, "magma", (0.0, max(0.2, float(np.max(residual))))),
    ]

    for ax, (title, arr, cmap, vlims) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vlims[0], vmax=vlims[1])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Stage 4d Classical ML: {model_name} on {sample_id}", fontsize=12, weight="bold")
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
    print("STAGE 4d: CLASSICAL TABULAR MACHINE LEARNING BASELINES")
    print("=" * 75)
    print(f"Manifest Path       : {manifest_path}")
    print(f"Splits Path         : {splits_path if splits_path.is_file() else 'None (all samples train)'}")
    print(f"Output Directory    : {output_dir}")
    print(f"Target Channel      : {args.target_channel}")
    print(f"Random State        : {args.random_state}")

    # 1. Load dataset
    try:
        dataset = PhysicalDesignDataset.load(manifest_path, load_tensors=True, load_scan=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        return 1

    print(f"Dataset ID          : {dataset.dataset_id} ({len(dataset)} samples)")
    print(f"Design / Platform   : {dataset.design} / {dataset.platform}")

    # 2. Build tabular dataset
    print("\nBuilding tabular feature and target matrices...")
    try:
        tabular_data = load_tabular_dataset(
            dataset=dataset,
            splits=splits_dict,
            target_channel=args.target_channel,
        )
    except Exception as e:
        print(f"[ERROR] Failed to construct tabular dataset: {e}", file=sys.stderr)
        return 1

    print(f"Features Count      : {tabular_data.num_features} channels")
    print(f"Train Matrix Shape  : X={tabular_data.X_train.shape}, y={tabular_data.y_train.shape}")
    print(f"Val Matrix Shape    : X={tabular_data.X_val.shape}, y={tabular_data.y_val.shape}")
    print(f"Test Matrix Shape   : X={tabular_data.X_test.shape}, y={tabular_data.y_test.shape}")
    print(f"Total Grid Cells    : {tabular_data.total_cells}")

    # 3. Train classical baselines
    print("\nTraining classical ML models...")
    results, skipped = train_classical_baselines(tabular_data, random_state=args.random_state)
    print(f"Models Trained ({len(results)}): {', '.join(results.keys())}")
    if skipped:
        print(f"Models Skipped ({len(skipped)}): {', '.join(skipped.keys())}")

    # 4. Save configuration and skipped models
    cfg_path = output_dir / "experiment_config.json"
    cfg_data = {
        "schema_version": "classical_experiment_v001",
        "dataset_id": dataset.dataset_id,
        "design": dataset.design,
        "platform": dataset.platform,
        "target_channel": args.target_channel,
        "feature_names": tabular_data.feature_names,
        "random_state": args.random_state,
        "matrix_shapes": {
            "train": list(tabular_data.X_train.shape),
            "val": list(tabular_data.X_val.shape),
            "test": list(tabular_data.X_test.shape),
        },
        "models_trained": list(results.keys()),
        "models_skipped": skipped,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    skipped_path = output_dir / "skipped_models.json"
    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)

    # 5. Export coefficients and feature importances
    coef_csv = output_dir / "coefficients.csv"
    export_coefficients_csv(results, tabular_data.feature_names, coef_csv)

    fi_csv = output_dir / "feature_importance.csv"
    export_feature_importance_csv(results, tabular_data.feature_names, fi_csv)

    # 6. Export metrics tables
    eval_partition = "val" if len(tabular_data.y_val) > 0 else "all"

    model_csv = output_dir / "metrics_per_model.csv"
    export_metrics_per_model_csv(results, eval_partition, model_csv)

    sample_csv = output_dir / "metrics_per_sample.csv"
    export_metrics_per_sample_csv(results, sample_csv)

    split_csv = output_dir / "metrics_by_split.csv"
    export_metrics_by_split_csv(results, split_csv)

    # 7. Determine best models on evaluation partition
    best_mae: Tuple[str, float] = ("none", float("inf"))
    best_recall: Tuple[str, float] = ("none", 0.0)

    for m_name, res in results.items():
        val_mae = res.metrics_by_split.get(eval_partition, {}).get("mae", {}).get("mean")
        val_rec = res.metrics_by_split.get(eval_partition, {}).get("recall_5pct", {}).get("mean")
        if val_mae is not None and val_mae < best_mae[1]:
            best_mae = (m_name, val_mae)
        if val_rec is not None and val_rec > best_recall[1]:
            best_recall = (m_name, val_rec)

    # Load Stage 4c heuristic reference if available
    heuristic_ref_path = Path("experiments/heuristics/uart_nangate45_smoke_v001/metrics_summary.json")
    heuristic_ref_val = None
    if heuristic_ref_path.is_file():
        try:
            with open(heuristic_ref_path, "r", encoding="utf-8") as f:
                h_data = json.load(f)
                h_val_data = h_data.get("by_split", {}).get("val", {}).get("simple_weighted_sum", {})
                heuristic_ref_val = {
                    "mae": h_val_data.get("mae", {}).get("mean", 0.0155),
                    "recall_5pct": h_val_data.get("recall_5pct", {}).get("mean", 0.6683),
                    "pearson": h_val_data.get("pearson", {}).get("mean", 0.6587),
                }
        except Exception:
            heuristic_ref_val = None

    # Write limitations markdown
    limitations_md = output_dir / "limitations.md"
    limitations_text = render_classical_limitations_markdown(
        eval_partition=eval_partition,
        best_model_mae=best_mae,
        best_model_recall=best_recall,
        heuristic_ref=heuristic_ref_val,
    )
    with open(limitations_md, "w", encoding="utf-8") as f:
        f.write(limitations_text)

    # 8. Export model_summary.json
    summary_data = {
        "schema_version": "classical_summary_v001",
        "dataset_id": dataset.dataset_id,
        "design": dataset.design,
        "platform": dataset.platform,
        "target_channel": args.target_channel,
        "evaluation_partition_used": eval_partition,
        "best_model_by_mae": {"model": best_mae[0], "mae": best_mae[1]},
        "best_model_by_recall_5pct": {"model": best_recall[0], "recall_5pct": best_recall[1]},
        "heuristic_reference_simple_weighted_sum": heuristic_ref_val,
        "models": {
            m_name: {
                "type": res.model_type,
                "hyperparameters": res.hyperparameters,
                "metrics_by_split": res.metrics_by_split,
            }
            for m_name, res in results.items()
        },
        "skipped_models": skipped,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    summary_json_path = output_dir / "model_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_metrics_for_json(summary_data), f, indent=2)

    # 9. Save predictions and previews for best model
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    best_res = results[best_mae[0]]

    for sid, pred_2d in best_res.predictions_2d.items():
        np.savez_compressed(preds_dir / f"{sid}_best_pred.npz", prediction=pred_2d)

    if not args.no_previews:
        previews_dir = output_dir / "previews"
        for sid, s_tab in tabular_data.samples.items():
            pred_2d = best_res.predictions_2d[sid]
            target_2d = s_tab.y.reshape(s_tab.grid_shape)
            preview_img = previews_dir / f"{sid}_{best_mae[0]}_preview.png"
            generate_best_model_preview(sid, target_2d, pred_2d, best_mae[0], preview_img)

    # 10. Print console summary
    print("\n" + "=" * 75)
    print(f"CLASSICAL ML EVALUATION SUMMARY: Partition '{eval_partition.upper()}'")
    print("=" * 75)
    print(f"Best Model by MAE        : {best_mae[0]} (MAE = {best_mae[1]:.4f})")
    print(f"Best Model by Recall@5%  : {best_recall[0]} (Recall = {best_recall[1]:.2%})")
    if heuristic_ref_val:
        print(f"Heuristic Reference (S4c): simple_weighted_sum (MAE = {heuristic_ref_val['mae']:.4f}, Recall = {heuristic_ref_val['recall_5pct']:.2%})")

    print(f"\n{eval_partition.upper()} Split Model Comparison Table:")
    print(f"{'Model':<28} | {'MAE':<7} | {'RMSE':<7} | {'Pearson':<8} | {'Recall@5%':<9} | {'IoU@5%':<7}")
    print("-" * 75)
    for m_name in results.keys():
        m_stats = results[m_name].metrics_by_split.get(eval_partition, {})
        mae_v = f"{m_stats.get('mae', {}).get('mean', 0.0):.4f}" if m_stats.get('mae', {}).get('mean') is not None else "N/A"
        rmse_v = f"{m_stats.get('rmse', {}).get('mean', 0.0):.4f}" if m_stats.get('rmse', {}).get('mean') is not None else "N/A"
        r_v = f"{m_stats.get('pearson', {}).get('mean', 0.0):.4f}" if m_stats.get('pearson', {}).get('mean') is not None else "N/A"
        rec_v = f"{m_stats.get('recall_5pct', {}).get('mean', 0.0):.2%}" if m_stats.get('recall_5pct', {}).get('mean') is not None else "N/A"
        iou_v = f"{m_stats.get('iou_5pct', {}).get('mean', 0.0):.4f}" if m_stats.get('iou_5pct', {}).get('mean') is not None else "N/A"
        print(f"{m_name:<28} | {mae_v:<7} | {rmse_v:<7} | {r_v:<8} | {rec_v:<9} | {iou_v:<7}")

    print("\nWritten Artifacts:")
    print(f"  - Config JSON        : {cfg_path}")
    print(f"  - Model Metrics CSV  : {model_csv}")
    print(f"  - Per-Sample CSV     : {sample_csv}")
    print(f"  - By-Split CSV       : {split_csv}")
    print(f"  - Model Summary JSON : {summary_json_path}")
    print(f"  - Coefficients CSV   : {coef_csv}")
    print(f"  - Feature Importances: {fi_csv}")
    print(f"  - Skipped Models     : {skipped_path}")
    print(f"  - Limitations Doc    : {limitations_md}")
    print(f"  - Predictions NPZ    : {preds_dir}")
    if not args.no_previews:
        print(f"  - Previews PNG       : {output_dir / 'previews'}")

    print("=" * 75)
    print("Stage 4d Classical ML Baselines Complete.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
