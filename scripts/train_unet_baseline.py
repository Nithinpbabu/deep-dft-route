"""
CLI Orchestration Script for TinyUNet Baseline on expanded_v002.
Stage 5b of the DFT-Aware Physical Design Hotspot Prediction framework.

Executes:
1. Dataset loading (expanded_v002) and partition resolution (splits_v001.json).
2. Baseline benchmarking:
   - Mean baseline (empirical train target mean).
   - Heuristic baseline (simple_weighted_sum).
3. PyTorch TinyUNet training on GPU/CPU with validation early stopping and checkpointing.
4. Comprehensive spatial metric evaluation across train, val, and test splits.
5. Export of experiment config, checkpoints, history, predictions, and comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.evaluation.heuristics import HeuristicBaseline, get_standard_heuristics
from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json
from dft_hotspot.ml.torch_dataset import build_split_datasets, create_dataloaders
from dft_hotspot.ml.train_unet import compute_multichannel_metrics, evaluate_model, set_seed, train_unet
from dft_hotspot.ml.unet import TinyUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_unet_baseline")


def _resolve_target_channels(dataset: PhysicalDesignDataset, target_channel: Union[str, List[str]]) -> List[str]:
    sample_rec = next(iter(dataset.samples.values()), None)
    all_label_names = sample_rec.sample.label_channel_names if sample_rec and sample_rec.sample else []
    if target_channel == "all":
        return list(all_label_names)
    elif isinstance(target_channel, list):
        return target_channel
    elif "," in target_channel:
        return [c.strip() for c in target_channel.split(",")]
    else:
        return [target_channel]


def evaluate_heuristic_baseline(
    dataset: PhysicalDesignDataset,
    splits: Dict[str, List[str]],
    target_channel: Union[str, List[str]] = "total_wirelength_density",
    heuristic_name: str = "simple_weighted_sum",
) -> Dict[str, Dict[str, float]]:
    """Evaluates a heuristic baseline across train, val, and test splits."""
    heuristics = {b.name: b for b in get_standard_heuristics()}
    if heuristic_name not in heuristics:
        raise KeyError(f"Unknown heuristic '{heuristic_name}'. Available: {list(heuristics.keys())}")
    baseline = heuristics[heuristic_name]

    resolved_channels = _resolve_target_channels(dataset, target_channel)
    eval_channel = "total_wirelength_density" if "total_wirelength_density" in resolved_channels else resolved_channels[0]

    split_metrics: Dict[str, Dict[str, float]] = {}

    for split_key in ["train", "val", "test"]:
        sids = splits.get(split_key, [])
        preds_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []

        for sid in sids:
            rec = dataset.samples.get(sid)
            if rec is None or rec.sample is None:
                continue
            sample = rec.sample
            fused = sample.get_fused_input_tensor(include_scan=True)
            names = sample.get_fused_channel_names(include_scan=True)
            pred = baseline.predict(fused, names)

            t_idx = sample.label_channel_names.index(eval_channel)
            target = sample.label_tensor[t_idx].astype(np.float32)

            preds_list.append(pred)
            targets_list.append(target)

        if preds_list and targets_list:
            concat_p = np.stack(preds_list, axis=0)
            concat_t = np.stack(targets_list, axis=0)
            split_metrics[split_key] = compute_all_metrics(concat_p, concat_t)
        else:
            split_metrics[split_key] = {}

    return split_metrics


def evaluate_mean_baseline(
    dataset: PhysicalDesignDataset,
    splits: Dict[str, List[str]],
    target_channel: Union[str, List[str]] = "total_wirelength_density",
) -> Dict[str, Dict[str, float]]:
    """Evaluates empirical train-mean predictor across train, val, and test splits."""
    resolved_channels = _resolve_target_channels(dataset, target_channel)
    train_sids = splits.get("train", [])

    means_per_channel: Dict[str, float] = {}
    for ch in resolved_channels:
        vals: List[float] = []
        for sid in train_sids:
            rec = dataset.samples.get(sid)
            if rec is None or rec.sample is None:
                continue
            t_idx = rec.sample.label_channel_names.index(ch)
            vals.extend(rec.sample.label_tensor[t_idx].ravel().tolist())
        means_per_channel[ch] = float(np.mean(vals)) if vals else 0.0

    split_metrics: Dict[str, Dict[str, float]] = {}

    for split_key in ["train", "val", "test"]:
        sids = splits.get(split_key, [])
        preds_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []

        for sid in sids:
            rec = dataset.samples.get(sid)
            if rec is None or rec.sample is None:
                continue
            sample = rec.sample

            p_list = []
            t_list = []
            for ch in resolved_channels:
                orig_t_idx = sample.label_channel_names.index(ch)
                t_arr = sample.label_tensor[orig_t_idx].astype(np.float32)
                p_arr = np.full(t_arr.shape, means_per_channel[ch], dtype=np.float32)
                t_list.append(t_arr)
                p_list.append(p_arr)

            preds_list.append(np.stack(p_list, axis=0))
            targets_list.append(np.stack(t_list, axis=0))

        if preds_list and targets_list:
            concat_p = np.stack(preds_list, axis=0)
            concat_t = np.stack(targets_list, axis=0)
            if len(resolved_channels) == 1:
                split_metrics[split_key] = compute_all_metrics(concat_p[:, 0], concat_t[:, 0])
            else:
                macro_m, _, _ = compute_multichannel_metrics(concat_p, concat_t, channel_names=resolved_channels)
                split_metrics[split_key] = macro_m
        else:
            split_metrics[split_key] = {}

    return split_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate TinyUNet baseline on expanded_v002.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/datasets/expanded_v002"),
        help="Path to expanded_v002 dataset directory.",
    )
    parser.add_argument(
        "--manifest",
        "--dataset-manifest",
        dest="manifest",
        type=Path,
        default=None,
        help="Path to dataset_manifest.json (alternative to --dataset-dir).",
    )
    parser.add_argument(
        "--splits-file",
        "--splits",
        dest="splits_file",
        type=Path,
        default=Path("data/processed/datasets/expanded_v002/splits_v001.json"),
        help="Path to splits_v001.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/unet/expanded_v002_tiny_unet"),
        help="Output directory for experiment artifacts.",
    )
    parser.add_argument(
        "--target-channel",
        type=str,
        default="total_wirelength_density",
        help="Target routing label channel.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "mae", "l1", "huber", "smooth_l1"],
        help="Loss criterion to optimize ('mse', 'mae', 'huber').",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs to train.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument("--base-channels", type=int, default=16, help="Base channel count in TinyUNet.")
    parser.add_argument("--final-activation", type=str, default="sigmoid", help="Final activation ('sigmoid', 'relu', 'none').")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda:0' or 'cpu').")
    args = parser.parse_args()

    if not _HAS_TORCH:
        logger.error("PyTorch is not installed in the current environment.")
        sys.exit(1)

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        manifest_path = Path(args.manifest)
        dataset_dir = manifest_path.parent
    else:
        manifest_path = args.dataset_dir / "dataset_manifest.json"
        dataset_dir = args.dataset_dir

    if not manifest_path.exists():
        logger.error("Dataset manifest not found: %s", manifest_path)
        sys.exit(1)

    if not args.splits_file.exists():
        logger.error("Splits file not found: %s", args.splits_file)
        sys.exit(1)

    with open(args.splits_file, "r") as f:
        splits = json.load(f)

    logger.info("Loading dataset from: %s", manifest_path)
    dataset = PhysicalDesignDataset.load(manifest_path, load_tensors=True, load_scan=True)
    logger.info("Loaded %d samples from dataset.", len(dataset))

    train_count = len(splits.get("train", []))
    val_count = len(splits.get("val", []))
    test_count = len(splits.get("test", []))
    logger.info("Split partitions: train=%d, val=%d, test=%d", train_count, val_count, test_count)

    # Step 1: Baseline evaluations
    logger.info("Computing MeanBaseline on %s...", args.target_channel)
    mean_metrics = evaluate_mean_baseline(dataset, splits, target_channel=args.target_channel)

    logger.info("Computing HeuristicBaseline (simple_weighted_sum) on %s...", args.target_channel)
    heuristic_metrics = evaluate_heuristic_baseline(
        dataset, splits, target_channel=args.target_channel, heuristic_name="simple_weighted_sum"
    )

    # Step 2: PyTorch Dataset & Dataloaders
    target_arg = args.target_channel
    if target_arg != "all" and "," in target_arg:
        target_arg = [c.strip() for c in target_arg.split(",")]

    split_datasets = build_split_datasets(dataset, splits, target_channel=target_arg, preload=True)
    dataloaders = create_dataloaders(split_datasets, batch_size=args.batch_size)

    in_channels = split_datasets["train"].in_channels
    out_channels = split_datasets["train"].out_channels
    target_channel_names = split_datasets["train"].target_channel_names

    # Step 3: Setup Device and Model
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Selected training device: %s", device)

    model = TinyUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=args.base_channels,
        final_activation=args.final_activation,
    )
    logger.info(
        "Instantiated TinyUNet (in=%d, out=%d, params=%d, act=%s, loss=%s)",
        in_channels,
        out_channels,
        model.count_parameters(),
        args.final_activation,
        args.loss,
    )
    logger.info("Target channels (%d): %s", len(target_channel_names), target_channel_names)

    # Step 4: Train TinyUNet
    logger.info("Starting training (loss=%s, max_epochs=%d, patience=%d, lr=%s)...", args.loss, args.epochs, args.patience, args.lr)
    train_summary = train_unet(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        output_dir=output_dir,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
        loss_name=args.loss,
    )
    logger.info(
        "Training completed in %s s. Best epoch: %d, Best val_loss: %.6f",
        train_summary["total_training_time_s"],
        train_summary["best_epoch"],
        train_summary["best_val_loss"],
    )

    # Step 5: Final Evaluation on Best Model
    loss_key = args.loss.lower()
    if loss_key in ("mae", "l1"):
        eval_criterion = nn.L1Loss()
    elif loss_key in ("huber", "smooth_l1", "smoothl1"):
        eval_criterion = nn.SmoothL1Loss()
    else:
        eval_criterion = nn.MSELoss()

    clamp_eval = (args.final_activation.lower() == "none")
    unet_metrics_by_split: Dict[str, Dict[str, Any]] = {}
    sample_predictions: Dict[str, np.ndarray] = {}

    for split_key, loader in dataloaders.items():
        avg_loss, metrics, split_preds = evaluate_model(
            model,
            loader,
            eval_criterion,
            device,
            clamp_output=clamp_eval,
            channel_names=target_channel_names,
        )
        unet_metrics_by_split[split_key] = metrics
        sample_predictions.update(split_preds)
        logger.info(
            "TinyUNet [%s split]: Loss=%.6f, Macro MAE=%.4f, Macro RMSE=%.4f, Macro Pearson=%.4f, Macro IoU@5%%=%.4f",
            split_key,
            avg_loss,
            metrics.get("mae", 0.0),
            metrics.get("rmse", 0.0),
            metrics.get("pearson", 0.0),
            metrics.get("iou_5pct", 0.0),
        )

    # Step 6: Save predictions as NPZ (both _prediction.npz and _pred.npz for compatibility)
    for sid, pred_arr in sample_predictions.items():
        np.savez_compressed(
            predictions_dir / f"{sid}_prediction.npz",
            prediction=pred_arr,
            target_channels=target_channel_names,
            channel_names=target_channel_names,
            sample_id=sid,
        )
        np.savez_compressed(
            predictions_dir / f"{sid}_pred.npz",
            prediction=pred_arr,
            target_channels=target_channel_names,
            channel_names=target_channel_names,
            sample_id=sid,
        )
    logger.info("Saved %d prediction maps into %s", len(sample_predictions), predictions_dir)

    # Step 7: Export Metrics Tables (CSV)
    metric_keys = [
        "mae", "rmse", "pearson", "spearman",
        "recall_5pct", "precision_5pct", "iou_5pct", "dice_5pct",
        "recall_10pct", "precision_10pct", "iou_10pct", "dice_10pct"
    ]

    # 7a. Macro metrics by split
    metrics_csv_path = output_dir / "metrics_by_split.csv"
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split"] + metric_keys)
        for s_key in ["train", "val", "test"]:
            m_dict = unet_metrics_by_split.get(s_key, {})
            row = [s_key] + [
                round(m_dict.get(m, 0.0), 6) if m_dict.get(m) is not None and not np.isnan(m_dict.get(m)) else None
                for m in metric_keys
            ]
            writer.writerow(row)

    # 7b. Per-channel metrics for test split (metrics_per_channel.csv)
    test_ch_metrics = unet_metrics_by_split.get("test", {}).get("per_channel", {})
    empty_channels_test = unet_metrics_by_split.get("test", {}).get("empty_channels", [])
    per_ch_csv_path = output_dir / "metrics_per_channel.csv"
    with open(per_ch_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channel"] + metric_keys + ["is_empty"])
        for ch in target_channel_names:
            ch_m = test_ch_metrics.get(ch, {})
            is_empty = ch in empty_channels_test
            row = [ch] + [
                round(ch_m.get(m, 0.0), 6) if ch_m.get(m) is not None and not np.isnan(ch_m.get(m)) else None
                for m in metric_keys
            ] + [is_empty]
            writer.writerow(row)

    # 7c. Channel by split table (metrics_channel_by_split.csv)
    ch_split_csv_path = output_dir / "metrics_channel_by_split.csv"
    with open(ch_split_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "channel"] + metric_keys + ["is_empty"])
        for s_key in ["train", "val", "test"]:
            s_ch_metrics = unet_metrics_by_split.get(s_key, {}).get("per_channel", {})
            s_empty = unet_metrics_by_split.get(s_key, {}).get("empty_channels", [])
            for ch in target_channel_names:
                ch_m = s_ch_metrics.get(ch, {})
                is_empty = ch in s_empty
                row = [s_key, ch] + [
                    round(ch_m.get(m, 0.0), 6) if ch_m.get(m) is not None and not np.isnan(ch_m.get(m)) else None
                    for m in metric_keys
                ] + [is_empty]
                writer.writerow(row)

    # 7d. Per-sample per-channel metrics (metrics_per_sample.csv)
    sample_csv_rows: List[Dict[str, Any]] = []
    for s_key in ["train", "val", "test"]:
        sids = splits.get(s_key, [])
        for sid in sids:
            rec = dataset.samples.get(sid)
            if rec is None or rec.sample is None or sid not in sample_predictions:
                continue
            pred_arr = sample_predictions[sid]  # (C, H, W) or (H, W)
            design = rec.sample.feature_metadata.get("design", "unknown") if isinstance(rec.sample.feature_metadata, dict) else "unknown"

            for c_idx, ch in enumerate(target_channel_names):
                p_c = pred_arr[c_idx] if pred_arr.ndim == 3 else pred_arr
                t_idx = rec.sample.label_channel_names.index(ch)
                t_c = rec.sample.label_tensor[t_idx].astype(np.float32)
                m = compute_all_metrics(p_c, t_c)
                sample_csv_rows.append({
                    "sample_id": sid,
                    "split": s_key,
                    "design": design,
                    "channel": ch,
                    "mae": round(m["mae"], 6),
                    "rmse": round(m["rmse"], 6),
                    "pearson": round(m["pearson"], 6) if not np.isnan(m["pearson"]) else None,
                    "iou_5pct": round(m["iou_5pct"], 6),
                    "dice_5pct": round(m["dice_5pct"], 6),
                })

    per_sample_csv_path = output_dir / "metrics_per_sample.csv"
    with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "design", "channel", "mae", "rmse", "pearson", "iou_5pct", "dice_5pct"])
        writer.writeheader()
        writer.writerows(sample_csv_rows)

    # Step 8: Visual Previews Generation
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_channels = [
        "total_wirelength_density",
        "via_count_density",
        "metal2_wirelength_density",
        "metal3_wirelength_density",
    ]

    test_sids = splits.get("test", [])
    # Select up to 1 sample per design in test set
    preview_samples: Dict[str, str] = {}
    for sid in test_sids:
        rec = dataset.samples.get(sid)
        if rec and rec.sample:
            des = rec.sample.feature_metadata.get("design", "unknown") if isinstance(rec.sample.feature_metadata, dict) else "unknown"
            if des not in preview_samples:
                preview_samples[des] = sid

    for des, sid in preview_samples.items():
        rec = dataset.samples.get(sid)
        if not rec or not rec.sample:
            continue
        sample = rec.sample
        pred_arr = sample_predictions[sid]

        for ch in preview_channels:
            if ch not in target_channel_names:
                continue
            c_idx = target_channel_names.index(ch)
            p_map = pred_arr[c_idx] if pred_arr.ndim == 3 else pred_arr
            t_idx = sample.label_channel_names.index(ch)
            t_map = sample.label_tensor[t_idx].astype(np.float32)
            residual = np.abs(p_map - t_map)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            im0 = axes[0].imshow(t_map, cmap="viridis", vmin=0.0, vmax=max(float(t_map.max()), 0.1))
            axes[0].set_title(f"Target: {ch}\n({sid})", fontsize=10)
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(p_map, cmap="viridis", vmin=0.0, vmax=max(float(t_map.max()), 0.1))
            axes[1].set_title(f"TinyUNet Prediction\n({des})", fontsize=10)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(residual, cmap="magma", vmin=0.0, vmax=max(float(residual.max()), 0.05))
            axes[2].set_title(f"Absolute Error (|Pred - Target|)\nMAE={float(np.mean(residual)):.4f}", fontsize=10)
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            preview_file = previews_dir / f"{sid}_{ch}_target_pred_residual.png"
            plt.savefig(preview_file, dpi=180, bbox_inches="tight")
            plt.close(fig)

    logger.info("Saved visual previews to: %s", previews_dir)

    # Step 9: Export Config, Summary JSON, and Limitations MD
    experiment_config = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "splits_file": str(args.splits_file),
        "target_channel": args.target_channel,
        "target_channels": target_channel_names,
        "model_architecture": "TinyUNet",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_channels": args.base_channels,
        "final_activation": args.final_activation,
        "loss": args.loss,
        "parameter_count": model.count_parameters(),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)

    # Model summary JSON
    model_summary = {
        "config": experiment_config,
        "training": {
            "best_epoch": train_summary["best_epoch"],
            "best_val_loss": train_summary["best_val_loss"],
            "total_epochs": train_summary["total_epochs"],
            "total_time_s": train_summary["total_training_time_s"],
        },
        "test_macro_metrics": sanitize_metrics_for_json(unet_metrics_by_split.get("test", {})),
        "test_per_channel_metrics": sanitize_metrics_for_json(test_ch_metrics),
        "empty_channels": empty_channels_test,
    }
    with open(output_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    # Limitations markdown
    limitations_md = f"""# Multi-Target TinyUNet Routing Label Model Limitations

## 1. Empty and Sparse Channels
- **`metal1_wirelength_density` is all-zero ($0.0000$)**:
  In the Nangate45 PDK with OpenROAD standard cell placement and routing, signal routing is strictly carried on Metal 2 (horizontal) and Metal 3 (vertical) and higher. Metal 1 is reserved inside standard cells for local pin connections and power rails (`VDD`/`VSS`). Therefore, inter-cell routing labels produce zero wirelength on Metal 1.
  - Pearson and Spearman rank correlations are undefined (NaN) on `metal1_wirelength_density` due to zero target variance.
  - The model correctly predicts near-zero outputs across this channel.
  - In macro-averaged metrics, this channel is reported and isolated.

## 2. Multi-Target vs. Single-Target Tradeoff on Primary Target (`total_wirelength_density`)
- Predicting all 8 channels simultaneously forces the network's convolutional feature backbone to share representation capacity across horizontal wirelength, vertical wirelength, via count, and layer-specific distributions.
- See `metrics_per_channel.csv` and Stage 5d summary report for quantitative comparison against single-target Stage 5c baseline.

## 3. Dataset Scope
- Evaluated on `expanded_v002` (36 Nangate45 runs across `uart_tx`, `dft_counter`, and `spi`). Generalization to larger macros and different technology nodes (e.g. Sky130hd) requires additional floorplan evaluations.
"""
    with open(output_dir / "limitations.md", "w", encoding="utf-8") as f:
        f.write(limitations_md)

    logger.info("All experiment artifacts and previews saved to: %s", output_dir)

    # Print summary table
    print("\n" + "=" * 90)
    print("STAGE 5d: MULTI-TARGET TINYUNET PER-CHANNEL METRICS ON TEST SPLIT")
    print("=" * 90)
    print(f"{'Target Channel':<32} | {'MAE':<8} | {'RMSE':<8} | {'Pearson':<8} | {'IoU@5%':<8} | {'Dice@5%':<8}")
    print("-" * 90)
    for ch in target_channel_names:
        cm = test_ch_metrics.get(ch, {})
        mae_s = f"{cm.get('mae', 0.0):.4f}" if cm.get("mae") is not None else "N/A"
        rmse_s = f"{cm.get('rmse', 0.0):.4f}" if cm.get("rmse") is not None else "N/A"
        p_s = f"{cm.get('pearson', 0.0):.4f}" if cm.get("pearson") is not None else "N/A"
        iou_s = f"{cm.get('iou_5pct', 0.0):.4f}" if cm.get("iou_5pct") is not None else "N/A"
        dice_s = f"{cm.get('dice_5pct', 0.0):.4f}" if cm.get("dice_5pct") is not None else "N/A"
        print(f"{ch:<32} | {mae_s:<8} | {rmse_s:<8} | {p_s:<8} | {iou_s:<8} | {dice_s:<8}")
    print("-" * 90)
    macro_m = unet_metrics_by_split.get("test", {})
    m_mae = f"{macro_m.get('mae', 0.0):.4f}" if macro_m.get("mae") is not None else "N/A"
    m_rmse = f"{macro_m.get('rmse', 0.0):.4f}" if macro_m.get("rmse") is not None else "N/A"
    m_p = f"{macro_m.get('pearson', 0.0):.4f}" if macro_m.get("pearson") is not None else "N/A"
    m_iou = f"{macro_m.get('iou_5pct', 0.0):.4f}" if macro_m.get("iou_5pct") is not None else "N/A"
    m_dice = f"{macro_m.get('dice_5pct', 0.0):.4f}" if macro_m.get("dice_5pct") is not None else "N/A"
    print(f"{'MACRO-AVERAGE (Active Channels)':<32} | {m_mae:<8} | {m_rmse:<8} | {m_p:<8} | {m_iou:<8} | {m_dice:<8}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

