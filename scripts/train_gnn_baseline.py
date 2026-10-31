#!/usr/bin/env python3
"""
CLI Orchestration Script for GraphSAGE Baseline on expanded_v002 (Stage 6e).

Executes:
1. Dataset loading and partition resolution via PyG DataLoader.
2. Baseline benchmarking:
   - Mean baseline (empirical train target mean).
   - Heuristic baseline (simple_weighted_sum).
   - TinyUNet baseline (from Stage 5c calibrated unet_nosigmoid experiment).
3. GraphSAGE + GraphToGridPooling model training with validation early stopping and checkpointing.
4. Comprehensive spatial metric evaluation across train, val, and holdout test splits.
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

from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.evaluation.heuristics import get_standard_heuristics
from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json
from dft_hotspot.graphs.dataset import create_pyg_dataloaders
from dft_hotspot.ml.train_gnn import evaluate_gnn_model, set_seed, train_gnn
from dft_hotspot.ml.train_unet import compute_multichannel_metrics
from dft_hotspot.models.gnn import GNNHotspotPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gnn_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE hotspot prediction baseline on expanded_v002."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/datasets/expanded_v002",
        help="Path to dataset directory containing samples/ and splits_v001.json.",
    )
    parser.add_argument(
        "--splits-file",
        type=str,
        default=None,
        help="Path to splits_v001.json (defaults to dataset_dir/splits_v001.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/gnn/expanded_v002_gnn_baseline",
        help="Directory to save checkpoints, logs, and prediction maps.",
    )
    parser.add_argument(
        "--target-channel",
        type=str,
        default="total_wirelength_density",
        help="Target routing channel name, 'all', or comma-separated list.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for graph data loading (default: 4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--gnn-hidden",
        type=int,
        default=64,
        help="Hidden channel dimension for GraphSAGE (default: 64).",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=3,
        help="Number of GraphSAGE message passing layers (default: 3).",
    )
    parser.add_argument(
        "--gnn-aggr",
        type=str,
        default="mean",
        choices=["mean", "max", "sum"],
        help="GraphSAGE neighborhood aggregator (default: mean).",
    )
    parser.add_argument(
        "--pooling-aggr",
        type=str,
        default="mean",
        choices=["mean", "sum", "cat_mean_sum"],
        help="Graph-to-grid spatial pooling aggregator (default: mean).",
    )
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="Disable the cell occupancy density channel in pooling.",
    )
    parser.add_argument(
        "--head-hidden",
        type=int,
        default=32,
        help="Hidden channel dimension for spatial refinement head (default: 32).",
    )
    parser.add_argument(
        "--final-activation",
        type=str,
        default="none",
        choices=["none", "sigmoid", "relu"],
        help="Activation function for the output layer (default: none).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "mae", "l1", "huber"],
        help="Loss criterion (default: mse).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cuda:0 or cpu; default: auto-detect).",
    )
    return parser.parse_args()


def evaluate_heuristic_baseline(
    dataset: PhysicalDesignDataset,
    splits: Dict[str, List[str]],
    target_channel: str = "total_wirelength_density",
    heuristic_name: str = "simple_weighted_sum",
) -> Dict[str, Dict[str, float]]:
    """Evaluates heuristic baseline across train, val, and test splits."""
    heuristics = {b.name: b for b in get_standard_heuristics()}
    baseline = heuristics.get(heuristic_name)
    if baseline is None:
        return {}

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

            t_idx = sample.label_channel_names.index(target_channel)
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
    target_channel: str = "total_wirelength_density",
) -> Dict[str, Dict[str, float]]:
    """Evaluates empirical train-mean baseline."""
    train_sids = splits.get("train", [])
    vals: List[float] = []
    for sid in train_sids:
        rec = dataset.samples.get(sid)
        if rec is None or rec.sample is None:
            continue
        t_idx = rec.sample.label_channel_names.index(target_channel)
        vals.extend(rec.sample.label_tensor[t_idx].ravel().tolist())
    train_mean = float(np.mean(vals)) if vals else 0.0

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
            t_idx = sample.label_channel_names.index(target_channel)
            target = sample.label_tensor[t_idx].astype(np.float32)
            pred = np.full(target.shape, train_mean, dtype=np.float32)

            preds_list.append(pred)
            targets_list.append(target)

        if preds_list and targets_list:
            concat_p = np.stack(preds_list, axis=0)
            concat_t = np.stack(targets_list, axis=0)
            split_metrics[split_key] = compute_all_metrics(concat_p, concat_t)
        else:
            split_metrics[split_key] = {}

    return split_metrics


def load_tiny_unet_baseline(unet_exp_dir: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Loads existing TinyUNet holdout test results from Stage 5c if available."""
    metrics_file = unet_exp_dir / "metrics_by_split.csv"
    if not metrics_file.is_file():
        return None

    results: Dict[str, Dict[str, float]] = {}
    with open(metrics_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "")
            results[split] = {
                k: float(v) if v and v != "nan" and v != "NaN" else float("nan")
                for k, v in row.items()
                if k != "split"
            }
    return results


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    splits_file = Path(args.splits_file) if args.splits_file else dataset_dir / "splits_v001.json"

    # 1. Device Selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cuda_device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    logger.info("Using execution device: %s (%s)", device, cuda_device_name)

    # 2. Build PyG DataLoaders
    target_channel = args.target_channel
    logger.info("Loading PyG graph dataset from %s for target '%s'...", dataset_dir, target_channel)

    train_loader, val_loader, test_loader, split_info = create_pyg_dataloaders(
        dataset_dir=dataset_dir,
        splits_path=splits_file,
        target_channel=target_channel,
        batch_size=args.batch_size,
        preload=True,
    )

    resolved_channels = split_info["target_channels"]
    out_channels = len(resolved_channels)
    is_multi_target = out_channels > 1

    logger.info(
        "PyG DataLoaders initialized: Train=%d, Val=%d, Test=%d | Out Channels=%d",
        split_info["train_samples"],
        split_info["val_samples"],
        split_info["test_samples"],
        out_channels,
    )

    # 3. Load Dataset Collection for Heuristic and Mean Baselines
    logger.info("Evaluating reference baselines (Mean, Heuristic)...")
    with open(splits_file, "r", encoding="utf-8") as f:
        splits_json = json.load(f)

    raw_ds = PhysicalDesignDataset.load(dataset_dir / "dataset_manifest.json", load_tensors=True, load_scan=True)
    eval_target = "total_wirelength_density" if "total_wirelength_density" in resolved_channels else resolved_channels[0]
    mean_baseline_metrics = evaluate_mean_baseline(raw_ds, splits_json, target_channel=eval_target)
    heuristic_baseline_metrics = evaluate_heuristic_baseline(
        raw_ds, splits_json, target_channel=eval_target, heuristic_name="simple_weighted_sum"
    )

    # Try loading existing TinyUNet baseline from Stage 5c
    unet_baseline_dir = Path("experiments/unet/expanded_v002_tiny_unet_nosigmoid")
    tiny_unet_metrics = load_tiny_unet_baseline(unet_baseline_dir)

    # 4. Instantiate GNNHotspotPredictor Model
    include_density = not args.no_density
    model = GNNHotspotPredictor(
        node_in_channels=10,
        gnn_hidden_channels=args.gnn_hidden,
        gnn_out_channels=args.gnn_hidden,
        num_gnn_layers=args.gnn_layers,
        gnn_aggr=args.gnn_aggr,
        grid_size=64,
        pooling_aggr=args.pooling_aggr,
        include_density=include_density,
        head_hidden_channels=args.head_hidden,
        out_channels=out_channels,
        final_activation=args.final_activation,
    ).to(device)

    param_count = model.count_parameters()
    logger.info("Constructed GNNHotspotPredictor model: %d trainable parameters", param_count)

    # 5. Train Model
    t0 = time.time()
    train_summary = train_gnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        loss_name=args.loss,
        clamp_output=(args.final_activation == "none"),
    )
    total_train_time = round(time.time() - t0, 2)
    logger.info("GNN training finished in %.2f seconds (Best epoch: %d)", total_train_time, train_summary["best_epoch"])

    # 6. Evaluate Model on All Splits
    loss_criterion = nn.MSELoss() if args.loss == "mse" else (nn.L1Loss() if args.loss in ("mae", "l1") else nn.SmoothL1Loss())
    clamp_eval = (args.final_activation == "none")

    train_loss, train_metrics, train_preds = evaluate_gnn_model(
        model, train_loader, loss_criterion, device, clamp_output=clamp_eval, channel_names=resolved_channels
    )
    val_loss, val_metrics, val_preds = evaluate_gnn_model(
        model, val_loader, loss_criterion, device, clamp_output=clamp_eval, channel_names=resolved_channels
    )
    test_loss, test_metrics, test_preds = evaluate_gnn_model(
        model, test_loader, loss_criterion, device, clamp_output=clamp_eval, channel_names=resolved_channels
    )

    all_preds_dict = {**train_preds, **val_preds, **test_preds}

    # 7. Save Predictions (.npz per sample)
    for sid, pred_arr in all_preds_dict.items():
        pred_path = predictions_dir / f"{sid}_prediction.npz"
        np.savez_compressed(pred_path, prediction=pred_arr, channel_names=np.array(resolved_channels))

    logger.info("Saved %d prediction maps to %s", len(all_preds_dict), predictions_dir)

    # 8. Export Metrics by Split CSV
    metrics_by_split_path = output_dir / "metrics_by_split.csv"
    fieldnames = [
        "split", "loss", "mae", "rmse", "pearson", "spearman",
        "recall_5pct", "precision_5pct", "iou_5pct", "dice_5pct",
        "recall_10pct", "precision_10pct", "iou_10pct", "dice_10pct",
    ]

    split_rows = [
        {"split": "train", "loss": train_loss, **{k: train_metrics.get(k) for k in fieldnames if k not in ("split", "loss")}},
        {"split": "val", "loss": val_loss, **{k: val_metrics.get(k) for k in fieldnames if k not in ("split", "loss")}},
        {"split": "test", "loss": test_loss, **{k: test_metrics.get(k) for k in fieldnames if k not in ("split", "loss")}},
    ]

    with open(metrics_by_split_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

    # 9. Export Baseline Comparison CSV (Test Split)
    baseline_comp_path = output_dir / "baseline_comparison.csv"
    comp_fields = [
        "model", "test_loss", "mae", "rmse", "pearson", "spearman",
        "recall_5pct", "precision_5pct", "iou_5pct", "dice_5pct",
    ]

    mean_test = mean_baseline_metrics.get("test", {})
    heur_test = heuristic_baseline_metrics.get("test", {})
    unet_test = tiny_unet_metrics.get("test", {}) if tiny_unet_metrics else {}

    comp_rows = [
        {
            "model": "Mean Baseline",
            "test_loss": "",
            **{k: mean_test.get(k) for k in comp_fields if k not in ("model", "test_loss")},
        },
        {
            "model": "Heuristic (simple_weighted_sum)",
            "test_loss": "",
            **{k: heur_test.get(k) for k in comp_fields if k not in ("model", "test_loss")},
        },
    ]

    if unet_test:
        comp_rows.append({
            "model": "TinyUNet Baseline (Stage 5c)",
            "test_loss": unet_test.get("loss", ""),
            **{k: unet_test.get(k) for k in comp_fields if k not in ("model", "test_loss")},
        })

    comp_rows.append({
        "model": "GraphSAGE Baseline (Ours, Stage 6e)",
        "test_loss": test_loss,
        **{k: test_metrics.get(k) for k in comp_fields if k not in ("model", "test_loss")},
    })

    with open(baseline_comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        writer.writerows(comp_rows)

    # 10. Save Full Experiment Summary JSON
    config_record = {
        "dataset_dir": str(dataset_dir),
        "splits_file": str(splits_file),
        "output_dir": str(output_dir),
        "target_channel": target_channel,
        "resolved_channels": resolved_channels,
        "model_architecture": "GNNHotspotPredictor",
        "node_in_channels": 10,
        "gnn_hidden_channels": args.gnn_hidden,
        "gnn_layers": args.gnn_layers,
        "gnn_aggr": args.gnn_aggr,
        "pooling_aggr": args.pooling_aggr,
        "include_density": include_density,
        "head_hidden_channels": args.head_hidden,
        "out_channels": out_channels,
        "final_activation": args.final_activation,
        "loss": args.loss,
        "parameter_count": param_count,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": cuda_device_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    full_summary = {
        "config": config_record,
        "training": train_summary,
        "metrics": {
            "train": sanitize_metrics_for_json(train_metrics),
            "val": sanitize_metrics_for_json(val_metrics),
            "test": sanitize_metrics_for_json(test_metrics),
        },
        "baselines": {
            "mean_baseline": sanitize_metrics_for_json(mean_baseline_metrics),
            "heuristic_simple_weighted_sum": sanitize_metrics_for_json(heuristic_baseline_metrics),
            "tiny_unet": sanitize_metrics_for_json(tiny_unet_metrics) if tiny_unet_metrics else None,
        },
    }

    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    # 11. Print Formatted Comparison Table to Stdout
    print("\n" + "=" * 105)
    print("STAGE 6e: STANDALONE GRAPHSAGE BASELINE BENCHMARK RESULTS (Holdout Test Split)")
    print("=" * 105)
    header = f"{'Model':<35} | {'MAE':<7} | {'RMSE':<7} | {'Pearson r':<9} | {'Top-5% Rec':<10} | {'IoU @ 5%':<8} | {'Dice @ 5%':<8}"
    print(header)
    print("-" * 105)

    for r in comp_rows:
        m_name = r["model"]
        mae_str = f"{float(r['mae']):.4f}" if r.get("mae") is not None and not np.isnan(float(r["mae"])) else "N/A"
        rmse_str = f"{float(r['rmse']):.4f}" if r.get("rmse") is not None and not np.isnan(float(r["rmse"])) else "N/A"
        pr_str = f"{float(r['pearson']):.4f}" if r.get("pearson") is not None and not np.isnan(float(r["pearson"])) else "N/A"
        rec_str = f"{float(r['recall_5pct']) * 100:.2f}%" if r.get("recall_5pct") is not None and not np.isnan(float(r["recall_5pct"])) else "N/A"
        iou_str = f"{float(r['iou_5pct']):.4f}" if r.get("iou_5pct") is not None and not np.isnan(float(r["iou_5pct"])) else "N/A"
        dice_str = f"{float(r['dice_5pct']):.4f}" if r.get("dice_5pct") is not None and not np.isnan(float(r["dice_5pct"])) else "N/A"
        print(f"{m_name:<35} | {mae_str:<7} | {rmse_str:<7} | {pr_str:<9} | {rec_str:<10} | {iou_str:<8} | {dice_str:<8}")

    print("=" * 105 + "\n")
    logger.info("Stage 6e GNN baseline benchmark complete. Artifacts saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
