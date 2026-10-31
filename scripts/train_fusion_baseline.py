#!/usr/bin/env python3
"""
CLI Orchestration Script for Multimodal CNN + GNN Hotspot Prediction (Stage 7b).

Executes:
1. Paired multimodal dataset loading and split resolution (Train: 24, Val: 6, Test: 6).
2. Model construction: CNNGNNFusionModel with spatial CNN, GraphSAGE encoder,
   GraphToGridPooling, FusionNeck, and hierarchical U-Net decoder.
3. End-to-end training with validation-based early stopping, ReduceLROnPlateau,
   and model checkpointing.
4. Comprehensive multi-model benchmarking against:
   - Empirical Mean Baseline
   - Heuristic Baseline (simple_weighted_sum)
   - Standalone GraphSAGE Baseline (Stage 6e)
   - Standalone TinyUNet Baseline (Stage 5c)
5. Export of experiment summary, checkpoints, history, predictions, and comparison tables.
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
from dft_hotspot.ml.torch_multimodal_dataset import create_multimodal_dataloaders
from dft_hotspot.ml.train_fusion import evaluate_fusion_model, train_fusion
from dft_hotspot.ml.train_unet import compute_multichannel_metrics, set_seed
from dft_hotspot.models.fusion import CNNGNNFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_fusion_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Multimodal CNN + GNN Hotspot Prediction Model on expanded_v002."
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
        default="experiments/fusion/expanded_v002_fusion_baseline",
        help="Directory to save checkpoints, logs, and prediction maps.",
    )
    parser.add_argument(
        "--target-channel",
        type=str,
        default="total_wirelength_density",
        help="Target routing channel name, 'all', or comma-separated list.",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="concat",
        choices=["concat", "gated", "add"],
        help="Cross-modal fusion neck mode (default: concat).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "spatial_only", "graph_only"],
        help="Operational mode for ablation study (default: multimodal).",
    )
    parser.add_argument(
        "--include-scan",
        action="store_true",
        default=True,
        help="Include 5 DFT/scan feature channels in spatial input (12 channels total).",
    )
    parser.add_argument(
        "--no-scan",
        action="store_false",
        dest="include_scan",
        help="Ablation: Use only 7 physical layout channels (no scan channels).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data loading (default: 4).",
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
        "--base-channels",
        type=int,
        default=16,
        help="Base feature channel capacity (default: 16).",
    )
    parser.add_argument(
        "--gnn-hidden",
        type=int,
        default=64,
        help="Hidden channel dimension for GraphSAGE message passing (default: 64).",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=3,
        help="Number of GraphSAGE message passing layers (default: 3).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or auto-detected).",
    )
    return parser.parse_args()


def load_baselines(dataset_dir: Path, target_channel: str = "total_wirelength_density") -> Dict[str, Dict[str, Any]]:
    """Loads reference baselines for comparison on the test split."""
    results: Dict[str, Dict[str, Any]] = {}

    manifest_path = dataset_dir / "dataset_manifest.json"
    splits_path = dataset_dir / "splits_v001.json"

    if not manifest_path.is_file() or not splits_path.is_file():
        return results

    try:
        dataset = PhysicalDesignDataset.load(manifest_path, load_tensors=True, load_scan=True)
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        train_ids = splits_data.get("train", [])
        test_ids = splits_data.get("test", [])

        train_samples = [dataset.get_sample(sid) for sid in train_ids if dataset.get_sample(sid) is not None]
        test_samples = [dataset.get_sample(sid) for sid in test_ids if dataset.get_sample(sid) is not None]

        # 1. Mean Baseline
        train_targets = []
        for s in train_samples:
            lt = s.get_routing_label_tensor()
            idx = s.label_channel_names.index(target_channel) if target_channel in s.label_channel_names else 0
            train_targets.append(lt[idx])
        train_mean_val = float(np.mean(np.stack(train_targets, axis=0)))

        test_targets = []
        for s in test_samples:
            lt = s.get_routing_label_tensor()
            idx = s.label_channel_names.index(target_channel) if target_channel in s.label_channel_names else 0
            test_targets.append(lt[idx])
        test_targets_all = np.stack(test_targets, axis=0)

        mean_preds = np.full_like(test_targets_all, fill_value=train_mean_val)
        results["mean_baseline"] = compute_all_metrics(mean_preds.reshape(-1), test_targets_all.reshape(-1))

        # 2. Heuristic Baseline (simple_weighted_sum)
        heuristics = get_standard_heuristics()
        h_func = heuristics.get("simple_weighted_sum")
        if h_func is not None:
            h_preds = []
            for s in test_samples:
                ft = s.get_physical_feature_tensor()
                h_map = h_func(ft, s.feature_channel_names)
                h_preds.append(h_map)
            h_preds_all = np.stack(h_preds, axis=0)
            results["heuristic_weighted_sum"] = compute_all_metrics(h_preds_all.reshape(-1), test_targets_all.reshape(-1))

    except Exception as e:
        logger.warning("Could not compute reference baselines: %s", e)

    # 3. Known Stage 5c Calibrated TinyUNet and Stage 6e GraphSAGE Baselines
    results["tiny_unet_baseline"] = {
        "mae": 0.0095,
        "rmse": 0.0399,
        "pearson": 0.8921,
        "spearman": 0.5480,
        "top5_recall": 0.8316,
        "iou_5pct": 0.8316,
        "dice_5pct": 0.9081,
    }

    results["gnn_baseline"] = {
        "mae": 0.0131,
        "rmse": 0.0514,
        "pearson": 0.8128,
        "spearman": 0.4497,
        "top5_recall": 0.6810,
        "iou_5pct": 0.5163,
        "dice_5pct": 0.6810,
    }

    return results


def main() -> int:
    if not _HAS_TORCH:
        logger.error("PyTorch and torch_geometric are required to run train_fusion_baseline.py.")
        return 1

    args = parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("===========================================================================")
    logger.info("STAGE 7b: MULTIMODAL CNN + GNN FUSION MODEL TRAINING")
    logger.info("===========================================================================")
    logger.info("Dataset Directory    : %s", dataset_dir)
    logger.info("Output Directory     : %s", output_dir)
    logger.info("Target Channel       : %s", args.target_channel)
    logger.info("Fusion Mode          : %s", args.fusion_mode)
    logger.info("Ablation Mode        : %s", args.mode)
    logger.info("Include Scan Features: %s (Spatial in: %d ch)", args.include_scan, 12 if args.include_scan else 7)
    logger.info("Batch Size / Epochs  : %d / %d", args.batch_size, args.epochs)
    logger.info("Base Channels / GNN  : %d / %d (layers: %d)", args.base_channels, args.gnn_hidden, args.gnn_layers)
    logger.info("Compute Device       : %s", device)
    logger.info("===========================================================================")

    # 1. Build DataLoaders
    train_loader, val_loader, test_loader, split_info = create_multimodal_dataloaders(
        dataset_dir=dataset_dir,
        splits_path=args.splits_file,
        target_channel=args.target_channel,
        include_scan=args.include_scan,
        batch_size=args.batch_size,
        shuffle_train=True,
        preload=True,
    )

    out_channels = len(split_info["target_channels"])
    spatial_in_channels = 12 if args.include_scan else 7

    # 2. Construct Model
    model = CNNGNNFusionModel(
        spatial_in_channels=spatial_in_channels,
        base_channels=args.base_channels,
        node_in_channels=10,
        gnn_hidden_channels=args.gnn_hidden,
        gnn_out_channels=args.gnn_hidden,
        num_gnn_layers=args.gnn_layers,
        fusion_mode=args.fusion_mode,
        out_channels=out_channels,
        final_activation="none",
        default_mode=args.mode,
    )
    total_params = model.count_parameters()
    logger.info("Initialized CNNGNNFusionModel with %d trainable parameters (~%.1f MB)", total_params, total_params * 4 / (1024 * 1024))
    logger.info("Branch parameters: %s", model.count_branch_parameters())

    # 3. Train Model
    results = train_fusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        mode=args.mode,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        channel_names=split_info["target_channels"],
    )

    # 4. Evaluate across all splits with the best model
    train_eval_loader, _, _, _ = create_multimodal_dataloaders(
        dataset_dir=dataset_dir,
        splits_path=args.splits_file,
        target_channel=args.target_channel,
        include_scan=args.include_scan,
        batch_size=args.batch_size,
        shuffle_train=False,
        preload=True,
    )

    criterion = nn.MSELoss()
    train_loss, train_metrics, train_preds = evaluate_fusion_model(
        model=model,
        dataloader=train_eval_loader,
        criterion=criterion,
        device=device,
        mode=args.mode,
        channel_names=split_info["target_channels"],
    )
    val_loss, val_metrics, val_preds = evaluate_fusion_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        mode=args.mode,
        channel_names=split_info["target_channels"],
    )
    test_loss, test_metrics, test_preds = evaluate_fusion_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        mode=args.mode,
        channel_names=split_info["target_channels"],
    )

    # Save per-sample predictions
    all_sample_preds = {**train_preds, **val_preds, **test_preds}
    for sid, pred_arr in all_sample_preds.items():
        np.savez_compressed(preds_dir / f"{sid}_pred.npz", prediction=pred_arr)
    logger.info("Saved %d prediction maps to %s", len(all_sample_preds), preds_dir)

    # 5. Export metrics_by_split.csv
    metrics_csv = output_dir / "metrics_by_split.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "num_samples", "loss", "mae", "rmse", "pearson", "spearman", "top5_recall", "iou_5pct", "dice_5pct"
        ])
        for sname, snum, sloss, sm in [
            ("train", split_info["train_samples"], train_loss, train_metrics),
            ("val", split_info["val_samples"], val_loss, val_metrics),
            ("test", split_info["test_samples"], test_loss, test_metrics),
        ]:
            writer.writerow([
                sname,
                snum,
                f"{sloss:.6f}",
                f"{sm.get('mae', 0.0):.6f}",
                f"{sm.get('rmse', 0.0):.6f}",
                f"{sm.get('pearson', 0.0):.4f}",
                f"{sm.get('spearman', 0.0):.4f}",
                f"{sm.get('top5_recall', 0.0):.4f}",
                f"{sm.get('iou_5pct', 0.0):.4f}",
                f"{sm.get('dice_5pct', 0.0):.4f}",
            ])

    # 6. Export baseline_comparison.csv
    baselines = load_baselines(dataset_dir, target_channel=split_info["target_channels"][0])
    comp_csv = output_dir / "baseline_comparison.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_family", "model_variant", "parameters", "test_mae", "test_rmse", "pearson", "spearman", "top5_recall", "iou_5pct", "dice_5pct"
        ])
        if "mean_baseline" in baselines:
            mb = baselines["mean_baseline"]
            writer.writerow(["empirical", "mean_baseline", 0, f"{mb['mae']:.6f}", f"{mb['rmse']:.6f}", "NaN", "NaN", f"{mb['top5_recall']:.4f}", f"{mb['iou_5pct']:.4f}", f"{mb['dice_5pct']:.4f}"])
        if "heuristic_weighted_sum" in baselines:
            hb = baselines["heuristic_weighted_sum"]
            writer.writerow(["heuristic", "simple_weighted_sum", 0, f"{hb['mae']:.6f}", f"{hb['rmse']:.6f}", f"{hb['pearson']:.4f}", f"{hb['spearman']:.4f}", f"{hb['top5_recall']:.4f}", f"{hb['iou_5pct']:.4f}", f"{hb['dice_5pct']:.4f}"])
        if "gnn_baseline" in baselines:
            gb = baselines["gnn_baseline"]
            writer.writerow(["gnn", "GraphSAGE_standalone", 42321, f"{gb['mae']:.6f}", f"{gb['rmse']:.6f}", f"{gb['pearson']:.4f}", f"{gb['spearman']:.4f}", f"{gb['top5_recall']:.4f}", f"{gb['iou_5pct']:.4f}", f"{gb['dice_5pct']:.4f}"])
        if "tiny_unet_baseline" in baselines:
            ub = baselines["tiny_unet_baseline"]
            writer.writerow(["cnn", "TinyUNet_standalone", 484033, f"{ub['mae']:.6f}", f"{ub['rmse']:.6f}", f"{ub['pearson']:.4f}", f"{ub['spearman']:.4f}", f"{ub['top5_recall']:.4f}", f"{ub['iou_5pct']:.4f}", f"{ub['dice_5pct']:.4f}"])

        # Ours
        writer.writerow([
            "multimodal_fusion",
            f"CNNGNN_{args.fusion_mode}_{args.mode}",
            total_params,
            f"{test_metrics.get('mae', 0.0):.6f}",
            f"{test_metrics.get('rmse', 0.0):.6f}",
            f"{test_metrics.get('pearson', 0.0):.4f}",
            f"{test_metrics.get('spearman', 0.0):.4f}",
            f"{test_metrics.get('top5_recall', 0.0):.4f}",
            f"{test_metrics.get('iou_5pct', 0.0):.4f}",
            f"{test_metrics.get('dice_5pct', 0.0):.4f}",
        ])

    # 7. Export experiment_summary.json
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_name": output_dir.name,
        "dataset_dir": str(dataset_dir),
        "split_info": split_info,
        "model_architecture": model.summary(),
        "total_parameters": total_params,
        "training": {
            "num_epochs": args.epochs,
            "best_epoch": results["best_epoch"],
            "total_epochs": results["total_epochs"],
            "total_training_time_s": results["total_training_time"],
            "best_val_loss": results["best_val_loss"],
            "initial_lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "fusion_mode": args.fusion_mode,
            "mode": args.mode,
            "include_scan": args.include_scan,
        },
        "metrics": {
            "train": sanitize_metrics_for_json(train_metrics),
            "val": sanitize_metrics_for_json(val_metrics),
            "test": sanitize_metrics_for_json(test_metrics),
        },
    }
    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("===========================================================================")
    logger.info("STAGE 7b TRAINING AND EVALUATION SUMMARY")
    logger.info("===========================================================================")
    logger.info("Total Epochs Trained : %d (Best Epoch: %d)", results["total_epochs"], results["best_epoch"])
    logger.info("Total Training Time  : %.2f seconds", results["total_training_time"])
    logger.info("Holdout Test Metrics :")
    logger.info("  MAE                : %.6f", test_metrics.get("mae", 0.0))
    logger.info("  RMSE               : %.6f", test_metrics.get("rmse", 0.0))
    logger.info("  Pearson r          : %.4f", test_metrics.get("pearson", 0.0))
    logger.info("  Spearman rho       : %.4f", test_metrics.get("spearman", 0.0))
    logger.info("  Top-5%% Recall     : %.2f%%", test_metrics.get("top5_recall", 0.0) * 100)
    logger.info("  IoU @ 5%% Hotspot  : %.4f", test_metrics.get("iou_5pct", 0.0))
    logger.info("  Dice @ 5%% Hotspot : %.4f", test_metrics.get("dice_5pct", 0.0))
    logger.info("Artifacts saved to   : %s", output_dir)
    logger.info("===========================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
