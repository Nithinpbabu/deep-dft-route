#!/usr/bin/env python3
"""
Multi-PDK Cross-Technology Training Engine (Stage 8).
Trains PDK-conditioned Multimodal CNN + GNN Fusion Model on combined 45nm + 14nm data.
Evaluates out-of-sample performance separately on 45nm holdout (SPI) and 14nm holdout (RISCY).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dft_hotspot.evaluation.metrics import (
    pearson_corr,
    spearman_corr,
    mae,
    rmse,
    topk_recall,
    iou_topk,
)
from dft_hotspot.ml.torch_multimodal_dataset import create_multimodal_dataloaders
from dft_hotspot.ml.train_fusion import evaluate_fusion_model, train_fusion
from dft_hotspot.ml.train_unet import set_seed
from dft_hotspot.models.fusion import CNNGNNFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_multi_pdk")


def evaluate_per_pdk(model: nn.Module, test_loader, device: torch.device) -> Dict[str, Any]:
    """Evaluates the model separately on 45nm and 14nm test samples."""
    model.eval()
    pdk_results: Dict[str, List[Dict[str, float]]] = {"nangate45": [], "circuitnet_14nm": []}

    with torch.no_grad():
        for batch in test_loader:
            x_spatial = batch["x_spatial"].to(device) if batch["x_spatial"] is not None else None
            graph = batch["graph"].to(device) if batch["graph"] is not None else None
            targets = batch["y"].to(device)
            sample_ids = batch["sample_ids"]
            pdks = batch["pdks"]

            preds = model(x_spatial=x_spatial, graph_data=graph, mode="multimodal")

            preds_np = preds.squeeze(1).cpu().numpy()
            targets_np = targets.squeeze(1).cpu().numpy()

            for i in range(len(sample_ids)):
                p = preds_np[i]
                t = targets_np[i]
                pdk = "circuitnet_14nm" if "14" in pdks[i] else "nangate45"

                res = {
                    "sample_id": sample_ids[i],
                    "pdk": pdk,
                    "pearson": float(pearson_corr(p, t)),
                    "spearman": float(spearman_corr(p, t)),
                    "mae": float(mae(p, t)),
                    "rmse": float(rmse(p, t)),
                    "recall_top5": float(topk_recall(p, t, percent=0.05)),
                    "iou_top5": float(iou_topk(p, t, percent=0.05)),
                    "recall_top10": float(topk_recall(p, t, percent=0.10)),
                }
                pdk_results[pdk].append(res)

    summary = {}
    for pdk, items in pdk_results.items():
        if items:
            summary[pdk] = {
                "count": len(items),
                "avg_pearson": float(np.mean([x["pearson"] for x in items])),
                "avg_spearman": float(np.mean([x["spearman"] for x in items])),
                "avg_mae": float(np.mean([x["mae"] for x in items])),
                "avg_rmse": float(np.mean([x["rmse"] for x in items])),
                "avg_recall_top5": float(np.mean([x["recall_top5"] for x in items])),
                "avg_iou_top5": float(np.mean([x["iou_top5"] for x in items])),
                "avg_recall_top10": float(np.mean([x["recall_top10"] for x in items])),
                "samples": items,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-PDK PDK-Conditioned Model Training")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/datasets/multi_pdk_v001")
    parser.add_argument("--output-dir", type=str, default="experiments/fusion/multi_pdk_conditioned_v001")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", type=str, default="correlation_aware", choices=["mse", "correlation_aware"])
    parser.add_argument("--lambda-corr", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = ROOT_DIR / args.dataset_dir
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" DEEPDFT-ROUTE: MULTI-PDK CONDITIONED TRAINING PIPELINE")
    print(" Approach: Approach 2 (PDK Conditioning - 14 Spatial Channels + 11 Graph Node Feats)")
    print(f" Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Build DataLoaders with PDK Conditioning
    print("\n[1/4] Initializing Multi-PDK DataLoaders...")
    train_loader, val_loader, test_loader, split_info = create_multimodal_dataloaders(
        dataset_dir=dataset_dir,
        include_scan=True,
        pdk_conditioning=True,
        batch_size=args.batch_size,
        shuffle_train=True,
        preload=True,
    )
    print(f"  * Train samples: {split_info['train_samples']} (Designs: {split_info['train_designs']})")
    print(f"  * Val samples  : {split_info['val_samples']} (Designs: {split_info['val_designs']})")
    print(f"  * Test samples : {split_info['test_samples']} (Hold-out: {split_info['test_designs']})")
    print(f"  * Spatial in channels: {split_info['spatial_in_channels']}")
    print(f"  * Graph node features: {split_info['node_in_channels']}")

    # 2. Initialize Model
    print("\n[2/4] Initializing PDK-Conditioned CNNGNNFusionModel...")
    model = CNNGNNFusionModel(
        spatial_in_channels=split_info["spatial_in_channels"],  # 14
        base_channels=16,
        node_in_channels=split_info["node_in_channels"],        # 11
        gnn_hidden_channels=64,
        gnn_out_channels=64,
        num_gnn_layers=3,
        grid_size=64,
        fusion_mode="concat",
        out_channels=1,
        final_activation="none",
        default_mode="multimodal",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  * Trainable Parameters: {total_params:,} (~{total_params*4/(1024*1024):.1f} MB)")

    # 3. Train
    print("\n[3/4] Training Model across 45nm + 14nm...")
    t0 = time.time()
    results = train_fusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        loss_type=args.loss_type,
        lambda_corr=args.lambda_corr,
    )
    print(f"Training completed in {time.time() - t0:.1f}s.")

    # 4. Load best checkpoint and evaluate per PDK
    print("\n[4/4] Evaluating Best Checkpoint on Cross-Technology Hold-out Cores...")
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.is_file():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    pdk_metrics = evaluate_per_pdk(model, test_loader, device)

    print("\n" + "=" * 80)
    print(" CROSS-TECHNOLOGY EVALUATION RESULTS (OUT-OF-SAMPLE TEST CORES)")
    print("=" * 80)
    for pdk, data in pdk_metrics.items():
        print(f"\n--- Technology Node: {pdk.upper()} ({data['count']} hold-out samples) ---")
        print(f"  * Pearson r     : {data['avg_pearson']:+.4f}")
        print(f"  * Spearman rho  : {data['avg_spearman']:+.4f}")
        print(f"  * MAE           : {data['avg_mae']:.4f}")
        print(f"  * RMSE          : {data['avg_rmse']:.4f}")
        print(f"  * Top-5% Recall : {data['avg_recall_top5']*100:.2f}%")
        print(f"  * Top-5% IoU    : {data['avg_iou_top5']:.4f}")
        print(f"  * Top-10% Recall: {data['avg_recall_top10']*100:.2f}%")
        for s in data["samples"]:
            print(f"    - [{s['sample_id']}]: Spearman={s['spearman']:+.3f}, Pearson={s['pearson']:+.3f}, Top-5% Recall={s['recall_top5']*100:.1f}%")

    # Save summary report
    report_path = output_dir / "cross_technology_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pdk_metrics, f, indent=2)
    print(f"\nSaved cross-technology report to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
