"""
Training Engine for Multimodal CNN + GNN Hotspot Prediction (Stage 7b).

Provides:
- train_fusion_epoch: Single training epoch over a multimodal DataLoader.
- evaluate_fusion_model: Comprehensive evaluation computing loss, continuous regression metrics
  (MAE, RMSE, Pearson r, Spearman rho), and discrete hotspot metrics (Top-5% Recall, IoU@5%, Dice@5%).
- train_fusion: End-to-end training loop with validation-based early stopping,
  learning rate scheduling (ReduceLROnPlateau), model checkpointing, and metric logging.
"""

from __future__ import annotations

import copy
import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # type: ignore

from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json
from dft_hotspot.ml.train_unet import compute_multichannel_metrics, set_seed
from dft_hotspot.models.fusion import CNNGNNFusionModel

logger = logging.getLogger(__name__)


class CorrelationAwareLoss(nn.Module):
    """
    Correlation-Aware Loss combining Mean Squared Error and Pearson Linear Correlation.
    L = MSE(pred, target) + lambda_corr * (1.0 - Pearson(pred, target))
    
    Penalizes both absolute calibration errors and ranking/contrast misalignments,
    specifically driving higher recall on sparse routing hotspots.
    """
    def __init__(self, lambda_corr: float = 0.15, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_corr = lambda_corr
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        
        # Flatten spatial dimensions per sample in batch: (B, -1)
        B = pred.size(0)
        p_flat = pred.view(B, -1)
        t_flat = target.view(B, -1)
        
        # Mean center
        p_cen = p_flat - p_flat.mean(dim=1, keepdim=True)
        t_cen = t_flat - t_flat.mean(dim=1, keepdim=True)
        
        # Pearson correlation per sample
        cov = (p_cen * t_cen).sum(dim=1)
        var_p = (p_cen ** 2).sum(dim=1)
        var_t = (t_cen ** 2).sum(dim=1)
        
        denom = torch.sqrt(var_p * var_t + self.eps)
        pearson = cov / denom
        
        # Average correlation loss across batch (1.0 - Pearson in [0, 2])
        corr_loss = 1.0 - pearson.mean()
        
        return mse_loss + self.lambda_corr * corr_loss


def train_fusion_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mode: Optional[str] = None,
) -> float:
    """
    Executes one training epoch over a multimodal DataLoader.

    Args:
        model: CNNGNNFusionModel instance.
        dataloader: Multimodal DataLoader delivering paired batches.
        optimizer: PyTorch optimizer.
        criterion: Loss function (e.g. MSELoss).
        device: Target compute device (CPU or CUDA).
        mode: Optional operational mode ('multimodal', 'spatial_only', 'graph_only').

    Returns:
        Average training loss across all samples in the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x_spatial = batch["x_spatial"].to(device) if batch["x_spatial"] is not None else None
        graph_data = batch["graph"].to(device) if batch["graph"] is not None else None
        y = batch["y"].to(device)
        batch_size = y.size(0)

        optimizer.zero_grad()
        outputs = model(x_spatial=x_spatial, graph_data=graph_data, mode=mode)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return running_loss / total_samples if total_samples > 0 else 0.0


def evaluate_fusion_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: Optional[str] = None,
    clamp_output: bool = True,
    channel_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Evaluates multimodal fusion model on a DataLoader supporting single- and multi-target modes.

    Returns:
        Tuple of:
        - average loss (float)
        - aggregate spatial metrics dict (macro metrics plus 'per_channel' and 'empty_channels')
        - dictionary of per-sample prediction arrays {sample_id: (C, H, W) or (H, W)}
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds_list: List[np.ndarray] = []
    all_targets_list: List[np.ndarray] = []
    predictions_by_sample: Dict[str, np.ndarray] = {}

    if channel_names is None:
        channel_names = getattr(dataloader.dataset, "target_channels", None)

    with torch.no_grad():
        for batch in dataloader:
            x_spatial = batch["x_spatial"].to(device) if batch["x_spatial"] is not None else None
            graph_data = batch["graph"].to(device) if batch["graph"] is not None else None
            y = batch["y"].to(device)
            batch_size = y.size(0)

            outputs = model(x_spatial=x_spatial, graph_data=graph_data, mode=mode)
            loss = criterion(outputs, y)
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds_np = outputs.cpu().numpy()
            if clamp_output:
                preds_np = np.clip(preds_np, 0.0, 1.0)
            targets_np = y.cpu().numpy()

            sample_ids = batch.get("sample_ids", None)
            if sample_ids is None or len(sample_ids) != batch_size:
                sample_ids = [f"sample_{i}" for i in range(batch_size)]

            for i, sid in enumerate(sample_ids):
                p_sample = preds_np[i].astype(np.float32)
                t_sample = targets_np[i].astype(np.float32)

                if p_sample.shape[0] == 1:
                    predictions_by_sample[sid] = p_sample[0]
                else:
                    predictions_by_sample[sid] = p_sample

                all_preds_list.append(p_sample)
                all_targets_list.append(t_sample)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    if not all_preds_list:
        return avg_loss, {}, predictions_by_sample

    preds_all = np.stack(all_preds_list, axis=0)      # (N_total, C, H, W)
    targets_all = np.stack(all_targets_list, axis=0)  # (N_total, C, H, W)

    num_channels = preds_all.shape[1]
    if num_channels == 1:
        metrics = compute_all_metrics(preds_all.reshape(-1), targets_all.reshape(-1))
    else:
        macro_m, per_ch_m, empty_ch = compute_multichannel_metrics(
            preds_all, targets_all, channel_names=channel_names
        )
        metrics = dict(macro_m)
        metrics["per_channel"] = per_ch_m
        metrics["empty_channels"] = empty_ch

    # Populate convenience alias for Top-5% recall
    if "recall_5pct" in metrics and "top5_recall" not in metrics:
        metrics["top5_recall"] = metrics["recall_5pct"]
    elif "top5_recall" in metrics and "recall_5pct" not in metrics:
        metrics["recall_5pct"] = metrics["top5_recall"]

    return avg_loss, metrics, predictions_by_sample


def train_fusion(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    mode: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    early_stopping_patience: int = 15,
    lr_patience: int = 7,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    seed: int = 42,
    channel_names: Optional[List[str]] = None,
    loss_type: str = "mse",
    lambda_corr: float = 0.15,
) -> Dict[str, Any]:
    """
    Complete training orchestration loop for the Multimodal CNN + GNN Fusion framework.

    Args:
        model: CNNGNNFusionModel instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Optional held-out test DataLoader.
        num_epochs: Maximum epochs.
        lr: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        device: Target compute device. Defaults to CUDA if available, else CPU.
        mode: Operational mode ('multimodal', 'spatial_only', 'graph_only').
        output_dir: Optional directory for saving checkpoints, logs, and predictions.
        early_stopping_patience: Epochs to wait without val loss improvement before stopping.
        lr_patience: Epochs to wait without val loss improvement before decaying learning rate.
        lr_factor: Multiplicative factor for learning rate decay.
        min_lr: Minimum learning rate floor.
        seed: Random seed for reproducibility.
        channel_names: Target channel names.
        loss_type: Loss formulation ('mse' or 'correlation_aware').
        lambda_corr: Weight for Pearson correlation loss component.

    Returns:
        Structured dictionary containing training history, best model state, and evaluation metrics.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for train_fusion.")

    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    if loss_type == "correlation_aware":
        criterion = CorrelationAwareLoss(lambda_corr=lambda_corr)
        logger.info("Using Correlation-Aware Loss (MSE + %.3f * (1.0 - Pearson))", lambda_corr)
    else:
        criterion = nn.MSELoss()
        logger.info("Using standard MSE Loss")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    if channel_names is None:
        channel_names = getattr(train_loader.dataset, "target_channels", None)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0

    start_time = time.time()
    logger.info("Commencing multimodal fusion training: %d epochs on device %s", num_epochs, device)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_fusion_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            mode=mode,
        )

        val_loss, val_metrics, _ = evaluate_fusion_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            mode=mode,
            channel_names=channel_names,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        epoch_elapsed = time.time() - epoch_start
        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(current_lr),
            "val_mae": float(val_metrics.get("mae", 0.0)),
            "val_rmse": float(val_metrics.get("rmse", 0.0)),
            "val_pearson": float(val_metrics.get("pearson", 0.0)),
            "val_spearman": float(val_metrics.get("spearman", 0.0)),
            "val_top5_recall": float(val_metrics.get("top5_recall", 0.0)),
            "val_iou_5pct": float(val_metrics.get("iou_5pct", 0.0)),
            "val_dice_5pct": float(val_metrics.get("dice_5pct", 0.0)),
            "epoch_time_s": float(epoch_elapsed),
        }
        history.append(epoch_record)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_metrics = copy.deepcopy(val_metrics)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or is_best or epoch == num_epochs:
            logger.info(
                "Epoch %03d/%03d | Train: %.6f | Val: %.6f | Best: %.6f (ep %03d) | r: %.4f | IoU@5%%: %.4f | lr: %.1e",
                epoch,
                num_epochs,
                train_loss,
                val_loss,
                best_val_loss,
                best_epoch,
                val_metrics.get("pearson", float("nan")),
                val_metrics.get("iou_5pct", float("nan")),
                current_lr,
            )

        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d (patience %d)", epoch, early_stopping_patience)
            break

    total_training_time = time.time() - start_time

    # Load best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on held-out test set
    test_loss = 0.0
    test_metrics: Dict[str, Any] = {}
    test_preds: Dict[str, np.ndarray] = {}
    if test_loader is not None:
        test_loss, test_metrics, test_preds = evaluate_fusion_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            mode=mode,
            channel_names=channel_names,
        )

    # Save artifacts if output_dir specified
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if best_model_state is not None:
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_model_state,
                    "val_loss": best_val_loss,
                    "val_metrics": sanitize_metrics_for_json(best_val_metrics),
                    "test_metrics": sanitize_metrics_for_json(test_metrics),
                },
                out_path / "best_model.pt",
            )
        torch.save(
            {
                "epoch": len(history),
                "model_state_dict": model.state_dict(),
            },
            out_path / "last_model.pt",
        )

        # Export training history CSV
        hist_csv = out_path / "training_history.csv"
        if history:
            fieldnames = list(history[0].keys())
            with open(hist_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(history)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "total_training_time": float(total_training_time),
        "total_epochs": len(history),
        "training_history": history,
        "best_val_metrics": best_val_metrics,
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "test_predictions": test_preds,
    }
