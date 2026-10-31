"""
Training Engine for GraphSAGE Dense 2D Hotspot Prediction Baseline (Stage 6e).

Provides:
- train_gnn_epoch: Single training epoch over a PyTorch Geometric DataLoader.
- evaluate_gnn_model: Evaluation pass computing loss and spatial hotspot metrics.
- train_gnn: Complete training orchestration loop with early stopping,
  learning rate scheduling, model checkpointing, and metric logging.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader as PyGDataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # type: ignore

from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json
from dft_hotspot.ml.train_unet import compute_multichannel_metrics, set_seed
from dft_hotspot.models.gnn import GNNHotspotPredictor

logger = logging.getLogger(__name__)


def train_gnn_epoch(
    model: nn.Module,
    dataloader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Executes one training epoch over a PyG DataLoader.

    Returns:
        Average training loss across all samples in the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device)
        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else batch.y.size(0)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return running_loss / total_samples if total_samples > 0 else 0.0


def evaluate_gnn_model(
    model: nn.Module,
    dataloader: PyGDataLoader,
    criterion: nn.Module,
    device: torch.device,
    clamp_output: bool = True,
    channel_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Evaluates GNN model on a PyG DataLoader supporting single-target and multi-target modes.

    Returns:
        Tuple of:
        - average loss
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
            batch = batch.to(device)
            batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else batch.y.size(0)

            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds_np = outputs.cpu().numpy()
            if clamp_output:
                preds_np = np.clip(preds_np, 0.0, 1.0)
            targets_np = batch.y.cpu().numpy()

            sample_ids = getattr(batch, "sample_id", None)
            if sample_ids is None or len(sample_ids) != batch_size:
                sample_ids = [f"sample_{i}" for i in range(batch_size)]

            for i, sid in enumerate(sample_ids):
                p_sample = preds_np[i].astype(np.float32)
                t_sample = targets_np[i].astype(np.float32)

                all_preds_list.append(p_sample)
                all_targets_list.append(t_sample)

                # If single-channel, squeeze to (H, W) for backward compatibility
                if p_sample.shape[0] == 1:
                    predictions_by_sample[sid] = p_sample[0]
                else:
                    predictions_by_sample[sid] = p_sample

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    if all_preds_list and all_targets_list:
        concat_preds = np.stack(all_preds_list, axis=0)
        concat_targets = np.stack(all_targets_list, axis=0)

        macro_m, per_ch_m, empty_ch = compute_multichannel_metrics(
            concat_preds, concat_targets, channel_names=channel_names
        )
        metrics: Dict[str, Any] = dict(macro_m)
        metrics["per_channel"] = per_ch_m
        metrics["empty_channels"] = empty_ch
    else:
        metrics = {}

    return avg_loss, metrics, predictions_by_sample


def train_gnn(
    model: GNNHotspotPredictor,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    output_dir: Path,
    device: Optional[torch.device] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    early_stopping_patience: int = 15,
    min_delta: float = 1e-5,
    seed: int = 42,
    loss_name: str = "mse",
    clamp_output: bool = True,
) -> Dict[str, Any]:
    """
    Trains GNNHotspotPredictor with validation early stopping, checkpointing, and logging.

    Args:
        model: GNNHotspotPredictor model instance.
        train_loader: Training PyG DataLoader.
        val_loader: Validation PyG DataLoader.
        output_dir: Directory to save checkpoints and training logs.
        device: Torch device (defaults to cuda:0 if available, else cpu).
        learning_rate: Initial learning rate for Adam.
        weight_decay: L2 regularization factor.
        max_epochs: Maximum epochs.
        early_stopping_patience: Epochs without improvement before stopping.
        min_delta: Minimum loss decrease to qualify as an improvement.
        seed: Random seed.
        loss_name: Loss criterion name ('mse', 'mae'/'l1', 'huber'/'smooth_l1').
        clamp_output: Whether to clamp predictions to [0, 1] during evaluation.

    Returns:
        Summary dict containing training history, best epoch, and best metrics.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for train_gnn.")

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    loss_key = loss_name.lower()
    if loss_key in ("mae", "l1"):
        criterion = nn.L1Loss()
    elif loss_key in ("huber", "smooth_l1", "smoothl1"):
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0

    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last_model.pt"

    channel_names = getattr(train_loader.dataset, "target_channels", None)

    logger.info("Starting GNN training on device '%s' for up to %d epochs...", device, max_epochs)
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_gnn_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _ = evaluate_gnn_model(
            model, val_loader, criterion, device, clamp_output=clamp_output, channel_names=channel_names
        )
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "epoch_time_s": round(elapsed, 3),
            "val_mae": val_metrics.get("mae"),
            "val_rmse": val_metrics.get("rmse"),
            "val_pearson": val_metrics.get("pearson"),
            "val_iou_5pct": val_metrics.get("iou_5pct"),
            "val_dice_5pct": val_metrics.get("dice_5pct"),
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %02d/%02d [%.2fs] - Train Loss: %.6f | Val Loss: %.6f | Val Pearson: %.4f | Val IoU@5%%: %.4f | LR: %.2e",
            epoch,
            max_epochs,
            elapsed,
            train_loss,
            val_loss,
            val_metrics.get("pearson", float("nan")),
            val_metrics.get("iou_5pct", float("nan")),
            current_lr,
        )

        # Checkpoint on validation improvement
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": sanitize_metrics_for_json(val_metrics),
                },
                best_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered at epoch %d (best val loss: %.6f at epoch %d).",
                    epoch,
                    best_val_loss,
                    best_epoch,
                )
                break

    total_time = round(time.time() - start_time, 2)

    # Save last model checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        },
        last_model_path,
    )

    # Export history to CSV
    csv_path = output_dir / "training_history.csv"
    if history:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # Reload best model weights for final evaluation
    if best_model_path.is_file():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Restored best model checkpoint from epoch %d.", best_epoch)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": sanitize_metrics_for_json(best_val_metrics),
        "total_epochs": len(history),
        "total_time_s": total_time,
        "device": str(device),
        "loss_function": loss_name,
    }
    return summary
