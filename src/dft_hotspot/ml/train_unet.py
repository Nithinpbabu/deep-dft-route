"""
Training Engine for TinyUNet Dense 2D Hotspot Prediction Baseline.
Stage 5b of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- train_epoch: single training epoch over a PyTorch DataLoader.
- evaluate: evaluation pass computing loss and spatial hotspot metrics.
- train_unet: complete training orchestration loop with early stopping,
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
    from torch.utils.data import DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # type: ignore

from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json
from dft_hotspot.ml.unet import TinyUNet

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Executes one training epoch.

    Returns:
        Average training loss across all batches.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return running_loss / total_samples if total_samples > 0 else 0.0


def compute_multichannel_metrics(
    concat_preds: np.ndarray,
    concat_targets: np.ndarray,
    channel_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[str]]:
    """
    Computes per-channel metrics and macro-average across channels.

    Args:
        concat_preds: (N, C, H, W) or (N, H, W) numpy array.
        concat_targets: (N, C, H, W) or (N, H, W) numpy array.
        channel_names: Optional list of channel names.

    Returns:
        Tuple of:
        - macro_metrics: Dict[str, float] (averaged across active non-empty channels)
        - per_channel_metrics: Dict[str, Dict[str, float]]
        - empty_channels: List[str] (channels where target variance is 0)
    """
    if concat_preds.ndim == 3:
        concat_preds = concat_preds[:, np.newaxis, :, :]
        concat_targets = concat_targets[:, np.newaxis, :, :]

    N, C, H, W = concat_preds.shape
    if channel_names is None or len(channel_names) != C:
        channel_names = [f"channel_{c}" for c in range(C)]

    per_channel: Dict[str, Dict[str, float]] = {}
    empty_channels: List[str] = []

    for c in range(C):
        ch_name = channel_names[c]
        p_c = concat_preds[:, c, :, :]
        t_c = concat_targets[:, c, :, :]

        ptp_val = float(np.ptp(t_c.ravel()))
        if ptp_val < 1e-12:
            empty_channels.append(ch_name)

        per_channel[ch_name] = compute_all_metrics(p_c, t_c)

    if C == 1:
        macro_metrics = dict(per_channel[channel_names[0]])
    else:
        macro_metrics = {}
        all_keys = list(per_channel[channel_names[0]].keys())
        active_channels = [ch for ch in channel_names if ch not in empty_channels]
        if not active_channels:
            active_channels = channel_names

        for k in all_keys:
            vals = [
                per_channel[ch][k]
                for ch in active_channels
                if k in per_channel[ch] and per_channel[ch][k] is not None and not np.isnan(per_channel[ch][k])
            ]
            macro_metrics[k] = float(np.mean(vals)) if vals else float("nan")

    return macro_metrics, per_channel, empty_channels


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    clamp_output: bool = False,
    channel_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Evaluates model on a dataloader supporting single-target and multi-target modes.

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
        channel_names = getattr(dataloader.dataset, "target_channel_names", None)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"]
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            # Move to CPU numpy for spatial metrics
            preds_np = outputs.cpu().numpy()
            if clamp_output:
                preds_np = np.clip(preds_np, 0.0, 1.0)
            targets_np = targets.cpu().numpy()

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

    # Compute aggregate spatial metrics across concatenated evaluation elements
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


def train_unet(
    model: TinyUNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    device: Optional[torch.device] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    early_stopping_patience: int = 15,
    min_delta: float = 1e-5,
    seed: int = 42,
    loss_name: str = "mse",
) -> Dict[str, Any]:
    """
    Trains TinyUNet with validation-based early stopping, checkpointing, and logging.

    Args:
        model: TinyUNet model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        output_dir: Directory to save checkpoints and training logs.
        device: Torch device (defaults to cuda:0 if available, else cpu).
        learning_rate: Initial learning rate for Adam.
        weight_decay: L2 regularization factor.
        max_epochs: Maximum epochs.
        early_stopping_patience: Epochs without improvement before stopping.
        min_delta: Minimum loss decrease to qualify as an improvement.
        seed: Random seed.
        loss_name: Loss criterion name ('mse', 'mae'/'l1', 'huber'/'smooth_l1').

    Returns:
        Summary dict containing training history and best metrics.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for train_unet.")

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
    best_epoch = 0
    epochs_no_improve = 0

    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            clamp_output=(model.final_activation == "none"),
            channel_names=getattr(val_loader.dataset, "target_channel_names", None),
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start
        val_mae = val_metrics.get("mae", float("nan"))
        val_pearson = val_metrics.get("pearson", float("nan"))
        val_iou_5pct = val_metrics.get("iou_5pct", float("nan"))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_pearson": val_pearson,
            "val_iou_5pct": val_iou_5pct,
            "lr": current_lr,
            "duration_s": round(epoch_duration, 3),
        }
        history.append(epoch_record)

        # Early stopping & model saving
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "model_config": {
                        "in_channels": model.in_channels,
                        "out_channels": model.out_channels,
                        "base_channels": model.base_channels,
                        "bilinear": model.bilinear,
                        "final_activation": model.final_activation,
                    },
                },
                best_checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d (best epoch was %d)", epoch, best_epoch)
            break

    # Save last state
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        },
        last_checkpoint_path,
    )

    # Export training history CSV
    history_csv = output_dir / "training_history.csv"
    if history:
        with open(history_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # Load best checkpoint into model for downstream evaluation
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    total_training_time = time.time() - start_time

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(history),
        "total_training_time_s": round(total_training_time, 2),
        "device": str(device),
        "history": history,
    }
