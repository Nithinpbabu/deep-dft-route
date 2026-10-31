"""
Spatial Evaluation Metrics for Hotspot Prediction.
Stage 4c of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides dense 2D spatial evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Pearson correlation coefficient (r)
- Spearman rank correlation coefficient (rho)
- Top-k hotspot mask extraction
- Top-k recall
- Precision@k
- Intersection over Union (IoU) for top-k masks
- Dice coefficient for top-k masks

All routines handle edge cases (constant arrays, NaNs, zero division) safely
without raising unhandled exceptions.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import scipy.stats as stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _validate_arrays(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validates and converts input arrays to float32."""
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred shape {p.shape} != target shape {t.shape}")
    return p, t


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Mean Absolute Error (MAE) across all spatial elements:
        MAE = (1 / N) * sum(|pred - target|)
    """
    p, t = _validate_arrays(pred, target)
    return float(np.mean(np.abs(p - t)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Root Mean Squared Error (RMSE) across all spatial elements:
        RMSE = sqrt((1 / N) * sum((pred - target)^2))
    """
    p, t = _validate_arrays(pred, target)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Pearson correlation coefficient between flattened pred and target.
    Returns float('nan') safely if either array has zero standard deviation.
    """
    p, t = _validate_arrays(pred, target)
    p_flat = p.ravel()
    t_flat = t.ravel()

    p_ptp = float(np.ptp(p_flat))
    t_ptp = float(np.ptp(t_flat))
    p_std = float(np.std(p_flat))
    t_std = float(np.std(t_flat))

    if p_ptp < 1e-6 or t_ptp < 1e-6 or p_std < 1e-6 or t_std < 1e-6:
        warnings.warn(
            "Pearson correlation is undefined for constant array with zero variance; returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(p_flat, t_flat)[0, 1])
    return corr


def spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Spearman rank correlation coefficient between flattened pred and target.
    Returns float('nan') safely if either array has zero standard deviation or if scipy is absent.
    """
    p, t = _validate_arrays(pred, target)
    p_flat = p.ravel()
    t_flat = t.ravel()

    p_ptp = float(np.ptp(p_flat))
    t_ptp = float(np.ptp(t_flat))
    p_std = float(np.std(p_flat))
    t_std = float(np.std(t_flat))

    if p_ptp < 1e-6 or t_ptp < 1e-6 or p_std < 1e-6 or t_std < 1e-6:
        warnings.warn(
            "Spearman correlation is undefined for constant array with zero variance; returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    if not _HAS_SCIPY:
        warnings.warn("scipy not installed; Spearman correlation cannot be computed.", UserWarning, stacklevel=2)
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.spearmanr(p_flat, t_flat)
        corr = getattr(res, "statistic", None)
        if corr is None:
            corr = getattr(res, "correlation", float("nan"))
    return float(corr)



def topk_mask(array: np.ndarray, percent: float) -> np.ndarray:
    """
    Generates a boolean binary mask indicating the top-k highest value elements.

    Args:
        array: Input numpy array of any shape.
        percent: Hotspot percentile fraction in (0.0, 1.0] or (1.0, 100.0].
                 E.g. 0.05 or 5.0 both correspond to the top 5% elements.

    Returns:
        np.ndarray of bool matching input shape, where True indicates a top-k hotspot pixel.
    """
    arr = np.asarray(array, dtype=np.float32)
    if percent > 1.0:
        percent = percent / 100.0
    if not (0.0 < percent <= 1.0):
        raise ValueError(f"Percent must be in range (0, 1] or (1, 100], got {percent}")

    n_total = arr.size
    k = max(1, int(round(n_total * percent)))

    # If entire array has zero variance or is all zeros, no distinct hotspots exist
    if float(np.ptp(arr)) < 1e-12:
        return np.zeros(arr.shape, dtype=bool)

    # Deterministic top-k by descending value sort
    order = np.argsort(arr.ravel())[::-1]
    mask_flat = np.zeros(n_total, dtype=bool)
    mask_flat[order[:k]] = True
    return mask_flat.reshape(arr.shape)


def topk_recall(pred: np.ndarray, target: np.ndarray, percent: float) -> float:
    """
    Computes recall for top-k hotspots:
        Recall@k = |M_pred & M_target| / |M_target|
    Where M_pred and M_target are binary masks of the top percent elements.
    """
    p, t = _validate_arrays(pred, target)
    m_pred = topk_mask(p, percent)
    m_target = topk_mask(t, percent)

    n_target = int(np.count_nonzero(m_target))
    if n_target == 0:
        return 1.0 if np.count_nonzero(m_pred) == 0 else 0.0

    intersection = int(np.count_nonzero(m_pred & m_target))
    return float(intersection / n_target)


def precision_at_k(pred: np.ndarray, target: np.ndarray, percent: float) -> float:
    """
    Computes precision for top-k hotspots:
        Precision@k = |M_pred & M_target| / |M_pred|
    Where M_pred and M_target are binary masks of the top percent elements.
    """
    p, t = _validate_arrays(pred, target)
    m_pred = topk_mask(p, percent)
    m_target = topk_mask(t, percent)

    n_pred = int(np.count_nonzero(m_pred))
    if n_pred == 0:
        return 1.0 if np.count_nonzero(m_target) == 0 else 0.0

    intersection = int(np.count_nonzero(m_pred & m_target))
    return float(intersection / n_pred)


def iou_topk(pred: np.ndarray, target: np.ndarray, percent: float) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index) for top-k hotspot masks:
        IoU = |M_pred & M_target| / |M_pred | M_target|
    """
    p, t = _validate_arrays(pred, target)
    m_pred = topk_mask(p, percent)
    m_target = topk_mask(t, percent)

    union = int(np.count_nonzero(m_pred | m_target))
    if union == 0:
        return 1.0

    intersection = int(np.count_nonzero(m_pred & m_target))
    return float(intersection / union)


def dice_topk(pred: np.ndarray, target: np.ndarray, percent: float) -> float:
    """
    Computes Dice similarity coefficient (F1 score) for top-k hotspot masks:
        Dice = 2 * |M_pred & M_target| / (|M_pred| + |M_target|)
    """
    p, t = _validate_arrays(pred, target)
    m_pred = topk_mask(p, percent)
    m_target = topk_mask(t, percent)

    total = int(np.count_nonzero(m_pred)) + int(np.count_nonzero(m_target))
    if total == 0:
        return 1.0

    intersection = int(np.count_nonzero(m_pred & m_target))
    return float((2.0 * intersection) / total)


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    k_percent_list: Optional[Tuple[float, ...]] = (0.05, 0.10),
) -> Dict[str, float]:
    """
    Computes all standard continuous and thresholded spatial evaluation metrics.

    Returns:
        Dict mapping metric names to float values:
        - 'mae'
        - 'rmse'
        - 'pearson'
        - 'spearman'
        - 'recall_5pct', 'precision_5pct', 'iou_5pct', 'dice_5pct'
        - 'recall_10pct', 'precision_10pct', 'iou_10pct', 'dice_10pct'
    """
    metrics: Dict[str, float] = {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "pearson": pearson_corr(pred, target),
        "spearman": spearman_corr(pred, target),
    }

    if k_percent_list:
        for pct in k_percent_list:
            pct_label = int(round(pct * 100)) if pct <= 1.0 else int(round(pct))
            metrics[f"recall_{pct_label}pct"] = topk_recall(pred, target, pct)
            metrics[f"precision_{pct_label}pct"] = precision_at_k(pred, target, pct)
            metrics[f"iou_{pct_label}pct"] = iou_topk(pred, target, pct)
            metrics[f"dice_{pct_label}pct"] = dice_topk(pred, target, pct)

    return metrics


def sanitize_metrics_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts NaN, Inf, and numpy float types to Python float or None for standard JSON compliance.
    """
    sanitized: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            sanitized[k] = sanitize_metrics_for_json(v)
        elif isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                sanitized[k] = None
            else:
                sanitized[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            sanitized[k] = int(v)
        else:
            sanitized[k] = v
    return sanitized
