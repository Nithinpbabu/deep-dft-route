"""
Post-hoc Calibration Routines for Dense 2D Hotspot Predictions.
Stage 5c of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- IdentityCalibrator: Passthrough baseline.
- GlobalAffineCalibrator: Linear affine shift/scale (y = a * x + b) clamped to [0, 1].
- GlobalScaleCalibrator: Multiplicative scaling (y = s * x) clamped to [0, 1].
- IsotonicCalibrator: Non-parametric monotonic transformation via IsotonicRegression.
- Calibration suite fitting on validation partitions and evaluating on test splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json


class BaseCalibrator:
    """Abstract base class for 2D spatial prediction calibrators."""

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> BaseCalibrator:
        raise NotImplementedError

    def predict(self, preds: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        raise NotImplementedError


class IdentityCalibrator(BaseCalibrator):
    """Uncalibrated passthrough."""

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> IdentityCalibrator:
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        return np.asarray(preds, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"method": "identity"}


class GlobalAffineCalibrator(BaseCalibrator):
    """
    Fits a linear affine transformation y = a * x + b using ordinary least squares.
    Clamps calibrated outputs to [0.0, 1.0].
    """

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> GlobalAffineCalibrator:
        p_flat = np.asarray(preds, dtype=np.float64).ravel()
        t_flat = np.asarray(targets, dtype=np.float64).ravel()

        if len(p_flat) == 0:
            raise ValueError("Cannot fit calibrator on empty array.")

        p_mean = float(np.mean(p_flat))
        t_mean = float(np.mean(t_flat))
        denom = float(np.sum((p_flat - p_mean) ** 2))

        if denom < 1e-12:
            # Constant prediction edge case: predict mean target directly
            self.a = 0.0
            self.b = t_mean
        else:
            self.a = float(np.sum((p_flat - p_mean) * (t_flat - t_mean)) / denom)
            self.b = float(t_mean - self.a * p_mean)

        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        p = np.asarray(preds, dtype=np.float32)
        calibrated = self.a * p + self.b
        return np.clip(calibrated, self.clip_min, self.clip_max).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "method": "global_affine",
            "slope_a": float(self.a),
            "intercept_b": float(self.b),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


class GlobalScaleCalibrator(BaseCalibrator):
    """
    Fits a multiplicative scaling factor y = s * x through the origin.
    s = sum(x * y) / sum(x^2).
    Clamps calibrated outputs to [0.0, 1.0].
    """

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        self.s: float = 1.0
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> GlobalScaleCalibrator:
        p_flat = np.asarray(preds, dtype=np.float64).ravel()
        t_flat = np.asarray(targets, dtype=np.float64).ravel()

        denom = float(np.sum(p_flat ** 2))
        if denom < 1e-12:
            self.s = 1.0
        else:
            self.s = float(np.sum(p_flat * t_flat) / denom)

        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        p = np.asarray(preds, dtype=np.float32)
        calibrated = self.s * p
        return np.clip(calibrated, self.clip_min, self.clip_max).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "method": "global_scale",
            "scale_s": float(self.s),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


class IsotonicCalibrator(BaseCalibrator):
    """
    Non-parametric monotonic piecewise-constant transformation via scikit-learn.
    Preserves rank order perfectly while fitting empirical distribution.
    """

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for IsotonicCalibrator.")
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=clip_min, y_max=clip_max)
        self.is_constant_ = False
        self.constant_val_ = 0.0

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> IsotonicCalibrator:
        p_flat = np.asarray(preds, dtype=np.float64).ravel()
        t_flat = np.asarray(targets, dtype=np.float64).ravel()

        if float(np.ptp(p_flat)) < 1e-12:
            self.is_constant_ = True
            self.constant_val_ = float(np.mean(t_flat))
        else:
            self.is_constant_ = False
            self.iso.fit(p_flat, t_flat)
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        p = np.asarray(preds, dtype=np.float32)
        if self.is_constant_:
            cal = np.full(p.shape, self.constant_val_, dtype=np.float32)
        else:
            orig_shape = p.shape
            cal = self.iso.predict(p.ravel().astype(np.float64)).reshape(orig_shape)
        return np.clip(cal, self.clip_min, self.clip_max).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "method": "isotonic_regression",
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


def fit_calibrators(
    val_preds: np.ndarray,
    val_targets: np.ndarray,
) -> Dict[str, BaseCalibrator]:
    """
    Fits all standard calibrators strictly on validation split data.

    Returns:
        Dict mapping calibrator name -> fitted calibrator instance.
    """
    calibrators: Dict[str, BaseCalibrator] = {
        "identity": IdentityCalibrator().fit(val_preds, val_targets),
        "global_affine": GlobalAffineCalibrator().fit(val_preds, val_targets),
        "global_scale": GlobalScaleCalibrator().fit(val_preds, val_targets),
    }

    if _HAS_SKLEARN:
        calibrators["isotonic"] = IsotonicCalibrator().fit(val_preds, val_targets)

    return calibrators
