"""
Classical Tabular Machine Learning Baselines for Hotspot Prediction.
Stage 4d of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- MeanBaseline: simple empirical target mean predictor.
- ClassicalModelResult: container for trained models, coefficients, and metrics.
- Model training suite:
  1. mean_baseline
  2. linear_regression (LinearRegression)
  3. ridge_regression (Ridge with alpha sweep [0.1, 1.0, 10.0])
  4. lasso_regression (Lasso with alpha sweep [0.0001, 0.001, 0.01])
  5. random_forest_regressor (RandomForestRegressor, max_depth=8)
  6. gradient_boosting_regressor (HistGradientBoostingRegressor, max_depth=6)
  7. xgboost_regressor (gracefully skipped if optional dependency not installed)
- Sample-level and split-level spatial metric evaluation.
- Exporters for coefficients, feature importances, and metric tables.
"""

from __future__ import annotations

import csv
import datetime
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json
from dft_hotspot.ml.tabular_dataset import TabularDataset

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


class MeanBaselineRegressor:
    """Predicts the empirical mean of the training target labels for all samples."""

    def __init__(self) -> None:
        self.mean_value_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanBaselineRegressor:
        self.mean_value_ = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0],), self.mean_value_, dtype=np.float32)


@dataclass
class ClassicalModelResult:
    """Holds trained model artifact, parameters, metrics, and spatial predictions."""

    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    model: Any
    coefficients: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None
    feature_importances: Optional[Dict[str, float]] = None
    metrics_per_sample: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metrics_by_split: Dict[str, Dict[str, Dict[str, Optional[float]]]] = field(default_factory=dict)
    predictions_2d: Dict[str, np.ndarray] = field(default_factory=dict)


def train_and_select_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Tuple[float, ...] = (0.1, 1.0, 10.0),
    random_state: int = 42,
) -> Tuple[Ridge, float]:
    """Selects best alpha for Ridge regression based on validation MAE."""
    best_alpha = alphas[0]
    best_mae = float("inf")
    best_model = None

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)
        if len(y_val) > 0:
            pred_val = model.predict(X_val)
            val_mae = float(np.mean(np.abs(pred_val - y_val)))
            if val_mae < best_mae:
                best_mae = val_mae
                best_alpha = alpha
                best_model = model
        else:
            best_model = model
            best_alpha = alpha
            break

    assert best_model is not None
    return best_model, best_alpha


def train_and_select_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Tuple[float, ...] = (0.0001, 0.001, 0.01),
    random_state: int = 42,
) -> Tuple[Lasso, float]:
    """Selects best alpha for Lasso regression based on validation MAE."""
    best_alpha = alphas[0]
    best_mae = float("inf")
    best_model = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alpha in alphas:
            model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
            model.fit(X_train, y_train)
            if len(y_val) > 0:
                pred_val = model.predict(X_val)
                val_mae = float(np.mean(np.abs(pred_val - y_val)))
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_alpha = alpha
                    best_model = model
            else:
                best_model = model
                best_alpha = alpha
                break

    assert best_model is not None
    return best_model, best_alpha


def evaluate_model_on_dataset(
    model: Any,
    tabular_data: TabularDataset,
    clip_output: bool = True,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Dict[str, Optional[float]]]],
    Dict[str, np.ndarray],
]:
    """
    Evaluates a trained model on all samples in tabular_data and aggregates metrics by split.

    Returns:
        (metrics_per_sample, metrics_by_split, predictions_2d)
    """
    H, W = tabular_data.grid_shape
    per_sample_metrics: Dict[str, Dict[str, float]] = {}
    predictions_2d: Dict[str, np.ndarray] = {}

    split_groups: Dict[str, List[str]] = {
        "all": list(tabular_data.samples.keys()),
        "train": tabular_data.train_sample_ids,
        "val": tabular_data.val_sample_ids,
        "test": tabular_data.test_sample_ids,
    }

    # Generate predictions and evaluate each sample
    for sid, s_tab in tabular_data.samples.items():
        pred_1d = model.predict(s_tab.X).astype(np.float32)
        if clip_output:
            pred_1d = np.clip(pred_1d, 0.0, 1.0)
        pred_2d = pred_1d.reshape(H, W)
        target_2d = s_tab.y.reshape(H, W)

        predictions_2d[sid] = pred_2d
        per_sample_metrics[sid] = compute_all_metrics(pred_2d, target_2d)

    # Aggregate metrics across splits
    by_split: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    first_metrics = next(iter(per_sample_metrics.values())) if per_sample_metrics else {}
    metric_keys = list(first_metrics.keys())

    for split_name, sids in split_groups.items():
        by_split[split_name] = {}
        valid_sids = [s for s in sids if s in per_sample_metrics]
        if not valid_sids:
            continue

        for m_key in metric_keys:
            vals = [
                per_sample_metrics[s][m_key]
                for s in valid_sids
                if per_sample_metrics[s].get(m_key) is not None and not np.isnan(per_sample_metrics[s][m_key])
            ]
            if vals:
                by_split[split_name][m_key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "valid_count": len(vals),
                }
            else:
                by_split[split_name][m_key] = {"mean": None, "std": None, "valid_count": 0}

    return per_sample_metrics, by_split, predictions_2d


def train_classical_baselines(
    tabular_data: TabularDataset,
    random_state: int = 42,
) -> Tuple[Dict[str, ClassicalModelResult], Dict[str, str]]:
    """
    Trains all standard classical ML baselines on tabular_data.X_train and evaluates on splits.

    Returns:
        (results_dict, skipped_dict)
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required to train classical ML baselines.")

    results: Dict[str, ClassicalModelResult] = {}
    skipped: Dict[str, str] = {}

    X_tr = tabular_data.X_train
    y_tr = tabular_data.y_train
    X_val = tabular_data.X_val
    y_val = tabular_data.y_val
    feat_names = tabular_data.feature_names

    # 1. Mean baseline
    mean_model = MeanBaselineRegressor().fit(X_tr, y_tr)
    m_sample, m_split, m_preds = evaluate_model_on_dataset(mean_model, tabular_data, clip_output=True)
    results["mean_baseline"] = ClassicalModelResult(
        model_name="mean_baseline",
        model_type="MeanBaselineRegressor",
        hyperparameters={"mean_value": mean_model.mean_value_},
        model=mean_model,
        intercept=mean_model.mean_value_,
        metrics_per_sample=m_sample,
        metrics_by_split=m_split,
        predictions_2d=m_preds,
    )

    # 2. Linear Regression
    lr_model = LinearRegression().fit(X_tr, y_tr)
    lr_sample, lr_split, lr_preds = evaluate_model_on_dataset(lr_model, tabular_data, clip_output=True)
    results["linear_regression"] = ClassicalModelResult(
        model_name="linear_regression",
        model_type="LinearRegression",
        hyperparameters={},
        model=lr_model,
        coefficients={fn: float(c) for fn, c in zip(feat_names, lr_model.coef_)},
        intercept=float(lr_model.intercept_),
        metrics_per_sample=lr_sample,
        metrics_by_split=lr_split,
        predictions_2d=lr_preds,
    )

    # 3. Ridge Regression
    ridge_model, best_ridge_alpha = train_and_select_ridge(
        X_tr, y_tr, X_val, y_val, alphas=(0.1, 1.0, 10.0), random_state=random_state
    )
    r_sample, r_split, r_preds = evaluate_model_on_dataset(ridge_model, tabular_data, clip_output=True)
    results["ridge_regression"] = ClassicalModelResult(
        model_name="ridge_regression",
        model_type="Ridge",
        hyperparameters={"alpha": best_ridge_alpha, "random_state": random_state},
        model=ridge_model,
        coefficients={fn: float(c) for fn, c in zip(feat_names, ridge_model.coef_)},
        intercept=float(ridge_model.intercept_),
        metrics_per_sample=r_sample,
        metrics_by_split=r_split,
        predictions_2d=r_preds,
    )

    # 4. Lasso Regression
    lasso_model, best_lasso_alpha = train_and_select_lasso(
        X_tr, y_tr, X_val, y_val, alphas=(0.0001, 0.001, 0.01), random_state=random_state
    )
    l_sample, l_split, l_preds = evaluate_model_on_dataset(lasso_model, tabular_data, clip_output=True)
    results["lasso_regression"] = ClassicalModelResult(
        model_name="lasso_regression",
        model_type="Lasso",
        hyperparameters={"alpha": best_lasso_alpha, "random_state": random_state, "max_iter": 2000},
        model=lasso_model,
        coefficients={fn: float(c) for fn, c in zip(feat_names, lasso_model.coef_)},
        intercept=float(lasso_model.intercept_),
        metrics_per_sample=l_sample,
        metrics_by_split=l_split,
        predictions_2d=l_preds,
    )

    # 5. Random Forest Regressor
    rf_params = {"n_estimators": 100, "max_depth": 8, "random_state": random_state, "n_jobs": -1}
    rf_model = RandomForestRegressor(**rf_params).fit(X_tr, y_tr)
    rf_sample, rf_split, rf_preds = evaluate_model_on_dataset(rf_model, tabular_data, clip_output=True)
    results["random_forest_regressor"] = ClassicalModelResult(
        model_name="random_forest_regressor",
        model_type="RandomForestRegressor",
        hyperparameters=rf_params,
        model=rf_model,
        feature_importances={fn: float(fi) for fn, fi in zip(feat_names, rf_model.feature_importances_)},
        metrics_per_sample=rf_sample,
        metrics_by_split=rf_split,
        predictions_2d=rf_preds,
    )

    # 6. Gradient Boosting Regressor (HistGradientBoostingRegressor)
    gb_params = {"max_iter": 100, "max_depth": 6, "random_state": random_state}
    gb_model = HistGradientBoostingRegressor(**gb_params).fit(X_tr, y_tr)
    gb_sample, gb_split, gb_preds = evaluate_model_on_dataset(gb_model, tabular_data, clip_output=True)
    results["gradient_boosting_regressor"] = ClassicalModelResult(
        model_name="gradient_boosting_regressor",
        model_type="HistGradientBoostingRegressor",
        hyperparameters=gb_params,
        model=gb_model,
        metrics_per_sample=gb_sample,
        metrics_by_split=gb_split,
        predictions_2d=gb_preds,
    )

    # 7. XGBoost Regressor (optional)
    if _HAS_XGBOOST:
        xgb_params = {"n_estimators": 100, "max_depth": 6, "random_state": random_state, "verbosity": 0}
        xgb_model = xgb.XGBRegressor(**xgb_params).fit(X_tr, y_tr)
        xgb_sample, xgb_split, xgb_preds = evaluate_model_on_dataset(xgb_model, tabular_data, clip_output=True)
        results["xgboost_regressor"] = ClassicalModelResult(
            model_name="xgboost_regressor",
            model_type="XGBRegressor",
            hyperparameters=xgb_params,
            model=xgb_model,
            feature_importances={fn: float(fi) for fn, fi in zip(feat_names, xgb_model.feature_importances_)},
            metrics_per_sample=xgb_sample,
            metrics_by_split=xgb_split,
            predictions_2d=xgb_preds,
        )
    else:
        skipped["xgboost_regressor"] = "xgboost package not installed (optional dependency skipped)"

    return results, skipped


def export_coefficients_csv(
    results: Dict[str, ClassicalModelResult],
    feature_names: List[str],
    output_path: Path,
) -> Path:
    """Exports linear/ridge/lasso regression coefficients to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    linear_models = [m for m in ["linear_regression", "ridge_regression", "lasso_regression"] if m in results]

    headers = ["feature_name"] + linear_models
    rows = []

    # Features
    for fn in feature_names:
        row = {"feature_name": fn}
        for lm in linear_models:
            val = results[lm].coefficients.get(fn) if results[lm].coefficients else None
            row[lm] = f"{val:.6f}" if val is not None else ""
        rows.append(row)

    # Intercept row
    intercept_row = {"feature_name": "[intercept]"}
    for lm in linear_models:
        val = results[lm].intercept
        intercept_row[lm] = f"{val:.6f}" if val is not None else ""
    rows.append(intercept_row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_feature_importance_csv(
    results: Dict[str, ClassicalModelResult],
    feature_names: List[str],
    output_path: Path,
) -> Path:
    """Exports tree-based model feature importances to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree_models = [m for m in ["random_forest_regressor", "xgboost_regressor"] if m in results and results[m].feature_importances]

    headers = ["feature_name"] + tree_models
    rows = []

    for fn in feature_names:
        row = {"feature_name": fn}
        for tm in tree_models:
            val = results[tm].feature_importances.get(fn) if results[tm].feature_importances else None
            row[tm] = f"{val:.6f}" if val is not None else ""
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_metrics_per_model_csv(
    results: Dict[str, ClassicalModelResult],
    eval_partition: str,
    output_path: Path,
) -> Path:
    """Exports model comparison table for the specified partition (e.g. 'val')."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["model", "mae", "rmse", "pearson", "spearman", "recall_5pct", "precision_5pct", "iou_5pct", "dice_5pct", "recall_10pct", "precision_10pct"]
    rows = []

    for m_name, res in sorted(results.items()):
        split_data = res.metrics_by_split.get(eval_partition, {})
        row = {"model": m_name}
        for col in headers[1:]:
            val = split_data.get(col, {}).get("mean")
            row[col] = f"{val:.6f}" if val is not None and not np.isnan(val) else ""
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_metrics_per_sample_csv(
    results: Dict[str, ClassicalModelResult],
    output_path: Path,
) -> Path:
    """Exports sample-level metrics across all models to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["sample_id", "model"]
    first_res = next(iter(results.values())) if results else None
    first_sample_metrics = next(iter(first_res.metrics_per_sample.values())) if first_res and first_res.metrics_per_sample else {}
    metric_cols = list(first_sample_metrics.keys())
    headers.extend(metric_cols)

    rows = []
    # Collect all sample IDs
    sample_ids = sorted({sid for res in results.values() for sid in res.metrics_per_sample.keys()})

    for sid in sample_ids:
        for m_name, res in sorted(results.items()):
            row = {"sample_id": sid, "model": m_name}
            s_metrics = res.metrics_per_sample.get(sid, {})
            for mc in metric_cols:
                v = s_metrics.get(mc)
                row[mc] = f"{v:.6f}" if v is not None and not np.isnan(v) else ""
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_metrics_by_split_csv(
    results: Dict[str, ClassicalModelResult],
    output_path: Path,
) -> Path:
    """Exports split-aggregated metrics (mean and std) across models to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = ["all", "train", "val", "test"]
    first_res = next(iter(results.values())) if results else None
    first_split_data = next(iter(first_res.metrics_by_split.values())) if first_res and first_res.metrics_by_split else {}
    metric_cols = list(first_split_data.keys())

    headers = ["split", "model"]
    for mc in metric_cols:
        headers.extend([f"{mc}_mean", f"{mc}_std"])

    rows = []
    for sp in splits:
        for m_name, res in sorted(results.items()):
            sp_data = res.metrics_by_split.get(sp, {})
            if not sp_data:
                continue
            row = {"split": sp, "model": m_name}
            for mc in metric_cols:
                stat = sp_data.get(mc, {})
                m_val = stat.get("mean")
                s_val = stat.get("std")
                row[f"{mc}_mean"] = f"{m_val:.6f}" if m_val is not None and not np.isnan(m_val) else ""
                row[f"{mc}_std"] = f"{s_val:.6f}" if s_val is not None and not np.isnan(s_val) else ""
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def render_classical_limitations_markdown(
    eval_partition: str,
    best_model_mae: Tuple[str, float],
    best_model_recall: Tuple[str, float],
    heuristic_ref: Optional[Dict[str, float]] = None,
) -> str:
    """Renders limitations.md summarizing Stage 4d empirical findings and caveats."""
    ref_mae_str = f"{heuristic_ref.get('mae', 0.0):.4f}" if heuristic_ref else "N/A"
    ref_rec_str = f"{heuristic_ref.get('recall_5pct', 0.0):.2%}" if heuristic_ref else "N/A"

    return f"""# Stage 4d Classical ML Baselines: Limitations & Generalization Disclaimers

## Overview
- **Evaluation Purpose**: Classical machine learning pipeline validation on grid-cell tabular formulation.
- **Formulation**: Each $64 \\times 64$ grid cell is an independent training row ($4,096$ rows per sample).
- **Target Channel**: `total_wirelength_density` (normalized post-route routing label).
- **Evaluation Partition**: `{eval_partition.upper()}`

## Methodological Boundaries & Limitations

1. **Tiny Dataset & Duplicate Layouts**:
   - The training set contains only 4 samples ($16,384$ rows) drawn from only **2 unique physical layouts** (`u45` and `u55`).
   - The validation sample (`u65_clk3p0`) and test sample (`u65_clk5p0`) are **identical physical layouts** with $0.0$ difference across all input and target channels.
   - Consequently, test set metrics replicate validation metrics with zero variance.

2. **Smoke-Only Pipeline Validation**:
   - These models and metrics are evaluated solely to verify feature matrix construction, model fitting, metric logging, and coefficient extraction.
   - They MUST NOT be cited or interpreted as scientific evidence of machine learning generalization on unseen physical designs.

3. **Spatial Independence Assumption in Tabular Regressors**:
   - Standard tabular models (Linear, Ridge, Lasso, Trees) treat each grid cell independently, discarding spatial 2D neighborhood context (receptive fields) except for coordinate channels (`physical/normalized_x`, `physical/normalized_y`).
   - Fully Convolutional Networks (FCNs / U-Nets) in Stage 5 will capture 2D spatial receptive fields natively.

## Comparison to Stage 4c Heuristic Reference Floor
- **Stage 4c Heuristic Reference (`simple_weighted_sum`)**:
  - Validation MAE: `{ref_mae_str}`
  - Validation Recall@5%: `{ref_rec_str}`
- **Stage 4d Best Classical ML Model**:
  - Best by MAE: `{best_model_mae[0]}` (MAE = {best_model_mae[1]:.4f})
  - Best by Recall@5%: `{best_model_recall[0]}` (Recall@5% = {best_model_recall[1]:.2%})
"""
