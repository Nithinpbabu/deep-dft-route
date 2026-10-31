"""
Non-Learning Heuristic Baselines for Physical Design Hotspot Prediction.
Stage 4c of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- Standard non-learned spatial heuristic baselines:
  1. pin_density (physical/pin_density)
  2. cell_density (physical/cell_density)
  3. sequential_cell_density (physical/sequential_cell_density)
  4. scan_cell_density (scan/scan_cell_density)
  5. scan_segment_density (scan/scan_chain_segment_density)
  6. scan_local_wirelength_proxy (scan/scan_chain_local_wirelength_proxy)
  7. simple_weighted_sum (heuristic combination: 0.35 pin + 0.25 cell + 0.25 scan_seg + 0.15 scan_wl)
- Sample-level and dataset-level evaluation drivers.
- Per-sample and split-level aggregation routines.
- CSV and JSON exporters for experiment tracking.
"""

from __future__ import annotations

import csv
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset
from dft_hotspot.evaluation.metrics import compute_all_metrics, sanitize_metrics_for_json


@dataclass
class HeuristicBaseline:
    """
    Defines a non-learning spatial heuristic baseline.
    """

    name: str
    description: str
    channel_name: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    normalize: bool = True

    def predict(self, fused_tensor: np.ndarray, channel_names: List[str]) -> np.ndarray:
        """
        Extracts or computes the 2D spatial prediction map from the fused input tensor.

        Args:
            fused_tensor: (C_fused, H, W) numpy array.
            channel_names: List of C_fused prefixed channel names.

        Returns:
            (H, W) float32 prediction map normalized to [0, 1].
        """
        ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}

        if self.channel_name is not None:
            if self.channel_name not in ch_to_idx:
                raise KeyError(
                    f"Baseline '{self.name}' requires channel '{self.channel_name}', "
                    f"which was not found in fused channels: {channel_names}"
                )
            idx = ch_to_idx[self.channel_name]
            pred = fused_tensor[idx].copy().astype(np.float32)

        elif self.weights is not None:
            pred = np.zeros(fused_tensor.shape[1:], dtype=np.float32)
            for ch_name, weight in self.weights.items():
                if ch_name not in ch_to_idx:
                    raise KeyError(
                        f"Baseline '{self.name}' requires channel '{ch_name}', "
                        f"which was not found in fused channels: {channel_names}"
                    )
                idx = ch_to_idx[ch_name]
                pred += float(weight) * fused_tensor[idx].astype(np.float32)
        else:
            raise ValueError(f"Baseline '{self.name}' must have either channel_name or weights defined.")

        if self.normalize:
            max_val = float(np.max(pred))
            if max_val > 0.0:
                pred = pred / max_val

        return pred


def get_standard_heuristics() -> List[HeuristicBaseline]:
    """
    Returns the standard suite of 7 non-learning heuristic baselines for Stage 4c.
    """
    return [
        HeuristicBaseline(
            name="pin_density",
            description="Pure pin-density spatial hotspot proxy from physical/pin_density",
            channel_name="physical/pin_density",
            normalize=True,
        ),
        HeuristicBaseline(
            name="cell_density",
            description="Standard cell area density proxy from physical/cell_density",
            channel_name="physical/cell_density",
            normalize=True,
        ),
        HeuristicBaseline(
            name="sequential_cell_density",
            description="Sequential flip-flop cell density from physical/sequential_cell_density",
            channel_name="physical/sequential_cell_density",
            normalize=True,
        ),
        HeuristicBaseline(
            name="scan_cell_density",
            description="Scan-chain cell density proxy from scan/scan_cell_density",
            channel_name="scan/scan_cell_density",
            normalize=True,
        ),
        HeuristicBaseline(
            name="scan_segment_density",
            description="Scan-chain routing Manhattan segment density from scan/scan_chain_segment_density",
            channel_name="scan/scan_chain_segment_density",
            normalize=True,
        ),
        HeuristicBaseline(
            name="scan_local_wirelength_proxy",
            description="Scan-chain localized wirelength demand proxy from scan/scan_chain_local_wirelength_proxy",
            channel_name="scan/scan_chain_local_wirelength_proxy",
            normalize=True,
        ),
        HeuristicBaseline(
            name="simple_weighted_sum",
            description=(
                "Heuristic non-learned weighted sum: "
                "0.35 pin + 0.25 cell + 0.25 scan_segment + 0.15 scan_local_wirelength"
            ),
            weights={
                "physical/pin_density": 0.35,
                "physical/cell_density": 0.25,
                "scan/scan_chain_segment_density": 0.25,
                "scan/scan_chain_local_wirelength_proxy": 0.15,
            },
            normalize=True,
        ),
    ]


def build_heuristic_predictions(
    sample: PhysicalDesignSample,
    baselines: Optional[List[HeuristicBaseline]] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes 2D spatial prediction maps for all requested baselines on a single sample.

    Returns:
        Dict mapping baseline_name -> (H, W) float32 prediction array.
    """
    if baselines is None:
        baselines = get_standard_heuristics()

    fused_tensor = sample.get_fused_input_tensor(include_scan=True)
    fused_names = sample.get_fused_channel_names(include_scan=True)

    predictions: Dict[str, np.ndarray] = {}
    for base in baselines:
        predictions[base.name] = base.predict(fused_tensor, fused_names)

    return predictions


def evaluate_heuristics_on_sample(
    sample: PhysicalDesignSample,
    target_channel: str = "total_wirelength_density",
    baselines: Optional[List[HeuristicBaseline]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all heuristic baselines against a target routing label channel on one sample.

    Returns:
        Dict mapping baseline_name -> {metric_name -> float_value}.
    """
    if baselines is None:
        baselines = get_standard_heuristics()

    if target_channel not in sample.label_channel_names:
        raise KeyError(
            f"Target channel '{target_channel}' not found in sample label channels: "
            f"{sample.label_channel_names}"
        )

    t_idx = sample.label_channel_names.index(target_channel)
    target_2d = sample.label_tensor[t_idx].copy().astype(np.float32)

    predictions = build_heuristic_predictions(sample, baselines=baselines)

    sample_metrics: Dict[str, Dict[str, float]] = {}
    for base_name, pred_2d in predictions.items():
        sample_metrics[base_name] = compute_all_metrics(pred_2d, target_2d)

    return sample_metrics


def evaluate_heuristics_on_dataset(
    dataset: PhysicalDesignDataset,
    splits: Optional[Dict[str, Any]] = None,
    target_channel: str = "total_wirelength_density",
    baselines: Optional[List[HeuristicBaseline]] = None,
) -> Dict[str, Any]:
    """
    Evaluates heuristic baselines across all samples in a PhysicalDesignDataset
    and computes aggregate metrics by split partition (train, val, test, all).

    Returns:
        Dict adhering to Stage 4c heuristic evaluation report schema.
    """
    if baselines is None:
        baselines = get_standard_heuristics()

    baseline_names = [b.name for b in baselines]

    # Evaluate each sample
    per_sample_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sid, rec in dataset.items():
        if rec.sample is None:
            continue
        per_sample_results[sid] = evaluate_heuristics_on_sample(
            rec.sample,
            target_channel=target_channel,
            baselines=baselines,
        )

    # Determine split groupings
    split_groups: Dict[str, List[str]] = {
        "all": list(per_sample_results.keys()),
    }

    if splits is not None:
        if "train" in splits:
            split_groups["train"] = [s for s in splits["train"] if s in per_sample_results]
        if "val" in splits:
            split_groups["val"] = [s for s in splits["val"] if s in per_sample_results]
        if "test" in splits:
            split_groups["test"] = [s for s in splits["test"] if s in per_sample_results]

    # Aggregate metrics across each split partition
    # Structure: by_split[split_name][baseline_name][metric_name] = {"mean": float, "std": float, "count": int}
    by_split: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}

    sample_first = next(iter(per_sample_results.values())) if per_sample_results else {}
    metric_keys = list(next(iter(sample_first.values())).keys()) if sample_first else []

    for split_name, sids in split_groups.items():
        by_split[split_name] = {}
        if not sids:
            continue

        for b_name in baseline_names:
            by_split[split_name][b_name] = {}
            for m_key in metric_keys:
                vals = []
                for sid in sids:
                    v = per_sample_results[sid][b_name].get(m_key)
                    if v is not None and not np.isnan(v):
                        vals.append(v)

                if vals:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
                    by_split[split_name][b_name][m_key] = {
                        "mean": mean_val,
                        "std": std_val,
                        "valid_count": len(vals),
                    }
                else:
                    by_split[split_name][b_name][m_key] = {
                        "mean": None,
                        "std": None,
                        "valid_count": 0,
                    }

    # Identify best baseline across 'all' (or 'val' if available) by lowest MAE and highest recall_5pct
    eval_partition = "val" if "val" in by_split and by_split["val"] else "all"
    best_by_mae: Optional[Tuple[str, float]] = None
    best_by_recall: Optional[Tuple[str, float]] = None
    worst_by_mae: Optional[Tuple[str, float]] = None

    for b_name in baseline_names:
        stats = by_split[eval_partition].get(b_name, {})
        mae_stat = stats.get("mae", {}).get("mean")
        rec_stat = stats.get("recall_5pct", {}).get("mean")

        if mae_stat is not None:
            if best_by_mae is None or mae_stat < best_by_mae[1]:
                best_by_mae = (b_name, mae_stat)
            if worst_by_mae is None or mae_stat > worst_by_mae[1]:
                worst_by_mae = (b_name, mae_stat)

        if rec_stat is not None:
            if best_by_recall is None or rec_stat > best_by_recall[1]:
                best_by_recall = (b_name, rec_stat)

    report: Dict[str, Any] = {
        "schema_version": "heuristic_evaluation_v001",
        "dataset_id": dataset.dataset_id,
        "design": dataset.design,
        "platform": dataset.platform,
        "target_channel": target_channel,
        "sample_count": len(per_sample_results),
        "split_counts": {k: len(v) for k, v in split_groups.items()},
        "normalization_policy": (
            "per_sample_max: each prediction map is scaled by 1.0 / max(pred) if max(pred) > 0, "
            "aligning with the per-sample normalized ground-truth routing labels."
        ),
        "evaluation_partition_used": eval_partition,
        "summary": {
            "best_baseline_by_mae": best_by_mae[0] if best_by_mae else None,
            "best_mae": best_by_mae[1] if best_by_mae else None,
            "worst_baseline_by_mae": worst_by_mae[0] if worst_by_mae else None,
            "worst_mae": worst_by_mae[1] if worst_by_mae else None,
            "best_baseline_by_recall_5pct": best_by_recall[0] if best_by_recall else None,
            "best_recall_5pct": best_by_recall[1] if best_by_recall else None,
        },
        "baselines_evaluated": [
            {
                "name": b.name,
                "description": b.description,
                "channel_name": b.channel_name,
                "weights": b.weights,
            }
            for b in baselines
        ],
        "by_split": by_split,
        "per_sample": per_sample_results,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return report


def export_metrics_per_sample_csv(
    report: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Exports per-sample evaluation metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample = report.get("per_sample", {})

    rows = []
    headers = ["sample_id", "baseline"]
    header_set = False

    for sid, base_dict in sorted(per_sample.items()):
        for b_name, m_dict in sorted(base_dict.items()):
            row = {"sample_id": sid, "baseline": b_name}
            if not header_set:
                for k in m_dict.keys():
                    headers.append(k)
                header_set = True
            for k, v in m_dict.items():
                row[k] = "" if v is None or np.isnan(v) else f"{v:.6f}"
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_metrics_by_split_csv(
    report: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Exports aggregated metrics by split partition to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_split = report.get("by_split", {})

    headers = ["split", "baseline"]
    metrics_list: List[str] = []
    for split_data in by_split.values():
        for b_data in split_data.values():
            for m_key in b_data.keys():
                if m_key not in metrics_list:
                    metrics_list.append(m_key)
                    headers.extend([f"{m_key}_mean", f"{m_key}_std"])
            break
        if metrics_list:
            break

    rows = []
    for split_name, base_dict in sorted(by_split.items()):
        for b_name, m_dict in sorted(base_dict.items()):
            row = {"split": split_name, "baseline": b_name}
            for m_key in metrics_list:
                stat = m_dict.get(m_key, {})
                mean_v = stat.get("mean")
                std_v = stat.get("std")
                row[f"{m_key}_mean"] = "" if mean_v is None or np.isnan(mean_v) else f"{mean_v:.6f}"
                row[f"{m_key}_std"] = "" if std_v is None or np.isnan(std_v) else f"{std_v:.6f}"
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def render_limitations_markdown(report: Dict[str, Any]) -> str:
    """Renders limitations.md summarizing Stage 4c scope boundaries."""
    summary = report.get("summary", {})
    return f"""# Stage 4c Limitations & Generalization Disclaimers

## Overview
- **Dataset ID**: `{report.get('dataset_id')}`
- **Design / Platform**: `{report.get('design')}` / `{report.get('platform')}`
- **Target Channel**: `{report.get('target_channel')}`
- **Evaluation Purpose**: Verification of metric computation and heuristic baseline pipeline readiness.

## Methodological Boundaries & Limitations

1. **Non-Learning Heuristics Only**:
   - None of the predictors evaluated in Stage 4c are trained or learned models.
   - All weights (e.g. in `simple_weighted_sum`) are arbitrarily specified heuristic coefficients based on EDA domain intuitions, NOT optimization against loss functions.

2. **Single Design & Tiny Sample Count**:
   - The dataset contains only 6 runs from a single small benchmark design (`uart_tx`).
   - Across the 6 runs, there are only **3 unique physical layouts** because timing slack invariance caused identical placement/routing solutions for clock periods 3.0 ns and 5.0 ns.

3. **Split Identity Warning**:
   - Under the `smoke_config_holdout` split policy, the validation sample (`uart_tx_nangate45_u65_clk3p0`) and test sample (`uart_tx_nangate45_u65_clk5p0`) are **identical physical layouts** with zero difference in either features or routing labels.
   - Any metric performance reported on this test split is strictly code-path smoke verification and must NEVER be cited as empirical evidence of generalization.

4. **Per-Sample Normalization Constraint**:
   - Each heuristic predictor is normalized to $[0.0, 1.0]$ using per-sample maximum scaling: $\\hat{{y}} = x / \\max(x)$.
   - While this matches the per-sample normalization of the ground-truth routing labels extracted from DEF in Stage 2b, it assumes knowledge of the peak density within each layout tile.

5. **Hotspot Sparsity & Tie Resolution**:
   - Routing congestion and cell placements in `uart_tx` occupy less than $10\\%$ of spatial bins. Top-k thresholds at $5\\%$ and $10\\%$ operate on sparse regions where small pixel shifts can cause sharp drops in IoU and Dice scores.

## Baseline Summary (Evaluation Partition: `{report.get('evaluation_partition_used')}`)
- **Best Baseline by MAE**: `{summary.get('best_baseline_by_mae')}` (MAE = {summary.get('best_mae'):.4f})
- **Worst Baseline by MAE**: `{summary.get('worst_baseline_by_mae')}` (MAE = {summary.get('worst_mae'):.4f})
- **Best Baseline by Top-5% Recall**: `{summary.get('best_baseline_by_recall_5pct')}` (Recall@5% = {summary.get('best_recall_5pct'):.2%})
"""


# Function aliases for API consistency
evaluate_sample_heuristics = evaluate_heuristics_on_sample
evaluate_dataset_heuristics = evaluate_heuristics_on_dataset

