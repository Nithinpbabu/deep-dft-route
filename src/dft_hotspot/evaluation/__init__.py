"""
Evaluation package for physical design hotspot prediction.
Stage 4c: Spatial metrics and non-learning heuristic baselines.
"""

from dft_hotspot.evaluation.heuristics import (
    HeuristicBaseline,
    build_heuristic_predictions,
    evaluate_dataset_heuristics,
    evaluate_heuristics_on_sample,
    get_standard_heuristics,
)
from dft_hotspot.evaluation.metrics import (
    compute_all_metrics,
    dice_topk,
    iou_topk,
    mae,
    pearson_corr,
    precision_at_k,
    rmse,
    spearman_corr,
    topk_mask,
    topk_recall,
)

__all__ = [
    "mae",
    "rmse",
    "pearson_corr",
    "spearman_corr",
    "topk_mask",
    "topk_recall",
    "precision_at_k",
    "iou_topk",
    "dice_topk",
    "compute_all_metrics",
    "HeuristicBaseline",
    "get_standard_heuristics",
    "build_heuristic_predictions",
    "evaluate_heuristics_on_sample",
    "evaluate_dataset_heuristics",
]
