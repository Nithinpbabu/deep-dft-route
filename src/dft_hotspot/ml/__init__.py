"""
Machine learning package for physical design hotspot prediction.
"""

from dft_hotspot.ml.classical_baselines import (
    ClassicalModelResult,
    MeanBaselineRegressor,
    export_coefficients_csv,
    export_feature_importance_csv,
    export_metrics_by_split_csv,
    export_metrics_per_model_csv,
    export_metrics_per_sample_csv,
    train_classical_baselines,
)
from dft_hotspot.ml.tabular_dataset import (
    SampleTabularData,
    TabularDataset,
    load_tabular_dataset,
    sample_to_grid_rows,
)
from dft_hotspot.ml.torch_multimodal_dataset import (
    MultimodalPhysicalDesignDataset,
    collate_multimodal,
    create_multimodal_dataloaders,
)
from dft_hotspot.ml.train_fusion import (
    evaluate_fusion_model,
    train_fusion,
    train_fusion_epoch,
)

__all__ = [
    "SampleTabularData",
    "TabularDataset",
    "sample_to_grid_rows",
    "load_tabular_dataset",
    "MeanBaselineRegressor",
    "ClassicalModelResult",
    "train_classical_baselines",
    "export_coefficients_csv",
    "export_feature_importance_csv",
    "export_metrics_per_model_csv",
    "export_metrics_per_sample_csv",
    "export_metrics_by_split_csv",
    "MultimodalPhysicalDesignDataset",
    "collate_multimodal",
    "create_multimodal_dataloaders",
    "train_fusion_epoch",
    "evaluate_fusion_model",
    "train_fusion",
]
