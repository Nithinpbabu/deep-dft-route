"""
Tabular Dataset Representation for Grid-Cell Level Hotspot Prediction.
Stage 4d of the DFT-Aware Physical Design Hotspot Prediction framework.

Flattens spatial 2D feature tensors (C_fused, H, W) and label maps (H, W)
into per-grid-cell tabular rows:
- X shape per sample: (H * W, C_fused) = (4096, 12)
- y shape per sample: (H * W,) float32
Enforces strict anti-leakage rules ensuring no post-route routing channels
are present in the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset

LEAKAGE_KEYWORDS = [
    "wirelength",
    "via_count",
    "labels_routing",
    "route_density",
    "metal1_wirelength",
    "metal2_wirelength",
    "metal3_wirelength",
    "metal4plus_wirelength",
]


@dataclass
class SampleTabularData:
    """Tabular feature matrix, target vector, and coordinate metadata for a single sample."""

    sample_id: str
    split: str
    X: np.ndarray              # (H * W, C) float32
    y: np.ndarray              # (H * W,) float32
    row_indices: np.ndarray    # (H * W,) int
    col_indices: np.ndarray    # (H * W,) int
    grid_shape: Tuple[int, int]
    feature_names: List[str]
    target_channel: str


@dataclass
class TabularDataset:
    """
    Unified tabular dataset container partitioning grid cells into train, val, and test matrices.
    """

    target_channel: str
    feature_names: List[str]
    grid_shape: Tuple[int, int]

    # Partition matrices
    X_train: np.ndarray  # (N_train * 4096, C)
    y_train: np.ndarray  # (N_train * 4096,)
    X_val: np.ndarray    # (N_val * 4096, C)
    y_val: np.ndarray    # (N_val * 4096,)
    X_test: np.ndarray   # (N_test * 4096, C)
    y_test: np.ndarray   # (N_test * 4096,)

    # Sample-level mapping
    samples: Dict[str, SampleTabularData] = field(default_factory=dict)
    train_sample_ids: List[str] = field(default_factory=list)
    val_sample_ids: List[str] = field(default_factory=list)
    test_sample_ids: List[str] = field(default_factory=list)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @property
    def total_cells(self) -> int:
        return len(self.y_train) + len(self.y_val) + len(self.y_test)


def sample_to_grid_rows(
    sample: PhysicalDesignSample,
    target_channel: str = "total_wirelength_density",
    split: str = "unassigned",
) -> SampleTabularData:
    """
    Flattens a PhysicalDesignSample's fused input tensor and target label channel into tabular rows.

    Args:
        sample: PhysicalDesignSample instance.
        target_channel: Name of the post-route label channel to extract as ground truth.
        split: Partition identifier ('train', 'val', 'test', 'unassigned').

    Returns:
        SampleTabularData container.

    Raises:
        ValueError: If leakage detected or target channel is missing.
    """
    fused_tensor = sample.get_fused_input_tensor(include_scan=True)  # (C, H, W)
    feature_names = sample.get_fused_channel_names(include_scan=True)

    # Strict anti-leakage check
    for fname in feature_names:
        low = fname.lower()
        # Ensure scan proxy wirelength is allowed (scan/scan_chain_local_wirelength_proxy),
        # but post-route routing labels are strictly blocked
        if "labels" in low or ("wirelength" in low and not fname.startswith("scan/")):
            raise ValueError(f"Target leakage detected in input features: '{fname}'")

    if target_channel not in sample.label_channel_names:
        raise KeyError(
            f"Target channel '{target_channel}' not found in sample label channels: "
            f"{sample.label_channel_names}"
        )

    t_idx = sample.label_channel_names.index(target_channel)
    target_2d = sample.label_tensor[t_idx].copy().astype(np.float32)  # (H, W)

    C, H, W = fused_tensor.shape

    # Row-major flattening: (H, W, C) -> (H * W, C)
    X = np.moveaxis(fused_tensor, 0, -1).reshape(-1, C).astype(np.float32)
    y = target_2d.ravel().astype(np.float32)

    # Coordinate tracking
    row_indices = np.repeat(np.arange(H, dtype=np.int32), W)
    col_indices = np.tile(np.arange(W, dtype=np.int32), H)

    # Verify finite numbers
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Feature matrix for sample '{sample.sample_id}' contains NaN or Inf.")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"Target vector for sample '{sample.sample_id}' contains NaN or Inf.")

    return SampleTabularData(
        sample_id=sample.sample_id,
        split=split,
        X=X,
        y=y,
        row_indices=row_indices,
        col_indices=col_indices,
        grid_shape=(H, W),
        feature_names=feature_names,
        target_channel=target_channel,
    )


def load_tabular_dataset(
    dataset: PhysicalDesignDataset,
    splits: Optional[Dict[str, Any]] = None,
    target_channel: str = "total_wirelength_density",
) -> TabularDataset:
    """
    Constructs a unified TabularDataset from a PhysicalDesignDataset and split assignments.

    Args:
        dataset: PhysicalDesignDataset instance.
        splits: Dict specifying 'train', 'val', and 'test' sample ID lists.
        target_channel: Target label channel name.

    Returns:
        TabularDataset container with train, val, and test matrices.
    """
    train_ids = splits.get("train", []) if splits else []
    val_ids = splits.get("val", []) if splits else []
    test_ids = splits.get("test", []) if splits else []

    sample_dict: Dict[str, SampleTabularData] = {}
    feature_names: Optional[List[str]] = None
    grid_shape: Optional[Tuple[int, int]] = None

    for sid, rec in dataset.items():
        if rec.sample is None:
            continue

        split_label = "train" if sid in train_ids else ("val" if sid in val_ids else ("test" if sid in test_ids else "all"))
        sample_tab = sample_to_grid_rows(rec.sample, target_channel=target_channel, split=split_label)
        sample_dict[sid] = sample_tab

        if feature_names is None:
            feature_names = sample_tab.feature_names
            grid_shape = sample_tab.grid_shape
        else:
            if sample_tab.feature_names != feature_names:
                raise ValueError(f"Feature names mismatch for sample '{sid}'")
            if sample_tab.grid_shape != grid_shape:
                raise ValueError(f"Grid shape mismatch for sample '{sid}'")

    if not sample_dict:
        raise ValueError("Dataset contains zero valid samples.")

    assert feature_names is not None
    assert grid_shape is not None
    num_features = len(feature_names)

    # Aggregate by split
    def _concat_split(sids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        valid_samples = [sample_dict[sid] for sid in sids if sid in sample_dict]
        if not valid_samples:
            return np.empty((0, num_features), dtype=np.float32), np.empty((0,), dtype=np.float32)
        X_cat = np.concatenate([s.X for s in valid_samples], axis=0)
        y_cat = np.concatenate([s.y for s in valid_samples], axis=0)
        return X_cat, y_cat

    # If no splits provided, all samples go into train
    if not train_ids and not val_ids and not test_ids:
        train_ids = list(sample_dict.keys())

    X_train, y_train = _concat_split(train_ids)
    X_val, y_val = _concat_split(val_ids)
    X_test, y_test = _concat_split(test_ids)

    return TabularDataset(
        target_channel=target_channel,
        feature_names=feature_names,
        grid_shape=grid_shape,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        samples=sample_dict,
        train_sample_ids=train_ids,
        val_sample_ids=val_ids,
        test_sample_ids=test_ids,
    )
