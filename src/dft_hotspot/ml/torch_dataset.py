"""
PyTorch Dataset Representation for Dense 2D Hotspot Prediction.
Stage 5b of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- TorchPhysicalDesignDataset: PyTorch Dataset wrapping physical design samples.
  - Extracts fused input tensor (C_fused, H, W) = (12, 64, 64)
  - Extracts target routing label tensor (1, H, W) = (1, 64, 64)
  - Enforces strict anti-leakage checks (no post-route channels in input features)
  - Supports in-memory preloading for zero I/O latency during training.
- Helper functions to build train/val/test DataLoader instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore

from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.data.dataset_collection import PhysicalDesignDataset

LEAKAGE_KEYWORDS = [
    "labels_routing",
    "route_density",
    "metal1_wirelength",
    "metal2_wirelength",
    "metal3_wirelength",
    "metal4plus_wirelength",
]


class TorchPhysicalDesignDataset(Dataset):
    """
    PyTorch Dataset providing dense 2D spatial feature tensors and target label maps.

    Shapes:
        input_tensor:  (C_fused, H, W) = (12, 64, 64) float32
        target_tensor: (C_target, H, W) float32 (e.g. (1, 64, 64) or (8, 64, 64))
    """

    def __init__(
        self,
        samples: List[PhysicalDesignSample],
        target_channel: Union[str, List[str]] = "total_wirelength_density",
        split_name: str = "unassigned",
        preload: bool = True,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for TorchPhysicalDesignDataset.")

        if not samples:
            raise ValueError("Cannot initialize TorchPhysicalDesignDataset with empty samples list.")

        self.target_channel = target_channel
        self.is_multi_target = (target_channel == "all") or isinstance(target_channel, (list, tuple))
        self.target_channels: Optional[List[str]] = None
        if isinstance(target_channel, (list, tuple)):
            self.target_channels = [str(c) for c in target_channel]
        elif target_channel != "all":
            self.target_channels = [str(target_channel)]

        self.split_name = split_name
        self.preload = preload

        self.sample_ids: List[str] = []
        self.designs: List[str] = []
        self.platforms: List[str] = []
        self.samples_list: List[PhysicalDesignSample] = []

        self.feature_names: Optional[List[str]] = None
        self.grid_shape: Optional[Tuple[int, int]] = None

        # Preloaded tensor buffers
        self._cached_inputs: List[torch.Tensor] = []
        self._cached_targets: List[torch.Tensor] = []

        for sample in samples:
            self._validate_and_register(sample)

    def _validate_and_register(self, sample: PhysicalDesignSample) -> None:
        """Validates sample anti-leakage and extracts/preloads tensors."""
        fused_tensor = sample.get_fused_input_tensor(include_scan=True)  # (C, H, W)
        fused_names = sample.get_fused_channel_names(include_scan=True)

        # Anti-leakage verification
        for fname in fused_names:
            low = fname.lower()
            if "labels" in low or ("wirelength" in low and not fname.startswith("scan/")):
                raise ValueError(f"Target leakage detected in input features for '{sample.sample_id}': '{fname}'")

        if self.target_channels is None:
            # target_channel == "all"
            self.target_channels = list(sample.label_channel_names)

        target_indices: List[int] = []
        for ch in self.target_channels:
            if ch not in sample.label_channel_names:
                raise KeyError(
                    f"Target channel '{ch}' not found in sample '{sample.sample_id}' "
                    f"label channels: {sample.label_channel_names}"
                )
            target_indices.append(sample.label_channel_names.index(ch))

        target_extracted = sample.label_tensor[target_indices].copy().astype(np.float32)  # (C_target, H, W)

        if not np.all(np.isfinite(fused_tensor)):
            raise ValueError(f"Feature tensor for sample '{sample.sample_id}' contains NaN or Inf.")
        if not np.all(np.isfinite(target_extracted)):
            raise ValueError(f"Target tensor for sample '{sample.sample_id}' contains NaN or Inf.")

        if self.feature_names is None:
            self.feature_names = fused_names
            self.grid_shape = (fused_tensor.shape[1], fused_tensor.shape[2])
        else:
            if fused_names != self.feature_names:
                raise ValueError(f"Feature names mismatch for sample '{sample.sample_id}'")
            if (fused_tensor.shape[1], fused_tensor.shape[2]) != self.grid_shape:
                raise ValueError(f"Grid shape mismatch for sample '{sample.sample_id}'")

        self.sample_ids.append(sample.sample_id)
        if isinstance(sample.feature_metadata, dict):
            design = sample.feature_metadata.get("design", sample.feature_metadata.get("design_name", "unknown"))
            pdk = sample.feature_metadata.get("pdk", sample.feature_metadata.get("platform", "unknown"))
        elif sample.feature_metadata is not None:
            design = getattr(sample.feature_metadata, "design_name", getattr(sample.feature_metadata, "design", "unknown"))
            pdk = getattr(sample.feature_metadata, "platform", getattr(sample.feature_metadata, "pdk", "unknown"))
        else:
            design = "unknown"
            pdk = "unknown"
        self.designs.append(str(design))
        self.platforms.append(str(pdk))
        self.samples_list.append(sample)

        if self.preload:
            # Convert to torch.FloatTensor: (C, H, W) and (C_target, H, W)
            x_t = torch.from_numpy(fused_tensor.astype(np.float32))
            y_t = torch.from_numpy(target_extracted.astype(np.float32))
            self._cached_inputs.append(x_t)
            self._cached_targets.append(y_t)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary for sample at index idx:
            - 'input': torch.FloatTensor of shape (C_fused, H, W)
            - 'target': torch.FloatTensor of shape (C_target, H, W)
            - 'sample_id': str
            - 'design': str
            - 'platform': str
        """
        if self.preload:
            x_t = self._cached_inputs[idx]
            y_t = self._cached_targets[idx]
        else:
            sample = self.samples_list[idx]
            fused_tensor = sample.get_fused_input_tensor(include_scan=True).astype(np.float32)
            assert self.target_channels is not None
            target_indices = [sample.label_channel_names.index(c) for c in self.target_channels]
            target_extracted = sample.label_tensor[target_indices].astype(np.float32)

            x_t = torch.from_numpy(fused_tensor)
            y_t = torch.from_numpy(target_extracted)

        return {
            "input": x_t,
            "target": y_t,
            "sample_id": self.sample_ids[idx],
            "design": self.designs[idx],
            "platform": self.platforms[idx],
        }

    @property
    def in_channels(self) -> int:
        return len(self.feature_names) if self.feature_names is not None else 0

    @property
    def out_channels(self) -> int:
        return len(self.target_channels) if self.target_channels is not None else 0

    @property
    def target_channel_names(self) -> List[str]:
        return list(self.target_channels) if self.target_channels is not None else []


def build_split_datasets(
    dataset: PhysicalDesignDataset,
    splits: Dict[str, List[str]],
    target_channel: Union[str, List[str]] = "total_wirelength_density",
    preload: bool = True,
) -> Dict[str, TorchPhysicalDesignDataset]:
    """
    Builds TorchPhysicalDesignDataset instances for train, val, and test partitions.

    Args:
        dataset: PhysicalDesignDataset collection.
        splits: Dict containing 'train', 'val', 'test' sample ID lists.
        target_channel: Name of post-route routing channel to target.
        preload: Whether to keep tensors in RAM.

    Returns:
        Dict mapping split_name ('train', 'val', 'test') -> TorchPhysicalDesignDataset.
    """
    split_datasets: Dict[str, TorchPhysicalDesignDataset] = {}

    for split_key in ["train", "val", "test"]:
        sids = splits.get(split_key, [])
        split_samples: List[PhysicalDesignSample] = []
        for sid in sids:
            rec = dataset.samples.get(sid)
            if rec is not None and rec.sample is not None:
                split_samples.append(rec.sample)

        if split_samples:
            split_datasets[split_key] = TorchPhysicalDesignDataset(
                samples=split_samples,
                target_channel=target_channel,
                split_name=split_key,
                preload=preload,
            )

    return split_datasets


def create_dataloaders(
    split_datasets: Dict[str, TorchPhysicalDesignDataset],
    batch_size: int = 4,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoader instances for the given split datasets.

    Args:
        split_datasets: Dict of split datasets.
        batch_size: Mini-batch size.
        num_workers: Number of DataLoader subprocesses (0 for fast in-memory loading).

    Returns:
        Dict mapping split_name -> DataLoader.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for DataLoader creation.")

    dataloaders: Dict[str, DataLoader] = {}
    for split_key, ds in split_datasets.items():
        is_train = split_key == "train"
        dataloaders[split_key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return dataloaders
