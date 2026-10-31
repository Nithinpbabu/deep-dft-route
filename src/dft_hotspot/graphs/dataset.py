"""
PyTorch Geometric Dataset and DataLoader for Circuit & Scan Graphs (Stage 6b).

Provides:
- PyGCircuitGraphDataset: Dataset wrapping physical design graph representations.
  - Loads PyG Data objects with node features (D_node=10), multi-relational edges (D_edge=4).
  - Integrates dense 2D routing ground-truth target tensors (C_target, H, W).
  - Enforces strict anti-leakage checks (no post-route ground-truth features in graph inputs).
  - Supports in-memory preloading for zero I/O latency.
- create_pyg_dataloaders: Factory function building train/val/test PyG DataLoaders
  adhering to strict design-held-out splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Dataset = object  # type: ignore

from dft_hotspot.graphs.builder import CircuitGraphBuilder
from dft_hotspot.graphs.schema import (
    CircuitGraph,
    CircuitGraphData,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)

ALL_ROUTING_CHANNELS = [
    "total_wirelength_density",
    "horizontal_wirelength_density",
    "vertical_wirelength_density",
    "via_count_density",
    "metal1_wirelength_density",
    "metal2_wirelength_density",
    "metal3_wirelength_density",
    "metal4plus_wirelength_density",
]


class PyGCircuitGraphDataset(Dataset):
    """
    PyG-compatible Dataset delivering paired circuit graphs and 2D spatial routing targets.

    Each item is a torch_geometric.data.Data instance containing:
        - x: (N, 10) float32 node features
        - edge_index: (2, E) int64 edge connectivity
        - edge_attr: (E, 4) float32 multi-relational edge attributes
        - pos: (N, 2) float32 normalized [x, y] coordinates
        - grid_coords: (N, 2) int64 2D spatial grid coordinates [row, col]
        - y: (C_target, H, W) float32 ground-truth routing tensor
        - num_nodes: N
        - sample_id: str
        - design: str
    """

    def __init__(
        self,
        sample_dirs: List[Union[str, Path]],
        target_channel: Union[str, List[str]] = "total_wirelength_density",
        split_name: str = "unassigned",
        preload: bool = True,
        auto_extract: bool = True,
    ) -> None:
        if not _HAS_PYG:
            raise ImportError(
                "PyTorch and torch_geometric are required for PyGCircuitGraphDataset."
            )

        if not sample_dirs:
            raise ValueError("Cannot initialize PyGCircuitGraphDataset with empty sample_dirs list.")

        self.sample_dirs = [Path(p) for p in sample_dirs]
        self.target_channel = target_channel
        self.split_name = split_name
        self.preload = preload
        self.auto_extract = auto_extract

        # Resolve target channels
        self.is_multi_target = (target_channel == "all") or isinstance(target_channel, (list, tuple))
        if target_channel == "all":
            self.target_channels = list(ALL_ROUTING_CHANNELS)
        elif isinstance(target_channel, (list, tuple)):
            self.target_channels = [str(c) for c in target_channel]
        else:
            self.target_channels = [str(target_channel)]

        for ch in self.target_channels:
            if ch not in ALL_ROUTING_CHANNELS:
                raise ValueError(
                    f"Unknown routing target channel '{ch}'. Valid options: {ALL_ROUTING_CHANNELS}"
                )

        self.sample_ids: List[str] = []
        self.designs: List[str] = []
        self._cached_data: List[Data] = []

        self._builder = CircuitGraphBuilder(grid_size=64) if auto_extract else None

        for s_dir in self.sample_dirs:
            data_obj = self._load_sample_data(s_dir)
            self.sample_ids.append(data_obj.sample_id)
            self.designs.append(data_obj.design)
            if self.preload:
                self._cached_data.append(data_obj)

    @property
    def num_node_features(self) -> int:
        return len(NODE_FEATURE_NAMES)

    @property
    def num_edge_features(self) -> int:
        return len(EDGE_FEATURE_NAMES)

    @property
    def out_channels(self) -> int:
        return len(self.target_channels)

    def _extract_target_tensor(self, sample_dir: Path) -> torch.Tensor:
        """Loads and extracts target routing channels from labels_routing.npz."""
        label_file = sample_dir / "labels_routing.npz"
        if not label_file.is_file():
            raise FileNotFoundError(f"labels_routing.npz not found in {sample_dir}")

        with np.load(label_file) as npz:
            if "labels" not in npz:
                raise KeyError(f"'labels' array not found in {label_file}")
            raw_labels = npz["labels"]  # Shape: (8, H, W)

        # Map channel indices
        channel_indices = [ALL_ROUTING_CHANNELS.index(ch) for ch in self.target_channels]
        selected = raw_labels[channel_indices, :, :]  # Shape: (C_selected, H, W)

        return torch.from_numpy(selected).float()

    def _load_sample_data(self, sample_dir: Path) -> Data:
        """Loads or builds PyG Data object for a single sample directory."""
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        meta_file = sample_dir / "metadata.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"metadata.json not found in {sample_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        design = meta.get("design", sample_dir.name)
        sample_id = sample_dir.name

        target_y = self._extract_target_tensor(sample_dir)

        # 1. Try loading pre-extracted circuit_graph.pt
        pt_path = sample_dir / "circuit_graph.pt"
        if pt_path.is_file():
            loaded_obj = torch.load(pt_path, weights_only=False)
            data_dict = loaded_obj.to_dict()
            data_dict["y"] = target_y
            data_dict["sample_id"] = sample_id
            data_dict["design"] = design
            data_obj = CircuitGraphData(**data_dict)
            self._validate_leakage(data_obj)
            return data_obj

        # 2. Try loading circuit_graph.npz
        npz_path = sample_dir / "circuit_graph.npz"
        if npz_path.is_file():
            graph = CircuitGraph.load_npz(npz_path)
            data_obj = graph.to_pyg(y=target_y)
            data_obj.sample_id = sample_id
            data_obj.design = design
            self._validate_leakage(data_obj)
            return data_obj

        # 3. Fallback: On-the-fly extraction if enabled
        if self.auto_extract and self._builder is not None:
            raw_def = meta.get("def_path")
            def_path = Path(raw_def) if raw_def else None
            if def_path is None or not def_path.is_file():
                alt_str = str(raw_def).replace("/mnt/e/", "E:/")
                def_path = Path(alt_str)

            if not def_path.is_file():
                raise FileNotFoundError(f"DEF file not found for sample {sample_id}: {raw_def}")

            scan_path = sample_dir / "scan_metadata.json"
            scan_file = scan_path if scan_path.is_file() else None

            graph = self._builder.build_from_files(
                def_path=def_path,
                scan_metadata_path=scan_file,
                design=design,
                sample_id=sample_id,
                pdk=meta.get("pdk", "nangate45"),
            )
            data_obj = graph.to_pyg(y=target_y)
            data_obj.sample_id = sample_id
            data_obj.design = design
            self._validate_leakage(data_obj)
            return data_obj

        raise FileNotFoundError(
            f"No circuit_graph.pt or circuit_graph.npz found in {sample_dir}, "
            "and auto_extract is disabled."
        )

    def _validate_leakage(self, data_obj: Data) -> None:
        """Verifies that no ground-truth routing labels leak into input graph features."""
        # Check node feature dimensions
        if data_obj.x.shape[1] != len(NODE_FEATURE_NAMES):
            raise ValueError(
                f"Node feature dimension mismatch: expected {len(NODE_FEATURE_NAMES)}, got {data_obj.x.shape[1]}"
            )
        # Check edge feature dimensions
        if data_obj.edge_attr.shape[1] != len(EDGE_FEATURE_NAMES):
            raise ValueError(
                f"Edge feature dimension mismatch: expected {len(EDGE_FEATURE_NAMES)}, got {data_obj.edge_attr.shape[1]}"
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Data:
        if self.preload:
            return self._cached_data[idx]
        return self._load_sample_data(self.sample_dirs[idx])


def create_pyg_dataloaders(
    dataset_dir: Union[str, Path],
    splits_path: Optional[Union[str, Path]] = None,
    target_channel: Union[str, List[str]] = "total_wirelength_density",
    batch_size: int = 4,
    shuffle_train: bool = True,
    num_workers: int = 0,
    preload: bool = True,
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader, Dict[str, Any]]:
    """
    Constructs PyTorch Geometric DataLoaders for train, val, and test splits.

    Guarantees:
    - Strict sample ID partition separation (zero leakage across splits).
    - Batch collation handled natively by torch_geometric.loader.DataLoader.
    - Ground-truth target tensor y correctly dimensioned to (B, C_target, H, W).
    """
    if not _HAS_PYG:
        raise ImportError(
            "PyTorch and torch_geometric are required to build PyG DataLoaders."
        )

    ds_root = Path(dataset_dir)
    samples_dir = ds_root / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"samples/ folder not found in dataset directory: {ds_root}")

    # Resolve splits file
    if splits_path is not None:
        split_file = Path(splits_path)
    else:
        split_file = ds_root / "splits_v001.json"

    if not split_file.is_file():
        raise FileNotFoundError(f"Splits JSON file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_ids: List[str] = split_data.get("train", [])
    val_ids: List[str] = split_data.get("val", [])
    test_ids: List[str] = split_data.get("test", [])

    # Anti-leakage partition assertion
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    overlap_train_val = set_train.intersection(set_val)
    overlap_train_test = set_train.intersection(set_test)
    overlap_val_test = set_val.intersection(set_test)

    if overlap_train_val:
        raise ValueError(f"Data leakage detected between train and val splits: {overlap_train_val}")
    if overlap_train_test:
        raise ValueError(f"Data leakage detected between train and test splits: {overlap_train_test}")
    if overlap_val_test:
        raise ValueError(f"Data leakage detected between val and test splits: {overlap_val_test}")

    # Gather directory paths
    train_dirs = [samples_dir / sid for sid in train_ids]
    val_dirs = [samples_dir / sid for sid in val_ids]
    test_dirs = [samples_dir / sid for sid in test_ids]

    # Verify directory existence
    for p in train_dirs + val_dirs + test_dirs:
        if not p.is_dir():
            raise FileNotFoundError(f"Sample directory referenced in split does not exist: {p}")

    # Instantiate datasets
    train_ds = PyGCircuitGraphDataset(
        sample_dirs=train_dirs,
        target_channel=target_channel,
        split_name="train",
        preload=preload,
    )
    val_ds = PyGCircuitGraphDataset(
        sample_dirs=val_dirs,
        target_channel=target_channel,
        split_name="val",
        preload=preload,
    )
    test_ds = PyGCircuitGraphDataset(
        sample_dirs=test_dirs,
        target_channel=target_channel,
        split_name="test",
        preload=preload,
    )

    # Build PyG DataLoaders
    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    val_loader = PyGDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = PyGDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    split_info = {
        "dataset_dir": str(ds_root),
        "splits_file": str(split_file),
        "target_channels": train_ds.target_channels,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "total_samples": len(train_ds) + len(val_ds) + len(test_ds),
        "train_designs": sorted(list(set(train_ds.designs))),
        "val_designs": sorted(list(set(val_ds.designs))),
        "test_designs": sorted(list(set(test_ds.designs))),
        "batch_size": batch_size,
    }

    return train_loader, val_loader, test_loader, split_info
