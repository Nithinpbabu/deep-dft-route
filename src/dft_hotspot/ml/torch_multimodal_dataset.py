"""
PyTorch Multimodal Dataset and DataLoader for Paired 2D Layout & Circuit Graph Representations.
Stage 7b of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- MultimodalPhysicalDesignDataset: PyTorch Dataset yielding paired:
    1. 2D Spatial Layout Tensors: (C_spatial, H, W) = (12, 64, 64) or (7, 64, 64)
    2. Topological Circuit Graphs: PyG CircuitGraphData (x, edge_index, edge_attr, grid_coords, pos)
    3. Dense Ground-Truth Routing Targets: (C_target, H, W)
- collate_multimodal: PyTorch collation function assembling synchronized batches:
    (x_spatial: (B, C, H, W), graph: PyG Batch, y: (B, C_target, H, W)).
- create_multimodal_dataloaders: Factory function creating Train, Val, and Test DataLoaders
  strictly adhering to design-held-out splits with automated leakage verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch_geometric.data import Batch, Data
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore

from dft_hotspot.graphs.dataset import ALL_ROUTING_CHANNELS
from dft_hotspot.graphs.schema import CircuitGraph, CircuitGraphData


def collate_multimodal(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collation function assembling synchronized multimodal batches.

    Args:
        batch_items: List of dictionary records produced by MultimodalPhysicalDesignDataset.

    Returns:
        Dictionary containing:
        - 'x_spatial': (B, C_spatial, H, W) FloatTensor or None
        - 'graph': PyG Batch object or None
        - 'y': (B, C_target, H, W) FloatTensor
        - 'sample_ids': List[str] of sample identifiers
        - 'designs': List[str] of design names
    """
    if not batch_items:
        return {}

    first = batch_items[0]

    # Collate spatial tensors
    if first["x_spatial"] is not None:
        x_spatial = torch.stack([item["x_spatial"] for item in batch_items], dim=0)
    else:
        x_spatial = None

    # Collate target tensors
    y = torch.stack([item["y"] for item in batch_items], dim=0)

    # Collate graph objects
    if first["graph"] is not None:
        graph_list = [item["graph"] for item in batch_items]
        graph_batch = Batch.from_data_list(graph_list)
    else:
        graph_batch = None

    sample_ids = [item["sample_id"] for item in batch_items]
    designs = [item["design"] for item in batch_items]

    return {
        "x_spatial": x_spatial,
        "graph": graph_batch,
        "y": y,
        "sample_ids": sample_ids,
        "designs": designs,
    }


class MultimodalPhysicalDesignDataset(Dataset):
    """
    Paired multimodal dataset delivering synchronized 2D continuous layout grids,
    circuit netlist/scan graphs, and ground-truth routing target maps.
    """

    def __init__(
        self,
        sample_dirs: List[Union[str, Path]],
        target_channel: Union[str, List[str]] = "total_wirelength_density",
        include_scan: bool = True,
        split_name: str = "unassigned",
        preload: bool = True,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch and torch_geometric are required for MultimodalPhysicalDesignDataset."
            )

        if not sample_dirs:
            raise ValueError(
                "Cannot initialize MultimodalPhysicalDesignDataset with empty sample_dirs list."
            )

        self.sample_dirs = [Path(p) for p in sample_dirs]
        self.include_scan = include_scan
        self.split_name = split_name
        self.preload = preload

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
                    f"Unknown target channel '{ch}'. Valid options: {ALL_ROUTING_CHANNELS}"
                )

        self.sample_ids: List[str] = []
        self.designs: List[str] = []
        self._cached_items: List[Dict[str, Any]] = []

        # Validate and preload
        for sdir in self.sample_dirs:
            item = self._load_sample(sdir)
            self.sample_ids.append(item["sample_id"])
            self.designs.append(item["design"])
            if self.preload:
                self._cached_items.append(item)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.preload:
            return self._cached_items[idx]
        return self._load_sample(self.sample_dirs[idx])

    def _load_sample(self, sample_dir: Path) -> Dict[str, Any]:
        """Loads and validates a single sample's spatial layout, graph, and routing targets."""
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        meta_file = sample_dir / "metadata.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"metadata.json not found in {sample_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        design = str(meta.get("design", sample_dir.name))
        sample_id = sample_dir.name

        # -------------------------------------------------------------
        # 1. Load Ground-Truth Target Routing Tensor
        # -------------------------------------------------------------
        lbl_file = sample_dir / "labels_routing.npz"
        if not lbl_file.is_file():
            raise FileNotFoundError(f"labels_routing.npz not found in {sample_dir}")

        lbl_data = np.load(lbl_file)
        full_labels = lbl_data["labels"]  # (8, 64, 64)

        target_indices = [ALL_ROUTING_CHANNELS.index(ch) for ch in self.target_channels]
        target_np = full_labels[target_indices].copy().astype(np.float32)

        if not np.all(np.isfinite(target_np)):
            raise ValueError(f"Target routing tensor in {sample_id} contains NaN or Inf.")

        target_y = torch.from_numpy(target_np)

        # -------------------------------------------------------------
        # 2. Load 2D Spatial Layout Features
        # -------------------------------------------------------------
        if self.include_scan:
            # 12-channel fused layout + scan features
            fused_file = sample_dir / "fused_input.npz"
            if fused_file.is_file():
                fused_data = np.load(fused_file)
                spatial_np = fused_data["fused_features"].copy().astype(np.float32)
            else:
                # Assemble from spatial + scan components if fused file missing
                spat_file = sample_dir / "features_spatial.npz"
                scan_file = sample_dir / "features_scan.npz"
                spat_np = np.load(spat_file)["features"].astype(np.float32)
                scan_np = np.load(scan_file)["features"].astype(np.float32)
                spatial_np = np.concatenate([spat_np, scan_np], axis=0)
        else:
            # 7-channel physical-only features (for ablation)
            spat_file = sample_dir / "features_spatial.npz"
            if not spat_file.is_file():
                raise FileNotFoundError(f"features_spatial.npz not found in {sample_dir}")
            spatial_np = np.load(spat_file)["features"].copy().astype(np.float32)

        if not np.all(np.isfinite(spatial_np)):
            raise ValueError(f"Spatial layout features in {sample_id} contain NaN or Inf.")

        x_spatial = torch.from_numpy(spatial_np)

        # -------------------------------------------------------------
        # 3. Load Circuit Graph Representation
        # -------------------------------------------------------------
        pt_path = sample_dir / "circuit_graph.pt"
        if pt_path.is_file():
            loaded_obj = torch.load(pt_path, weights_only=False)
            data_dict = loaded_obj.to_dict()
            data_dict["y"] = target_y
            data_dict["sample_id"] = sample_id
            data_dict["design"] = design
            graph_obj = CircuitGraphData(**data_dict)
        else:
            npz_path = sample_dir / "circuit_graph.npz"
            if npz_path.is_file():
                cg = CircuitGraph.load_npz(npz_path)
                graph_obj = cg.to_pyg(y=target_y)
                graph_obj.sample_id = sample_id
                graph_obj.design = design
            else:
                raise FileNotFoundError(
                    f"Neither circuit_graph.pt nor circuit_graph.npz found in {sample_dir}"
                )

        # -------------------------------------------------------------
        # 4. Anti-Leakage Verification
        # -------------------------------------------------------------
        self._validate_leakage(graph_obj, sample_id)

        return {
            "x_spatial": x_spatial,
            "graph": graph_obj,
            "y": target_y,
            "sample_id": sample_id,
            "design": design,
        }

    def _validate_leakage(self, data_obj: CircuitGraphData, sample_id: str) -> None:
        """Guarantees graph node features and spatial inputs contain no target leakages."""
        if hasattr(data_obj, "x") and data_obj.x is not None:
            if not torch.isfinite(data_obj.x).all():
                raise ValueError(f"Graph node features in {sample_id} contain NaN or Inf.")
            if data_obj.x.shape[-1] != 10:
                raise ValueError(
                    f"Expected 10 node features, got {data_obj.x.shape[-1]} in {sample_id}."
                )


def create_multimodal_dataloaders(
    dataset_dir: Union[str, Path],
    splits_path: Optional[Union[str, Path]] = None,
    target_channel: Union[str, List[str]] = "total_wirelength_density",
    include_scan: bool = True,
    batch_size: int = 4,
    shuffle_train: bool = True,
    num_workers: int = 0,
    preload: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Builds synchronized Train, Validation, and Test DataLoaders adhering to
    strict design-held-out splits.

    Args:
        dataset_dir: Path to root dataset directory (e.g. data/processed/datasets/expanded_v002).
        splits_path: Path to splits_v001.json. If None, resolves dataset_dir/splits_v001.json.
        target_channel: Single channel name or 'all'.
        include_scan: True for 12-channel fused spatial inputs, False for 7-channel physical-only.
        batch_size: Batch size for training and evaluation.
        shuffle_train: Whether to shuffle the training set.
        num_workers: Number of DataLoader worker subprocesses.
        preload: Preload all sample tensors and graphs into memory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, split_info)
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch and torch_geometric are required for create_multimodal_dataloaders."
        )

    ds_root = Path(dataset_dir)
    samples_dir = ds_root / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"samples/ folder not found in dataset directory: {ds_root}")

    split_file = Path(splits_path) if splits_path is not None else ds_root / "splits_v001.json"
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

    if set_train.intersection(set_val):
        raise ValueError("Data leakage detected between train and val splits.")
    if set_train.intersection(set_test):
        raise ValueError("Data leakage detected between train and test splits.")
    if set_val.intersection(set_test):
        raise ValueError("Data leakage detected between val and test splits.")

    train_dirs = [samples_dir / sid for sid in train_ids]
    val_dirs = [samples_dir / sid for sid in val_ids]
    test_dirs = [samples_dir / sid for sid in test_ids]

    for p in train_dirs + val_dirs + test_dirs:
        if not p.is_dir():
            raise FileNotFoundError(f"Sample directory in split does not exist: {p}")

    train_ds = MultimodalPhysicalDesignDataset(
        sample_dirs=train_dirs,
        target_channel=target_channel,
        include_scan=include_scan,
        split_name="train",
        preload=preload,
    )
    val_ds = MultimodalPhysicalDesignDataset(
        sample_dirs=val_dirs,
        target_channel=target_channel,
        include_scan=include_scan,
        split_name="val",
        preload=preload,
    )
    test_ds = MultimodalPhysicalDesignDataset(
        sample_dirs=test_dirs,
        target_channel=target_channel,
        include_scan=include_scan,
        split_name="test",
        preload=preload,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
    )

    split_info = {
        "dataset_dir": str(ds_root),
        "splits_file": str(split_file),
        "target_channels": train_ds.target_channels,
        "include_scan": include_scan,
        "spatial_in_channels": 12 if include_scan else 7,
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
