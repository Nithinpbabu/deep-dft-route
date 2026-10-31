"""
Circuit and Scan Graph Data Schema (Stage 6a).

Defines the core data representation for circuit netlist and DFT scan-chain graphs,
providing serialization (NPZ / PT) and conversion to PyTorch Geometric Data objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

NODE_FEATURE_NAMES = [
    "norm_x",
    "norm_y",
    "is_sequential",
    "is_combinational",
    "is_scan_cell",
    "scan_chain_id",
    "scan_order_index",
    "pin_degree",
    "relative_cell_area",
    "fanout_degree",
]

EDGE_FEATURE_NAMES = [
    "edge_type",
    "is_scan_edge",
    "normalized_distance",
    "weight",
]

EDGE_TYPE_NETLIST = 0.0
EDGE_TYPE_SCAN_FORWARD = 1.0
EDGE_TYPE_SCAN_REVERSE = 2.0
EDGE_TYPE_SPATIAL = 3.0

try:
    import torch
    from torch_geometric.data import Data

    class CircuitGraphData(Data):
        """
        PyTorch Geometric Data container tailored for physical design graphs.
        Ensures 2D/3D ground-truth routing tensors (y) are stacked along dim 0
        into (B, C, H, W) instead of concatenated along dim 0.
        """

        def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
            if key == "y":
                return None  # PyG Collater will torch.stack along dim 0
            return super().__cat_dim__(key, value, *args, **kwargs)

    _HAS_PYG_SCHEMA = True
except ImportError:
    CircuitGraphData = object  # type: ignore
    _HAS_PYG_SCHEMA = False


@dataclass
class CircuitGraph:
    """
    In-memory representation of a circuit and scan graph.
    
    Contains node features, edge connectivity, edge attributes,
    physical coordinates, and metadata.
    """
    node_features: np.ndarray          # Shape: (N, D_node), dtype float32
    edge_index: np.ndarray             # Shape: (2, E), dtype int64
    edge_features: np.ndarray          # Shape: (E, D_edge), dtype float32
    node_names: List[str]              # Length: N (instance names)
    node_positions: np.ndarray         # Shape: (N, 2), dtype float32 (normalized [x, y])
    node_positions_dbu: np.ndarray     # Shape: (N, 2), dtype float32 (raw DBU [x, y])
    node_grid_coords: np.ndarray       # Shape: (N, 2), dtype int64 (grid row r, col c in [0, H-1]x[0, W-1])
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.ndim == 2 else 0

    @property
    def node_dim(self) -> int:
        return self.node_features.shape[1] if self.node_features.ndim == 2 else 0

    @property
    def edge_dim(self) -> int:
        return self.edge_features.shape[1] if self.edge_features.ndim == 2 else 0

    def to_pyg(self, y: Optional[Any] = None) -> Any:
        """
        Converts this circuit graph to a CircuitGraphData (PyG Data) object.
        Requires PyTorch and torch_geometric to be installed.
        """
        if not _HAS_PYG_SCHEMA:
            raise ImportError(
                "PyTorch and torch_geometric are required to convert CircuitGraph to PyG Data. "
                "Ensure torch and torch_geometric are installed in your virtual environment."
            )

        data_kwargs: Dict[str, Any] = {
            "x": torch.from_numpy(self.node_features).float(),
            "edge_index": torch.from_numpy(self.edge_index).long(),
            "edge_attr": torch.from_numpy(self.edge_features).float(),
            "pos": torch.from_numpy(self.node_positions).float(),
            "grid_coords": torch.from_numpy(self.node_grid_coords).long(),
            "num_nodes": self.num_nodes,
        }

        if y is not None:
            if isinstance(y, np.ndarray):
                data_kwargs["y"] = torch.from_numpy(y).float()
            elif isinstance(y, torch.Tensor):
                data_kwargs["y"] = y.float()
            else:
                data_kwargs["y"] = torch.tensor(y).float()

        # Add optional metadata attributes
        if "design" in self.metadata:
            data_kwargs["design"] = self.metadata["design"]
        if "sample_id" in self.metadata:
            data_kwargs["sample_id"] = self.metadata["sample_id"]

        return CircuitGraphData(**data_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes graph to a dictionary of standard Python / numpy arrays."""
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "node_names": self.node_names,
            "node_positions": self.node_positions,
            "node_positions_dbu": self.node_positions_dbu,
            "node_grid_coords": self.node_grid_coords,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitGraph":
        """Reconstructs CircuitGraph from a serialized dictionary."""
        return cls(
            node_features=np.asarray(data["node_features"], dtype=np.float32),
            edge_index=np.asarray(data["edge_index"], dtype=np.int64),
            edge_features=np.asarray(data["edge_features"], dtype=np.float32),
            node_names=list(data["node_names"]),
            node_positions=np.asarray(data["node_positions"], dtype=np.float32),
            node_positions_dbu=np.asarray(data["node_positions_dbu"], dtype=np.float32),
            node_grid_coords=np.asarray(data["node_grid_coords"], dtype=np.int64),
            metadata=dict(data.get("metadata", {})),
        )

    def save_npz(self, filepath: Union[str, Path]) -> Path:
        """
        Saves the graph structure and metadata to a compressed .npz archive.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(self.metadata)
        names_json = json.dumps(self.node_names)

        np.savez_compressed(
            path,
            node_features=self.node_features,
            edge_index=self.edge_index,
            edge_features=self.edge_features,
            node_positions=self.node_positions,
            node_positions_dbu=self.node_positions_dbu,
            node_grid_coords=self.node_grid_coords,
            node_names_json=np.array(names_json),
            metadata_json=np.array(meta_json),
        )
        return path

    @classmethod
    def load_npz(cls, filepath: Union[str, Path]) -> "CircuitGraph":
        """
        Loads CircuitGraph from a compressed .npz archive.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Graph archive not found: {path}")

        with np.load(path, allow_pickle=False) as npz:
            node_features = npz["node_features"]
            edge_index = npz["edge_index"]
            edge_features = npz["edge_features"]
            node_positions = npz["node_positions"]
            node_positions_dbu = npz["node_positions_dbu"]
            node_grid_coords = npz["node_grid_coords"]
            node_names = json.loads(str(npz["node_names_json"]))
            metadata = json.loads(str(npz["metadata_json"]))

        return cls(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_names=node_names,
            node_positions=node_positions,
            node_positions_dbu=node_positions_dbu,
            node_grid_coords=node_grid_coords,
            metadata=metadata,
        )

    def save_pt(self, filepath: Union[str, Path], y: Optional[Any] = None) -> Path:
        """
        Saves graph as a PyTorch Geometric CircuitGraphData object via torch.save.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is required to save .pt graph files.") from e

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        pyg_data = self.to_pyg(y=y)
        torch.save(pyg_data, path)
        return path

    @classmethod
    def load_pt(cls, filepath: Union[str, Path]) -> Any:
        """
        Loads a PyTorch Geometric Data object from a .pt file.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is required to load .pt graph files.") from e

        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"PyG graph file not found: {path}")
        return torch.load(path, weights_only=False)
