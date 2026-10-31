"""
Paired Dataset Representation for Physical Design ML.
Pairs Stage 2a physical layout feature tensors (7 x H x W, intended to be pre-route-compatible)
with Stage 2b post-route routing ground-truth label tensors (8 x H x W)
for unified sample loading, multi-level validation, and reporting.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


FEATURE_NPZ_NAME = "features_spatial.npz"
FEATURE_META_NAME = "metadata.json"
LABEL_NPZ_NAME = "labels_routing.npz"
LABEL_META_NAME = "routing_label_metadata.json"
SCAN_NPZ_NAME = "features_scan.npz"
SCAN_META_NAME = "scan_feature_metadata.json"


@dataclass
class PhysicalDesignSample:
    """
    Unified paired physical design sample containing:
    - Physical input features intended to be pre-route-compatible and metadata (Stage 2a)
    - Post-route routing ground-truth label tensors and metadata (Stage 2b)
    - Optional scan-aware input features and metadata (Stage 3e/3f)
    """

    sample_id: str
    sample_dir: Path
    feature_tensor: np.ndarray        # (C_feat, H, W) normalized float32
    raw_feature_tensor: np.ndarray    # (C_feat, H, W) raw counts float32
    feature_channel_names: List[str]
    label_tensor: np.ndarray          # (C_lbl, H, W) normalized float32
    raw_label_tensor: np.ndarray      # (C_lbl, H, W) raw metrics float32
    label_channel_names: List[str]
    feature_metadata: Dict[str, Any]
    label_metadata: Dict[str, Any]
    scan_feature_tensor: Optional[np.ndarray] = None        # (C_scan, H, W) normalized float32
    raw_scan_feature_tensor: Optional[np.ndarray] = None    # (C_scan, H, W) raw counts float32
    scan_feature_channel_names: List[str] = field(default_factory=list)
    scan_feature_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_dir, Path):
            self.sample_dir = Path(self.sample_dir)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Spatial resolution (H, W)."""
        return int(self.feature_tensor.shape[1]), int(self.feature_tensor.shape[2])

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        """Feature tensor shape (C_feat, H, W)."""
        return tuple(self.feature_tensor.shape)  # type: ignore

    @property
    def label_shape(self) -> Tuple[int, int, int]:
        """Label tensor shape (C_lbl, H, W)."""
        return tuple(self.label_tensor.shape)  # type: ignore

    @property
    def num_feature_channels(self) -> int:
        """Number of feature channels."""
        return int(self.feature_tensor.shape[0])

    @property
    def num_label_channels(self) -> int:
        """Number of label channels."""
        return int(self.label_tensor.shape[0])

    @property
    def source_run_dir(self) -> Optional[str]:
        """Source run directory path recorded in metadata."""
        return self.feature_metadata.get("source_run_dir") or self.label_metadata.get("source_run_dir")

    @property
    def design(self) -> Optional[str]:
        """Design name from metadata."""
        return self.feature_metadata.get("design") or self.label_metadata.get("design")

    @property
    def pdk(self) -> Optional[str]:
        """PDK identifier from metadata."""
        return self.feature_metadata.get("pdk") or self.label_metadata.get("pdk")

    @property
    def die_area_um(self) -> Dict[str, float]:
        """Die area dimensions in microns."""
        return self.feature_metadata.get("die_area_um") or self.label_metadata.get("die_area_um") or {}

    @property
    def has_scan_features(self) -> bool:
        """True if scan-aware feature tensor is loaded and present."""
        return self.scan_feature_tensor is not None

    @property
    def scan_feature_shape(self) -> Optional[Tuple[int, int, int]]:
        """Scan feature tensor shape (C_scan, H, W) or None."""
        if self.scan_feature_tensor is not None:
            return tuple(self.scan_feature_tensor.shape)  # type: ignore
        return None

    @property
    def num_scan_channels(self) -> int:
        """Number of scan feature channels (0 if absent)."""
        if self.scan_feature_tensor is not None:
            return int(self.scan_feature_tensor.shape[0])
        return 0

    @property
    def scan_source(self) -> Optional[str]:
        """Scan source ('openroad_dft' or 'synthetic_proxy') from scan metadata."""
        if self.scan_feature_metadata:
            return self.scan_feature_metadata.get("scan_source")
        return None

    def validate(self) -> None:
        """
        Validates internal consistency of the paired sample:
        1. Feature tensor dimensionality (C x H x W).
        2. Label tensor dimensionality (C x H x W).
        3. Raw tensors match normalized shapes.
        4. Spatial grid alignment (feature H, W == label H, W).
        5. Finite values across all tensors (no NaN or Inf).
        6. Channel count matching channel names.
        7. Metadata grid size / tensor shape consistency where provided.
        8. Source run directory and design consistency across metadatas.

        Raises:
            ValueError: If any validation check fails.
        """
        # 1. Feature tensor dimensionality
        if self.feature_tensor.ndim != 3:
            raise ValueError(
                f"Feature tensor must be 3-dimensional (C, H, W), got shape {self.feature_tensor.shape}"
            )
        if self.raw_feature_tensor.ndim != 3:
            raise ValueError(
                f"Raw feature tensor must be 3-dimensional (C, H, W), got shape {self.raw_feature_tensor.shape}"
            )
        if self.feature_tensor.shape != self.raw_feature_tensor.shape:
            raise ValueError(
                f"Feature tensor shape {self.feature_tensor.shape} does not match raw feature tensor shape {self.raw_feature_tensor.shape}"
            )

        # 2. Label tensor dimensionality
        if self.label_tensor.ndim != 3:
            raise ValueError(
                f"Label tensor must be 3-dimensional (C, H, W), got shape {self.label_tensor.shape}"
            )
        if self.raw_label_tensor.ndim != 3:
            raise ValueError(
                f"Raw label tensor must be 3-dimensional (C, H, W), got shape {self.raw_label_tensor.shape}"
            )
        if self.label_tensor.shape != self.raw_label_tensor.shape:
            raise ValueError(
                f"Label tensor shape {self.label_tensor.shape} does not match raw label tensor shape {self.raw_label_tensor.shape}"
            )

        # 3. Spatial grid alignment (H, W)
        feat_h, feat_w = self.feature_tensor.shape[1], self.feature_tensor.shape[2]
        lbl_h, lbl_w = self.label_tensor.shape[1], self.label_tensor.shape[2]
        if (feat_h, feat_w) != (lbl_h, lbl_w):
            raise ValueError(
                f"Grid resolution mismatch: feature grid is ({feat_h}, {feat_w}) "
                f"but label grid is ({lbl_h}, {lbl_w})"
            )

        # 4. Finite values verification
        if not np.all(np.isfinite(self.feature_tensor)):
            raise ValueError("Feature tensor contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.raw_feature_tensor)):
            raise ValueError("Raw feature tensor contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.label_tensor)):
            raise ValueError("Label tensor contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.raw_label_tensor)):
            raise ValueError("Raw label tensor contains non-finite values (NaN or Inf)")

        # 5. Channel count vs channel names matching
        if len(self.feature_channel_names) != self.feature_tensor.shape[0]:
            raise ValueError(
                f"Feature channel names count ({len(self.feature_channel_names)}) does not match "
                f"feature tensor channel dimension ({self.feature_tensor.shape[0]})"
            )
        if len(self.label_channel_names) != self.label_tensor.shape[0]:
            raise ValueError(
                f"Label channel names count ({len(self.label_channel_names)}) does not match "
                f"label tensor channel dimension ({self.label_tensor.shape[0]})"
            )

        # 6. Metadata grid size / shape consistency
        if "grid_size" in self.feature_metadata:
            meta_grid = tuple(self.feature_metadata["grid_size"])
            if meta_grid != (feat_h, feat_w):
                raise ValueError(
                    f"Feature metadata grid_size {meta_grid} differs from tensor grid ({feat_h}, {feat_w})"
                )
        if "grid_size" in self.label_metadata:
            meta_grid = tuple(self.label_metadata["grid_size"])
            if meta_grid != (lbl_h, lbl_w):
                raise ValueError(
                    f"Label metadata grid_size {meta_grid} differs from tensor grid ({lbl_h}, {lbl_w})"
                )

        if "tensor_shape" in self.feature_metadata:
            meta_shape = tuple(self.feature_metadata["tensor_shape"])
            if meta_shape != self.feature_tensor.shape:
                raise ValueError(
                    f"Feature metadata tensor_shape {meta_shape} differs from tensor shape {self.feature_tensor.shape}"
                )
        if "tensor_shape" in self.label_metadata:
            meta_shape = tuple(self.label_metadata["tensor_shape"])
            if meta_shape != self.label_tensor.shape:
                raise ValueError(
                    f"Label metadata tensor_shape {meta_shape} differs from tensor shape {self.label_tensor.shape}"
                )

        # 7. Source run directory and design consistency
        feat_run = self.feature_metadata.get("source_run_dir")
        lbl_run = self.label_metadata.get("source_run_dir")
        if feat_run and lbl_run:
            f_stem = Path(str(feat_run).rstrip("/\\").replace("\\", "/")).name
            l_stem = Path(str(lbl_run).rstrip("/\\").replace("\\", "/")).name
            if f_stem != l_stem:
                raise ValueError(
                    f"Source run directory mismatch: feature run '{feat_run}' differs from label run '{lbl_run}'"
                )
        elif not feat_run and not lbl_run:
            warnings.warn(
                "Optional metadata field 'source_run_dir' is missing from both feature and label metadata.",
                UserWarning,
                stacklevel=2,
            )
        elif not feat_run:
            warnings.warn(
                "Optional metadata field 'source_run_dir' is missing from feature metadata.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Optional metadata field 'source_run_dir' is missing from label metadata.",
                UserWarning,
                stacklevel=2,
            )

        feat_design = self.feature_metadata.get("design")
        lbl_design = self.label_metadata.get("design")
        if feat_design and lbl_design:
            if feat_design != lbl_design:
                raise ValueError(
                    f"Design name mismatch: feature metadata design '{feat_design}' != label metadata design '{lbl_design}'"
                )
        elif not feat_design and not lbl_design:
            warnings.warn(
                "Optional metadata field 'design' is missing from both feature and label metadata.",
                UserWarning,
                stacklevel=2,
            )
        elif not feat_design:
            warnings.warn(
                "Optional metadata field 'design' is missing from feature metadata.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Optional metadata field 'design' is missing from label metadata.",
                UserWarning,
                stacklevel=2,
            )

        # 8. Scan feature validation (if scan features present)
        self.validate_scan_features()

    def validate_scan_features(self) -> None:
        """
        Validates scan-aware feature tensor consistency if present:
        1. Scan feature tensor dimensionality (C x H x W).
        2. Raw scan feature tensor matches normalized shape.
        3. Spatial grid alignment (scan H, W == physical H, W).
        4. Finite values across all scan tensors.
        5. Scan channel count matches scan channel names.
        6. Metadata grid size and tensor shape consistency where present.

        Raises:
            ValueError: If scan features are present but inconsistent or malformed.
        """
        if not self.has_scan_features:
            if self.raw_scan_feature_tensor is not None:
                raise ValueError("Raw scan feature tensor is set but scan feature tensor is None")
            if self.scan_feature_channel_names:
                raise ValueError("Scan feature channel names provided but scan feature tensor is None")
            return

        assert self.scan_feature_tensor is not None

        # 1. Dimensionality
        if self.scan_feature_tensor.ndim != 3:
            raise ValueError(
                f"Scan feature tensor must be 3-dimensional (C, H, W), got shape {self.scan_feature_tensor.shape}"
            )
        if self.raw_scan_feature_tensor is None:
            raise ValueError("Raw scan feature tensor is required when scan feature tensor is present")
        if self.raw_scan_feature_tensor.ndim != 3:
            raise ValueError(
                f"Raw scan feature tensor must be 3-dimensional (C, H, W), got shape {self.raw_scan_feature_tensor.shape}"
            )
        if self.scan_feature_tensor.shape != self.raw_scan_feature_tensor.shape:
            raise ValueError(
                f"Scan feature tensor shape {self.scan_feature_tensor.shape} does not match raw scan feature tensor shape {self.raw_scan_feature_tensor.shape}"
            )

        # 2. Spatial grid alignment
        scan_h, scan_w = self.scan_feature_tensor.shape[1], self.scan_feature_tensor.shape[2]
        feat_h, feat_w = self.grid_shape
        if (scan_h, scan_w) != (feat_h, feat_w):
            raise ValueError(
                f"Grid resolution mismatch: scan feature grid is ({scan_h}, {scan_w}) "
                f"but physical feature grid is ({feat_h}, {feat_w})"
            )

        # 3. Finite values
        if not np.all(np.isfinite(self.scan_feature_tensor)):
            raise ValueError("Scan feature tensor contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.raw_scan_feature_tensor)):
            raise ValueError("Raw scan feature tensor contains non-finite values (NaN or Inf)")

        # 4. Channel count vs names
        if len(self.scan_feature_channel_names) != self.scan_feature_tensor.shape[0]:
            raise ValueError(
                f"Scan channel names count ({len(self.scan_feature_channel_names)}) does not match "
                f"scan feature tensor channel dimension ({self.scan_feature_tensor.shape[0]})"
            )

        # 5. Metadata consistency
        if self.scan_feature_metadata:
            if "grid_size" in self.scan_feature_metadata:
                meta_grid = tuple(self.scan_feature_metadata["grid_size"])
                if meta_grid != (scan_h, scan_w):
                    raise ValueError(
                        f"Scan metadata grid_size {meta_grid} differs from scan tensor grid ({scan_h}, {scan_w})"
                    )
            if "tensor_shape" in self.scan_feature_metadata:
                meta_shape = tuple(self.scan_feature_metadata["tensor_shape"])
                if meta_shape != self.scan_feature_tensor.shape:
                    raise ValueError(
                        f"Scan metadata tensor_shape {meta_shape} differs from scan tensor shape {self.scan_feature_tensor.shape}"
                    )

    def get_fused_input_tensor(self, include_scan: bool = True) -> np.ndarray:
        """
        Returns the fused input feature tensor for model consumption.

        If include_scan is True and scan features are present, concatenates
        physical layout features and scan-aware features along the channel axis (axis 0),
        yielding shape (C_feat + C_scan, H, W), e.g. (12, 64, 64).
        Otherwise, returns a copy of physical layout feature tensor of shape (C_feat, H, W), e.g. (7, 64, 64).

        Args:
            include_scan: Whether to fuse scan features if available.

        Returns:
            np.ndarray: float32 fused feature tensor.

        Raises:
            ValueError: If scan features are present but malformed.
        """
        if include_scan and self.has_scan_features:
            self.validate_scan_features()
            assert self.scan_feature_tensor is not None
            return np.concatenate([self.feature_tensor, self.scan_feature_tensor], axis=0).astype(np.float32)
        return self.feature_tensor.copy().astype(np.float32)

    def get_fused_channel_names(self, include_scan: bool = True) -> List[str]:
        """
        Returns prefixed channel names corresponding to the fused input tensor.
        Physical features are prefixed with 'physical/', and scan features with 'scan/'.

        Args:
            include_scan: Whether to include scan feature channel names.

        Returns:
            List[str]: List of prefixed channel names.
        """
        names = [f"physical/{name}" for name in self.feature_channel_names]
        if include_scan and self.has_scan_features:
            names.extend([f"scan/{name}" for name in self.scan_feature_channel_names])
        return names

    @classmethod
    def load(
        cls,
        sample_dir: Union[str, Path],
        sample_id: Optional[str] = None,
        validate: bool = True,
        load_scan: bool = True,
    ) -> PhysicalDesignSample:
        """
        Loads a paired physical design sample from disk.

        Args:
            sample_dir: Directory containing features_spatial.npz, metadata.json,
                        labels_routing.npz, and routing_label_metadata.json.
            sample_id: Optional identifier. Defaults to directory name.
            validate: Whether to run validate() upon loading (default True).
            load_scan: Whether to load optional scan features if present (default True).

        Returns:
            PhysicalDesignSample instance.

        Raises:
            FileNotFoundError: If sample directory or required artifact files do not exist.
            ValueError: If tensor parsing or validation checks fail.
        """
        path = Path(sample_dir).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Sample directory does not exist: {path}")

        # Locate required artifact files
        feat_npz_path = path / FEATURE_NPZ_NAME
        feat_meta_path = path / FEATURE_META_NAME
        lbl_npz_path = path / LABEL_NPZ_NAME
        lbl_meta_path = path / LABEL_META_NAME

        missing = []
        for p in [feat_npz_path, feat_meta_path, lbl_npz_path, lbl_meta_path]:
            if not p.is_file():
                missing.append(p.name)

        if missing:
            raise FileNotFoundError(
                f"Missing required sample artifact(s) in {path}: {', '.join(missing)}"
            )

        # 1. Load feature NPZ
        try:
            with np.load(feat_npz_path, allow_pickle=False) as feat_npz:
                if "features" not in feat_npz:
                    raise ValueError(f"'{FEATURE_NPZ_NAME}' missing 'features' array")
                feature_tensor = feat_npz["features"].astype(np.float32)

                if "raw_features" in feat_npz:
                    raw_feature_tensor = feat_npz["raw_features"].astype(np.float32)
                else:
                    raw_feature_tensor = feature_tensor.copy()

                if "channel_names" in feat_npz:
                    feat_channels = [str(c) for c in feat_npz["channel_names"]]
                else:
                    feat_channels = []
        except Exception as err:
            if isinstance(err, (ValueError, FileNotFoundError)):
                raise
            raise ValueError(f"Failed to read '{feat_npz_path}': {err}") from err

        # 2. Load feature metadata
        try:
            feature_metadata = json.loads(feat_meta_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Failed to parse '{feat_meta_path}': {err}") from err

        if not feat_channels and "channel_names" in feature_metadata:
            feat_channels = [str(c) for c in feature_metadata["channel_names"]]

        # 3. Load label NPZ
        try:
            with np.load(lbl_npz_path, allow_pickle=False) as lbl_npz:
                if "labels" not in lbl_npz:
                    raise ValueError(f"'{LABEL_NPZ_NAME}' missing 'labels' array")
                label_tensor = lbl_npz["labels"].astype(np.float32)

                if "raw_labels" in lbl_npz:
                    raw_label_tensor = lbl_npz["raw_labels"].astype(np.float32)
                else:
                    raw_label_tensor = label_tensor.copy()

                if "channel_names" in lbl_npz:
                    lbl_channels = [str(c) for c in lbl_npz["channel_names"]]
                else:
                    lbl_channels = []
        except Exception as err:
            if isinstance(err, (ValueError, FileNotFoundError)):
                raise
            raise ValueError(f"Failed to read '{lbl_npz_path}': {err}") from err

        # 4. Load label metadata
        try:
            label_metadata = json.loads(lbl_meta_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Failed to parse '{lbl_meta_path}': {err}") from err

        if not lbl_channels and "channel_names" in label_metadata:
            lbl_channels = [str(c) for c in label_metadata["channel_names"]]

        # 5. Load optional scan features if present and requested
        scan_feature_tensor: Optional[np.ndarray] = None
        raw_scan_feature_tensor: Optional[np.ndarray] = None
        scan_channels: List[str] = []
        scan_feature_metadata: Optional[Dict[str, Any]] = None

        scan_npz_path = path / SCAN_NPZ_NAME
        scan_meta_path = path / SCAN_META_NAME

        if load_scan and (scan_npz_path.is_file() or scan_meta_path.is_file()):
            if not scan_npz_path.is_file():
                raise FileNotFoundError(
                    f"Found scan metadata '{SCAN_META_NAME}' but missing scan feature tensor '{SCAN_NPZ_NAME}' in {path}"
                )
            if not scan_meta_path.is_file():
                raise FileNotFoundError(
                    f"Found scan feature tensor '{SCAN_NPZ_NAME}' but missing scan metadata '{SCAN_META_NAME}' in {path}"
                )

            try:
                with np.load(scan_npz_path, allow_pickle=False) as s_npz:
                    if "scan_features" in s_npz:
                        scan_feature_tensor = s_npz["scan_features"].astype(np.float32)
                    elif "features" in s_npz:
                        scan_feature_tensor = s_npz["features"].astype(np.float32)
                    else:
                        raise ValueError(f"'{SCAN_NPZ_NAME}' missing 'scan_features' array")

                    if "raw_scan_features" in s_npz:
                        raw_scan_feature_tensor = s_npz["raw_scan_features"].astype(np.float32)
                    elif "raw_features" in s_npz:
                        raw_scan_feature_tensor = s_npz["raw_features"].astype(np.float32)
                    else:
                        raw_scan_feature_tensor = scan_feature_tensor.copy()

                    if "channel_names" in s_npz:
                        scan_channels = [str(c) for c in s_npz["channel_names"]]
            except Exception as err:
                if isinstance(err, (ValueError, FileNotFoundError)):
                    raise
                raise ValueError(f"Failed to read '{scan_npz_path}': {err}") from err

            try:
                scan_feature_metadata = json.loads(scan_meta_path.read_text(encoding="utf-8"))
            except Exception as err:
                raise ValueError(f"Failed to parse '{scan_meta_path}': {err}") from err

            if not scan_channels and "channel_names" in scan_feature_metadata:
                scan_channels = [str(c) for c in scan_feature_metadata["channel_names"]]

        sid = sample_id or path.name

        sample = cls(
            sample_id=sid,
            sample_dir=path,
            feature_tensor=feature_tensor,
            raw_feature_tensor=raw_feature_tensor,
            feature_channel_names=feat_channels,
            label_tensor=label_tensor,
            raw_label_tensor=raw_label_tensor,
            label_channel_names=lbl_channels,
            feature_metadata=feature_metadata,
            label_metadata=label_metadata,
            scan_feature_tensor=scan_feature_tensor,
            raw_scan_feature_tensor=raw_scan_feature_tensor,
            scan_feature_channel_names=scan_channels,
            scan_feature_metadata=scan_feature_metadata,
        )

        if validate:
            sample.validate()

        return sample

    def summary(self) -> Dict[str, Any]:
        """
        Returns structured dictionary summary of the paired design sample.
        """
        return {
            "sample_id": self.sample_id,
            "sample_dir": str(self.sample_dir),
            "design": self.design,
            "pdk": self.pdk,
            "source_run_dir": self.source_run_dir,
            "grid_shape": list(self.grid_shape),
            "feature_tensor_shape": list(self.feature_shape),
            "label_tensor_shape": list(self.label_shape),
            "feature_channel_count": self.num_feature_channels,
            "label_channel_count": self.num_label_channels,
            "feature_channels": list(self.feature_channel_names),
            "label_channels": list(self.label_channel_names),
            "die_area_um": self.die_area_um,
            "design_statistics": self.feature_metadata.get("design_statistics", {}),
            "routing_statistics": self.label_metadata.get("routing_statistics", {}),
            "has_scan_features": self.has_scan_features,
            "scan_feature_tensor_shape": list(self.scan_feature_shape) if self.has_scan_features else None,
            "scan_channel_count": self.num_scan_channels,
            "scan_channels": list(self.scan_feature_channel_names),
            "scan_source": self.scan_source,
            "fused_tensor_shape": list(self.get_fused_input_tensor().shape),
            "fused_channel_count": len(self.get_fused_channel_names()),
        }

    def format_summary(self) -> str:
        """
        Returns formatted human-readable summary text.
        """
        s = self.summary()
        die = s.get("die_area_um", {})
        d_stats = s.get("design_statistics", {})
        r_stats = s.get("routing_statistics", {})

        lines = [
            f"Physical Design Sample: {s['sample_id']}",
            f"  Directory        : {s['sample_dir']}",
            f"  Design / PDK     : {s.get('design', 'unknown')} / {s.get('pdk', 'unknown')}",
            f"  Source Run       : {s.get('source_run_dir', 'N/A')}",
            f"  Spatial Grid     : {s['grid_shape'][0]} x {s['grid_shape'][1]}",
            f"  Die Dimensions   : {die.get('width', 0.0):.1f} x {die.get('height', 0.0):.1f} µm",
            f"  Feature Tensor   : {tuple(s['feature_tensor_shape'])} ({s['feature_channel_count']} channels)",
            f"  Label Tensor     : {tuple(s['label_tensor_shape'])} ({s['label_channel_count']} channels)",
        ]
        if self.has_scan_features:
            lines.extend([
                f"  Scan Tensor      : {tuple(s['scan_feature_tensor_shape'])} ({s['scan_channel_count']} channels, source: {s['scan_source'] or 'unknown'})",
                f"  Fused In Tensor  : {tuple(s['fused_tensor_shape'])} ({s['fused_channel_count']} channels)",
            ])
        else:
            lines.append("  Scan Features    : Not Present (Physical-only baseline)")

        lines.extend([
            f"  Total Components : {d_stats.get('total_components', 'N/A')} (Seq: {d_stats.get('sequential_components', 'N/A')}, Comb: {d_stats.get('combinational_components', 'N/A')})",
            f"  Total Nets       : {d_stats.get('total_nets', 'N/A')} (Routed: {r_stats.get('routed_nets', 'N/A')})",
            f"  Total Wirelength : {r_stats.get('total_wirelength_um', 0.0):.2f} µm",
            f"  Total Vias       : {r_stats.get('total_vias', 'N/A')}",
        ])
        return "\n".join(lines)
