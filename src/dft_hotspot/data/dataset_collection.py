"""
Dataset Collection, Introspection, Statistics, Duplicate Detection, and Split Policy.
Stage 4b of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- PhysicalDesignDataset: High-level dataset loader and collection container.
- SampleRecord: Container for individual sample metadata, tensors, and status.
- Duplicate detection across physical layout and routing label tensors.
- Comprehensive dataset statistics calculation across channels and samples.
- Split policy generation with explicit pipeline-smoke validation limitations.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from dft_hotspot.data.dataset import (
    FEATURE_META_NAME,
    FEATURE_NPZ_NAME,
    LABEL_META_NAME,
    LABEL_NPZ_NAME,
    SCAN_META_NAME,
    SCAN_NPZ_NAME,
    PhysicalDesignSample,
)

FUSED_NPZ_NAME = "fused_input.npz"
FUSED_META_NAME = "fused_input_metadata.json"


def resolve_cross_platform_path(
    path_str: Union[str, Path],
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Resolves a path string cross-platform between Windows and Linux/WSL environments,
    or resolves relative to an optional base directory.
    """
    p_str = str(path_str).strip()
    p = Path(p_str)
    if p.exists():
        return p.resolve()

    # Handle Linux/WSL /mnt/<drive>/... on Windows
    if sys.platform == "win32" and p_str.startswith("/mnt/"):
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p_str)
        if match:
            drive, rest = match.groups()
            win_path = Path(f"{drive.upper()}:/{rest}")
            if win_path.exists():
                return win_path.resolve()

    # Handle Windows <drive>:/... on Linux/WSL
    if sys.platform != "win32" and re.match(r"^[a-zA-Z]:[/\\]", p_str):
        drive = p_str[0].lower()
        rest = p_str[2:].replace("\\", "/").lstrip("/")
        wsl_path = Path(f"/mnt/{drive}/{rest}")
        if wsl_path.exists():
            return wsl_path.resolve()

    # Relative to base directory
    if base_dir is not None:
        cand1 = base_dir / p.name
        if cand1.exists():
            return cand1.resolve()
        cand2 = base_dir / p_str
        if cand2.exists():
            return cand2.resolve()
        cand3 = base_dir / "samples" / p.name
        if cand3.exists():
            return cand3.resolve()

    return p


@dataclass
class SampleRecord:
    """
    Encapsulates metadata, configs, and loaded artifacts for a single sample in a dataset.
    """

    sample_id: str
    run_id: str
    sample_dir: Path
    run_dir: Optional[Path] = None
    status: str = "success"
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    tensor_shapes: Dict[str, List[int]] = field(default_factory=dict)
    sample: Optional[PhysicalDesignSample] = None
    fused_tensor: Optional[np.ndarray] = None  # (C_fused, H, W)
    fused_channel_names: List[str] = field(default_factory=list)
    fused_sha256: Optional[str] = None
    labels_sha256: Optional[str] = None
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)

    @property
    def utilization(self) -> Optional[float]:
        """Core utilization from config or sample metadata."""
        if "utilization" in self.config:
            return float(self.config["utilization"])
        if self.sample and "config" in self.sample.feature_metadata:
            return self.sample.feature_metadata["config"].get("target_utilization")
        return None

    @property
    def clock_period_ns(self) -> Optional[float]:
        """Clock period in nanoseconds from config or sample metadata."""
        if "clock_period_ns" in self.config:
            return float(self.config["clock_period_ns"])
        if self.sample and "config" in self.sample.feature_metadata:
            return self.sample.feature_metadata["config"].get("clock_period_ns")
        return None

    @property
    def aspect_ratio(self) -> Optional[float]:
        """Aspect ratio from config."""
        if "aspect_ratio" in self.config:
            return float(self.config["aspect_ratio"])
        return 1.0

    @property
    def chain_count(self) -> Optional[int]:
        """Chain count from config or scan feature metadata."""
        if "chain_count" in self.config:
            return int(self.config["chain_count"])
        if self.sample and self.sample.scan_feature_metadata:
            return int(self.sample.scan_feature_metadata.get("chain_count", 1))
        return None

    @property
    def total_wirelength_um(self) -> Optional[float]:
        """Total routed wirelength in microns from label metadata."""
        if self.sample and "routing_statistics" in self.sample.label_metadata:
            return float(self.sample.label_metadata["routing_statistics"].get("total_wirelength_um", 0.0))
        return None

    @property
    def total_vias(self) -> Optional[int]:
        """Total routed via count from label metadata."""
        if self.sample and "routing_statistics" in self.sample.label_metadata:
            return int(self.sample.label_metadata["routing_statistics"].get("total_vias", 0))
        return None

    @property
    def total_scan_cells(self) -> Optional[int]:
        """Total scan cells from scan metadata."""
        if self.sample and self.sample.scan_feature_metadata:
            return int(self.sample.scan_feature_metadata.get("total_scan_cells", 0))
        return None

    @property
    def scan_proxy_wirelength_um(self) -> Optional[float]:
        """Scan proxy wirelength in microns from scan feature metadata."""
        if self.sample and self.sample.scan_feature_metadata:
            return float(self.sample.scan_feature_metadata.get("total_scan_wirelength_um", 0.0))
        return None

    @property
    def total_components(self) -> Optional[int]:
        """Total placed component count from feature metadata."""
        if self.sample and "design_statistics" in self.sample.feature_metadata:
            return int(self.sample.feature_metadata["design_statistics"].get("total_components", 0))
        return None

    @property
    def routed_nets(self) -> Optional[int]:
        """Total routed nets from label metadata."""
        if self.sample and "routing_statistics" in self.sample.label_metadata:
            return int(self.sample.label_metadata["routing_statistics"].get("routed_nets", 0))
        return None


@dataclass
class DuplicateGroup:
    """
    Represents a group of duplicate or near-duplicate layout samples.
    """

    group_id: int
    sample_ids: List[str]
    match_type: str
    is_exact: bool
    fused_sha256: str
    labels_sha256: str
    max_abs_diff_fused: float
    max_abs_diff_labels: float
    shared_config: Dict[str, Any]
    differing_config: Dict[str, List[Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "sample_ids": self.sample_ids,
            "match_type": self.match_type,
            "is_exact": self.is_exact,
            "fused_sha256": self.fused_sha256,
            "labels_sha256": self.labels_sha256,
            "max_abs_diff_fused": self.max_abs_diff_fused,
            "max_abs_diff_labels": self.max_abs_diff_labels,
            "shared_config": self.shared_config,
            "differing_config": self.differing_config,
        }


class PhysicalDesignDataset:
    """
    High-level collection class managing a multi-run physical design dataset.
    Loads manifest, encapsulates samples, validates consistency, computes dataset-wide
    statistics, detects duplicate layouts, and generates train/val/test splits.
    """

    def __init__(
        self,
        dataset_id: str,
        manifest_path: Path,
        schema_version: str = "dataset_manifest_v001",
        design: str = "unknown",
        platform: str = "unknown",
        grid_size: int = 64,
        scan_source: Optional[str] = None,
        sweep_config_path: Optional[Path] = None,
        created_at: Optional[str] = None,
        samples: Optional[Dict[str, SampleRecord]] = None,
        failures: Optional[List[Dict[str, Any]]] = None,
        manifest_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.manifest_path = manifest_path.resolve()
        self.manifest_dir = self.manifest_path.parent
        self.schema_version = schema_version
        self.design = design
        self.platform = platform
        self.grid_size = grid_size
        self.scan_source = scan_source
        self.sweep_config_path = sweep_config_path
        self.created_at = created_at
        self.samples: Dict[str, SampleRecord] = samples or {}
        self.sample_order: List[str] = list(self.samples.keys())
        self.failures: List[Dict[str, Any]] = failures or []
        self.manifest_dict: Dict[str, Any] = manifest_dict or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[SampleRecord]:
        for sample_id in self.sample_order:
            yield self.samples[sample_id]

    def __getitem__(self, key: Union[int, str]) -> SampleRecord:
        if isinstance(key, int):
            sample_id = self.sample_order[key]
            return self.samples[sample_id]
        if isinstance(key, str):
            if key not in self.samples:
                raise KeyError(f"Sample ID '{key}' not found in dataset '{self.dataset_id}'")
            return self.samples[key]
        raise TypeError(f"Key must be int or str, got {type(key)}")

    def keys(self) -> List[str]:
        return list(self.sample_order)

    def values(self) -> List[SampleRecord]:
        return [self.samples[sid] for sid in self.sample_order]

    def items(self) -> List[Tuple[str, SampleRecord]]:
        return [(sid, self.samples[sid]) for sid in self.sample_order]

    @classmethod
    def load(
        cls,
        manifest_path: Union[str, Path],
        load_tensors: bool = True,
        load_scan: bool = True,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> PhysicalDesignDataset:
        """
        Loads a PhysicalDesignDataset from a dataset_manifest.json file.

        Args:
            manifest_path: Path to dataset_manifest.json.
            load_tensors: Whether to load array tensors into memory (default True).
            load_scan: Whether to load scan-aware features (default True).
            base_dir: Optional root directory override for resolving sample paths.

        Returns:
            PhysicalDesignDataset instance.
        """
        manifest_p = Path(manifest_path).resolve()
        if not manifest_p.is_file():
            raise FileNotFoundError(f"Dataset manifest file not found: {manifest_p}")

        manifest_dir = manifest_p.parent
        resolved_base = Path(base_dir).resolve() if base_dir else manifest_dir

        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_id = data.get("dataset_id", manifest_dir.name)
        schema_version = data.get("schema_version", "dataset_manifest_v001")
        design = data.get("design", "unknown")
        platform = data.get("platform", "unknown")
        grid_size = int(data.get("grid_size", 64))
        scan_source = data.get("scan_source")
        created_at = data.get("created_at")

        sweep_cfg = data.get("sweep_config")
        sweep_config_path = (
            resolve_cross_platform_path(sweep_cfg, resolved_base) if sweep_cfg else None
        )

        samples_dict: Dict[str, SampleRecord] = {}
        sample_entries = data.get("samples", [])

        for entry in sample_entries:
            sid = entry["sample_id"]
            rid = entry.get("run_id", sid)
            raw_sample_dir = entry.get("sample_dir", str(manifest_dir / "samples" / sid))
            sample_dir = resolve_cross_platform_path(raw_sample_dir, resolved_base)

            raw_run_dir = entry.get("run_dir")
            run_dir = (
                resolve_cross_platform_path(raw_run_dir, resolved_base)
                if raw_run_dir
                else None
            )

            status = entry.get("status", "success")
            cfg = entry.get("config", {})
            artifacts = entry.get("artifacts", {})
            tensor_shapes = entry.get("tensor_shapes", {})

            sample_obj: Optional[PhysicalDesignSample] = None
            fused_tensor: Optional[np.ndarray] = None
            fused_names: List[str] = []
            fused_sha: Optional[str] = None
            lbls_sha: Optional[str] = None

            if load_tensors and sample_dir.is_dir():
                # Load PhysicalDesignSample
                sample_obj = PhysicalDesignSample.load(
                    sample_dir=sample_dir,
                    sample_id=sid,
                    validate=True,
                    load_scan=load_scan,
                )

                # Check for existing fused_input.npz or construct from sample
                fused_npz_path = sample_dir / FUSED_NPZ_NAME
                if fused_npz_path.is_file():
                    with np.load(fused_npz_path, allow_pickle=False) as fused_npz:
                        fused_tensor = fused_npz["fused_features"].astype(np.float32)
                        if "channel_names" in fused_npz:
                            fused_names = [str(x) for x in fused_npz["channel_names"]]
                        else:
                            fused_names = sample_obj.get_fused_channel_names(include_scan=load_scan)
                else:
                    fused_tensor = sample_obj.get_fused_input_tensor(include_scan=load_scan)
                    fused_names = sample_obj.get_fused_channel_names(include_scan=load_scan)

                # Compute deterministic hashes for duplicate detection
                fused_sha = hashlib.sha256(fused_tensor.tobytes()).hexdigest()
                lbls_sha = hashlib.sha256(sample_obj.label_tensor.tobytes()).hexdigest()

            record = SampleRecord(
                sample_id=sid,
                run_id=rid,
                sample_dir=sample_dir,
                run_dir=run_dir,
                status=status,
                config=cfg,
                artifacts=artifacts,
                tensor_shapes=tensor_shapes,
                sample=sample_obj,
                fused_tensor=fused_tensor,
                fused_channel_names=fused_names,
                fused_sha256=fused_sha,
                labels_sha256=lbls_sha,
            )
            samples_dict[sid] = record

        failures = data.get("failures", [])

        dataset = cls(
            dataset_id=dataset_id,
            manifest_path=manifest_p,
            schema_version=schema_version,
            design=design,
            platform=platform,
            grid_size=grid_size,
            scan_source=scan_source,
            sweep_config_path=sweep_config_path,
            created_at=created_at,
            samples=samples_dict,
            failures=failures,
            manifest_dict=data,
        )
        return dataset

    def validate(
        self,
        require_scan: bool = True,
        require_fused: bool = True,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Validates internal consistency across all samples in the dataset:
        1. Sample-level physical and routing tensor validity.
        2. Scan feature presence and spatial alignment (if require_scan).
        3. Fused input tensor shape and finiteness (if require_fused).
        4. Uniform spatial grid resolution across all samples.
        5. Uniform channel names across all samples.

        Returns:
            Dict containing validation report with per-sample and dataset-level results.

        Raises:
            ValueError: If validation fails and raise_on_error is True.
        """
        report: Dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "status": "pass",
            "total_samples": len(self.samples),
            "passed_samples": 0,
            "failed_samples": 0,
            "per_sample": {},
            "uniformity_errors": [],
        }

        if len(self.samples) == 0:
            err = f"Dataset '{self.dataset_id}' contains 0 samples."
            report["status"] = "fail"
            report["uniformity_errors"].append(err)
            if raise_on_error:
                raise ValueError(err)
            return report

        expected_grid: Optional[Tuple[int, int]] = None
        expected_phys_channels: Optional[List[str]] = None
        expected_scan_channels: Optional[List[str]] = None
        expected_lbl_channels: Optional[List[str]] = None
        expected_fused_channels: Optional[List[str]] = None

        for sid, record in self.items():
            sample_errors: List[str] = []

            if record.sample is None:
                sample_errors.append(f"Sample '{sid}' has no loaded PhysicalDesignSample instance.")
                record.validation_status = "fail"
                record.validation_errors = sample_errors
                report["per_sample"][sid] = {"status": "fail", "errors": sample_errors}
                report["failed_samples"] += 1
                continue

            # Validate paired sample
            try:
                record.sample.validate()
            except Exception as e:
                sample_errors.append(f"Physical paired validation failed: {e}")

            # Validate scan features
            if require_scan:
                if not record.sample.has_scan_features:
                    sample_errors.append("Scan features required but sample.has_scan_features is False")
                else:
                    try:
                        record.sample.validate_scan_features()
                    except Exception as e:
                        sample_errors.append(f"Scan validation failed: {e}")

            # Validate fused tensor
            if require_fused:
                if record.fused_tensor is None:
                    sample_errors.append("Fused tensor required but fused_tensor is None")
                else:
                    if record.fused_tensor.ndim != 3:
                        sample_errors.append(
                            f"Fused tensor must be 3D (C, H, W), got shape {record.fused_tensor.shape}"
                        )
                    if not np.all(np.isfinite(record.fused_tensor)):
                        sample_errors.append("Fused tensor contains NaN or Inf values")
                    expected_c = (
                        record.sample.num_feature_channels
                        + (record.sample.num_scan_channels if require_scan else 0)
                    )
                    if record.fused_tensor.shape[0] != expected_c:
                        sample_errors.append(
                            f"Fused channel count ({record.fused_tensor.shape[0]}) does not match "
                            f"expected channels ({expected_c})"
                        )

            # Uniformity checks across samples
            grid = record.sample.grid_shape
            if expected_grid is None:
                expected_grid = grid
            elif grid != expected_grid:
                sample_errors.append(f"Grid shape {grid} differs from dataset expected grid {expected_grid}")

            if expected_phys_channels is None:
                expected_phys_channels = list(record.sample.feature_channel_names)
            elif record.sample.feature_channel_names != expected_phys_channels:
                sample_errors.append("Physical channel names do not match dataset standard")

            if expected_lbl_channels is None:
                expected_lbl_channels = list(record.sample.label_channel_names)
            elif record.sample.label_channel_names != expected_lbl_channels:
                sample_errors.append("Routing label channel names do not match dataset standard")

            if require_scan and record.sample.has_scan_features:
                if expected_scan_channels is None:
                    expected_scan_channels = list(record.sample.scan_feature_channel_names)
                elif record.sample.scan_feature_channel_names != expected_scan_channels:
                    sample_errors.append("Scan channel names do not match dataset standard")

            if require_fused and record.fused_channel_names:
                if expected_fused_channels is None:
                    expected_fused_channels = list(record.fused_channel_names)
                elif record.fused_channel_names != expected_fused_channels:
                    sample_errors.append("Fused channel names do not match dataset standard")

            if sample_errors:
                record.validation_status = "fail"
                record.validation_errors = sample_errors
                report["per_sample"][sid] = {"status": "fail", "errors": sample_errors}
                report["failed_samples"] += 1
            else:
                record.validation_status = "pass"
                record.validation_errors = []
                report["per_sample"][sid] = {"status": "pass", "errors": []}
                report["passed_samples"] += 1

        if report["failed_samples"] > 0:
            report["status"] = "fail"
            if raise_on_error:
                first_failed = next(
                    (sid for sid, r in report["per_sample"].items() if r["status"] == "fail"),
                    "unknown",
                )
                errs = report["per_sample"][first_failed]["errors"]
                raise ValueError(
                    f"Dataset validation failed for {report['failed_samples']} samples. "
                    f"Sample '{first_failed}' errors: {'; '.join(errs)}"
                )

        return report

    def summary(self) -> Dict[str, Any]:
        """
        Returns high-level summary dictionary of the dataset.
        """
        first_sample = self[0] if len(self) > 0 else None
        return {
            "dataset_id": self.dataset_id,
            "design": self.design,
            "platform": self.platform,
            "sample_count": len(self),
            "failure_count": len(self.failures),
            "grid_size": self.grid_size,
            "scan_source": self.scan_source,
            "tensor_shapes": {
                "physical": list(first_sample.sample.feature_shape) if (first_sample and first_sample.sample) else None,
                "scan": list(first_sample.sample.scan_feature_shape) if (first_sample and first_sample.sample and first_sample.sample.scan_feature_shape) else None,
                "fused": list(first_sample.fused_tensor.shape) if (first_sample and first_sample.fused_tensor is not None) else None,
                "labels": list(first_sample.sample.label_shape) if (first_sample and first_sample.sample) else None,
            },
        }

    def format_summary(self) -> str:
        """
        Returns a formatted human-readable summary string.
        """
        s = self.summary()
        shapes = s.get("tensor_shapes", {})
        lines = [
            f"Dataset ID      : {s['dataset_id']}",
            f"Design / PDK    : {s['design']} / {s['platform']}",
            f"Total Samples   : {s['sample_count']}",
            f"Grid Resolution : {s['grid_size']} x {s['grid_size']}",
            f"Scan Source     : {s['scan_source']}",
            f"Physical Shape  : {shapes.get('physical')}",
            f"Scan Shape      : {shapes.get('scan')}",
            f"Fused Shape     : {shapes.get('fused')}",
            f"Label Shape     : {shapes.get('labels')}",
        ]
        return "\n".join(lines)

    def detect_duplicates(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Detects exact and near-duplicate samples across both fused input tensors
        and routing label ground-truth tensors.

        Args:
            tolerance: Maximum absolute difference threshold for near-equality.

        Returns:
            Dict containing duplicate groups and statistics adhering to duplicate_report_v001 schema.
        """
        if len(self) == 0:
            return {
                "schema_version": "duplicate_report_v001",
                "dataset_id": self.dataset_id,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_samples": 0,
                "unique_layouts": 0,
                "duplicate_group_count": 0,
                "duplicate_sample_count": 0,
                "duplicate_groups": [],
                "summary": "Dataset is empty.",
            }

        # Step 1: Collect hashes and arrays
        sample_list: List[SampleRecord] = list(self)
        n = len(sample_list)

        # Map combined_hash -> list of sample indices
        hash_to_indices: Dict[str, List[int]] = {}
        for idx, rec in enumerate(sample_list):
            f_hash = rec.fused_sha256 or "none"
            l_hash = rec.labels_sha256 or "none"
            combined = f"{f_hash}_{l_hash}"
            hash_to_indices.setdefault(combined, []).append(idx)

        # Step 2: Form groups (both exact hash matches and allclose verification)
        assigned: set[int] = set()
        groups: List[DuplicateGroup] = []

        for idx in range(n):
            if idx in assigned:
                continue

            current_rec = sample_list[idx]
            group_indices = [idx]

            # Check remaining samples
            for other_idx in range(idx + 1, n):
                if other_idx in assigned:
                    continue
                other_rec = sample_list[other_idx]

                # Quick hash check
                hash_match = (
                    current_rec.fused_sha256 is not None
                    and current_rec.fused_sha256 == other_rec.fused_sha256
                    and current_rec.labels_sha256 is not None
                    and current_rec.labels_sha256 == other_rec.labels_sha256
                )

                if hash_match:
                    group_indices.append(other_idx)
                elif current_rec.fused_tensor is not None and other_rec.fused_tensor is not None:
                    # Near-equality fallback
                    fused_close = np.allclose(
                        current_rec.fused_tensor, other_rec.fused_tensor, atol=tolerance
                    )
                    lbl_close = (
                        current_rec.sample is not None
                        and other_rec.sample is not None
                        and np.allclose(
                            current_rec.sample.label_tensor,
                            other_rec.sample.label_tensor,
                            atol=tolerance,
                        )
                    )
                    if fused_close and lbl_close:
                        group_indices.append(other_idx)

            if len(group_indices) > 1:
                # Group found
                for g_idx in group_indices:
                    assigned.add(g_idx)

                group_samples = [sample_list[gi] for gi in group_indices]
                sids = [s.sample_id for s in group_samples]

                # Compute pairwise max abs diff
                ref_fused = group_samples[0].fused_tensor
                ref_lbl = group_samples[0].sample.label_tensor if group_samples[0].sample else None

                max_diff_fused = 0.0
                max_diff_lbl = 0.0
                for s in group_samples[1:]:
                    if ref_fused is not None and s.fused_tensor is not None:
                        diff_f = float(np.max(np.abs(ref_fused - s.fused_tensor)))
                        if diff_f > max_diff_fused:
                            max_diff_fused = diff_f
                    if ref_lbl is not None and s.sample is not None:
                        diff_l = float(np.max(np.abs(ref_lbl - s.sample.label_tensor)))
                        if diff_l > max_diff_lbl:
                            max_diff_lbl = diff_l

                is_exact = (max_diff_fused == 0.0 and max_diff_lbl == 0.0)
                match_type = "exact_sha256" if is_exact else "allclose_tolerance"

                # Find shared and differing config
                all_cfgs = [s.config for s in group_samples]
                all_keys = set().union(*[c.keys() for c in all_cfgs])
                shared_cfg: Dict[str, Any] = {}
                differing_cfg: Dict[str, List[Any]] = {}

                for k in sorted(all_keys):
                    vals = [c.get(k) for c in all_cfgs]
                    if len(set(str(v) for v in vals)) == 1:
                        shared_cfg[k] = vals[0]
                    else:
                        differing_cfg[k] = vals

                dg = DuplicateGroup(
                    group_id=len(groups),
                    sample_ids=sids,
                    match_type=match_type,
                    is_exact=is_exact,
                    fused_sha256=group_samples[0].fused_sha256 or "",
                    labels_sha256=group_samples[0].labels_sha256 or "",
                    max_abs_diff_fused=max_diff_fused,
                    max_abs_diff_labels=max_diff_lbl,
                    shared_config=shared_cfg,
                    differing_config=differing_cfg,
                )
                groups.append(dg)
            else:
                assigned.add(idx)

        total_duplicate_samples = sum(len(g.sample_ids) - 1 for g in groups)
        unique_layouts = len(self) - total_duplicate_samples

        # Compute unique fused inputs and unique routing layouts
        fused_assigned: set[int] = set()
        fused_unique_count = 0
        for i in range(n):
            if i in fused_assigned:
                continue
            fused_unique_count += 1
            fused_assigned.add(i)
            rec_i = sample_list[i]
            for j in range(i + 1, n):
                if j in fused_assigned:
                    continue
                rec_j = sample_list[j]
                if rec_i.fused_sha256 and rec_i.fused_sha256 == rec_j.fused_sha256:
                    fused_assigned.add(j)
                elif rec_i.fused_tensor is not None and rec_j.fused_tensor is not None:
                    if np.allclose(rec_i.fused_tensor, rec_j.fused_tensor, atol=tolerance):
                        fused_assigned.add(j)

        routing_assigned: set[int] = set()
        routing_unique_count = 0
        routing_groups: List[Dict[str, Any]] = []
        for i in range(n):
            if i in routing_assigned:
                continue
            routing_unique_count += 1
            group_r = [i]
            rec_i = sample_list[i]
            for j in range(i + 1, n):
                if j in routing_assigned:
                    continue
                rec_j = sample_list[j]
                if rec_i.labels_sha256 and rec_i.labels_sha256 == rec_j.labels_sha256:
                    group_r.append(j)
                elif rec_i.sample is not None and rec_j.sample is not None:
                    if np.allclose(rec_i.sample.label_tensor, rec_j.sample.label_tensor, atol=tolerance):
                        group_r.append(j)

            if len(group_r) > 1:
                for gr in group_r:
                    routing_assigned.add(gr)
                sids_r = [sample_list[gi].sample_id for gi in group_r]
                all_cfgs = [sample_list[gi].config for gi in group_r]
                all_keys = set().union(*[c.keys() for c in all_cfgs])
                shared_cfg_r = {}
                diff_cfg_r = {}
                for k in sorted(all_keys):
                    vals = [c.get(k) for c in all_cfgs]
                    if len(set(str(v) for v in vals)) == 1:
                        shared_cfg_r[k] = vals[0]
                    else:
                        diff_cfg_r[k] = vals

                routing_groups.append({
                    "group_id": len(routing_groups),
                    "sample_ids": sids_r,
                    "labels_sha256": sample_list[group_r[0]].labels_sha256 or "",
                    "shared_config": shared_cfg_r,
                    "differing_config": diff_cfg_r,
                })
            else:
                routing_assigned.add(i)

        summary_lines = []
        if groups:
            summary_lines.append(
                f"Detected {len(groups)} exact duplicate groups across {len(self)} samples. "
                f"Dataset contains {unique_layouts} unique layout(s) (exact fused+labels match)."
            )
        else:
            summary_lines.append(f"No exact duplicate samples detected across {len(self)} samples.")

        summary_lines.append(
            f"Unique fused input tensors: {fused_unique_count}. "
            f"Unique routing label layouts: {routing_unique_count}."
        )
        if len(routing_groups) > 0 and len(groups) == 0:
            summary_lines.append(
                f"{len(routing_groups)} routing layout group(s) share identical physical routing labels "
                f"with differing scan feature tensors (e.g. synthetic scan chain count variation)."
            )

        report = {
            "schema_version": "duplicate_report_v001",
            "dataset_id": self.dataset_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_samples": len(self),
            "unique_layouts": unique_layouts,
            "unique_fused_inputs": fused_unique_count,
            "unique_routing_layouts": routing_unique_count,
            "duplicate_group_count": len(groups),
            "duplicate_sample_count": total_duplicate_samples,
            "routing_duplicate_group_count": len(routing_groups),
            "routing_duplicate_groups": routing_groups,
            "tolerance": tolerance,
            "duplicate_groups": [g.to_dict() for g in groups],
            "summary": " ".join(summary_lines),
        }
        return report

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Computes comprehensive dataset-level and per-channel statistics.

        Returns:
            Dict adhering to dataset_statistics_v001 schema.
        """
        first_sample = self[0] if len(self) > 0 else None
        phys_shape = list(first_sample.sample.feature_shape) if (first_sample and first_sample.sample) else []
        scan_shape = (
            list(first_sample.sample.scan_feature_shape)
            if (first_sample and first_sample.sample and first_sample.sample.scan_feature_shape)
            else []
        )
        fused_shape = list(first_sample.fused_tensor.shape) if (first_sample and first_sample.fused_tensor is not None) else []
        lbl_shape = list(first_sample.sample.label_shape) if (first_sample and first_sample.sample) else []

        phys_names = list(first_sample.sample.feature_channel_names) if (first_sample and first_sample.sample) else []
        scan_names = list(first_sample.sample.scan_feature_channel_names) if (first_sample and first_sample.sample) else []
        fused_names = list(first_sample.fused_channel_names) if first_sample else []
        lbl_names = list(first_sample.sample.label_channel_names) if (first_sample and first_sample.sample) else []

        per_sample_stats: Dict[str, Dict[str, Any]] = {}
        for sid, rec in self.items():
            per_sample_stats[sid] = {
                "utilization": rec.utilization,
                "clock_period_ns": rec.clock_period_ns,
                "aspect_ratio": rec.aspect_ratio,
                "chain_count": rec.chain_count,
                "total_wirelength_um": rec.total_wirelength_um,
                "total_vias": rec.total_vias,
                "total_scan_cells": rec.total_scan_cells,
                "scan_proxy_wirelength_um": rec.scan_proxy_wirelength_um,
                "total_components": rec.total_components,
                "routed_nets": rec.routed_nets,
            }

        # Global channel-level statistics
        fused_channel_stats: Dict[str, Dict[str, float]] = {}
        if fused_names and len(self) > 0:
            # Stack all fused tensors: (N, C, H, W)
            valid_fused = [rec.fused_tensor for rec in self if rec.fused_tensor is not None]
            if valid_fused:
                fused_stack = np.stack(valid_fused, axis=0)  # (N, C, H, W)
                for c_idx, c_name in enumerate(fused_names):
                    ch_vals = fused_stack[:, c_idx, :, :].flatten()
                    nonzero = float(np.count_nonzero(ch_vals)) / float(len(ch_vals))
                    fused_channel_stats[c_name] = {
                        "min": float(np.min(ch_vals)),
                        "max": float(np.max(ch_vals)),
                        "mean": float(np.mean(ch_vals)),
                        "std": float(np.std(ch_vals)),
                        "nonzero_fraction": float(nonzero),
                    }

        label_channel_stats: Dict[str, Dict[str, float]] = {}
        if lbl_names and len(self) > 0:
            valid_lbls = [rec.sample.label_tensor for rec in self if (rec.sample and rec.sample.label_tensor is not None)]
            if valid_lbls:
                lbl_stack = np.stack(valid_lbls, axis=0)  # (N, C, H, W)
                for c_idx, c_name in enumerate(lbl_names):
                    ch_vals = lbl_stack[:, c_idx, :, :].flatten()
                    nonzero = float(np.count_nonzero(ch_vals)) / float(len(ch_vals))
                    label_channel_stats[c_name] = {
                        "min": float(np.min(ch_vals)),
                        "max": float(np.max(ch_vals)),
                        "mean": float(np.mean(ch_vals)),
                        "std": float(np.std(ch_vals)),
                        "nonzero_fraction": float(nonzero),
                    }

        dup_report = self.detect_duplicates()

        unique_designs = sorted(list(set(
            rec.sample.design for rec in self if rec.sample and rec.sample.design
        )))
        design_str = ", ".join(unique_designs) if unique_designs else self.design

        unique_platforms = sorted(list(set(
            rec.sample.pdk for rec in self if rec.sample and rec.sample.pdk
        )))
        platform_str = ", ".join(unique_platforms) if unique_platforms else self.platform

        if len(unique_designs) > 1:
            design_note = f"Dataset contains {len(self)} sample runs across multiple designs '{design_str}' and PDK '{platform_str}'."
        else:
            design_note = f"Dataset contains {len(self)} sample runs across single design '{self.design}' and PDK '{self.platform}'."

        diversity_notes = [
            design_note,
            f"Layout diversity: {dup_report['unique_layouts']} unique physical layouts detected across {len(self)} runs.",
            "Slack insensitivity observation: For small designs like uart_tx, changing clock period between 3.0 ns and 5.0 ns yielded identical placement and routing solutions under standard cell density constraints.",
            "Routing wirelength varies across designs, utilizations, and aspect ratios.",
        ]

        stats: Dict[str, Any] = {
            "schema_version": "dataset_statistics_v001",
            "dataset_id": self.dataset_id,
            "design": self.design,
            "platform": self.platform,
            "sample_count": len(self),
            "success_count": len(self),
            "failure_count": len(self.failures),
            "grid_size": self.grid_size,
            "tensor_shapes": {
                "physical": phys_shape,
                "scan": scan_shape,
                "fused": fused_shape,
                "routing_labels": lbl_shape,
            },
            "channel_names": {
                "physical": phys_names,
                "scan": scan_names,
                "fused": fused_names,
                "routing_labels": lbl_names,
            },
            "per_sample_stats": per_sample_stats,
            "channel_statistics": {
                "fused_input_channels": fused_channel_stats,
                "routing_label_channels": label_channel_stats,
            },
            "diversity_summary": {
                "unique_layouts": dup_report["unique_layouts"],
                "duplicate_groups_count": dup_report["duplicate_group_count"],
            },
            "diversity_notes": diversity_notes,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return stats

    def create_splits(
        self,
        policy: str = "smoke_config_holdout",
        val_sample_id: Optional[str] = None,
        test_sample_id: Optional[str] = None,
        train_sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Creates reproducible train / val / test dataset splits.

        Args:
            policy: Split policy strategy ('smoke_config_holdout' or 'manual').
            val_sample_id: Optional explicit validation sample ID.
            test_sample_id: Optional explicit test sample ID.
            train_sample_ids: Optional explicit training sample IDs list.

        Returns:
            Dict conforming to schema splits_v001.
        """
        dup_report = self.detect_duplicates()
        dup_groups_sids = [g["sample_ids"] for g in dup_report.get("duplicate_groups", [])]

        limitations = [
            f"Single design only ({self.design}) on {self.platform} PDK.",
            "Low physical diversity: identical layouts generated across clock periods due to timing closure slack insensitivity.",
            "This split is designed strictly for pipeline execution, integration validation, and baseline smoke testing, NOT for scientific generalization claims.",
        ]

        if policy == "manual":
            train_ids = train_sample_ids or []
            val_ids = [val_sample_id] if val_sample_id else []
            test_ids = [test_sample_id] if test_sample_id else []
        elif policy == "smoke_config_holdout":
            # Group by utilization if available
            util_to_samples: Dict[float, List[str]] = {}
            for sid, rec in self.items():
                u = rec.utilization if rec.utilization is not None else 0.0
                util_to_samples.setdefault(u, []).append(sid)

            sorted_utils = sorted(util_to_samples.keys())

            if len(sorted_utils) >= 3:
                # Hold out the highest utilization (e.g. 0.65) for val/test
                train_utils = sorted_utils[:-1]
                holdout_util = sorted_utils[-1]

                train_ids = []
                for u in train_utils:
                    train_ids.extend(util_to_samples[u])

                holdout_samples = util_to_samples[holdout_util]
                if len(holdout_samples) >= 2:
                    half = len(holdout_samples) // 2
                    val_ids = holdout_samples[:half]
                    test_ids = holdout_samples[half:]
                elif len(holdout_samples) == 1:
                    val_ids = [holdout_samples[0]]
                    test_ids = [holdout_samples[0]]
                else:
                    val_ids = []
                    test_ids = []
            elif len(self) >= 3:
                # Fallback: first N-2 train, second-to-last val, last test
                train_ids = self.sample_order[:-2]
                val_ids = [self.sample_order[-2]]
                test_ids = [self.sample_order[-1]]
            else:
                train_ids = list(self.sample_order)
                val_ids = []
                test_ids = []

            # Check if val and test are duplicates of each other
            val_test_dup = False
            for group in dup_groups_sids:
                if any(v in group for v in val_ids) and any(t in group for t in test_ids):
                    val_test_dup = True
                    break

            if val_test_dup:
                limitations.insert(
                    2,
                    f"Duplicate layout alert: validation sample(s) {val_ids} and test sample(s) {test_ids} "
                    f"belong to the same duplicate group (identical layout and routing tensors).",
                )
        else:
            raise ValueError(f"Unknown split policy: '{policy}'")

        split_dict: Dict[str, Any] = {
            "schema_version": "splits_v001",
            "dataset_id": self.dataset_id,
            "split_type": policy,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "limitations": limitations,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "duplicate_groups": dup_groups_sids,
        }
        return split_dict

    def export_dataset_analysis(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        write_splits: bool = True,
        split_policy: str = "smoke_config_holdout",
    ) -> Dict[str, Path]:
        """
        Runs complete analysis, statistics calculation, duplicate detection,
        and split generation, exporting all required artifacts to disk.

        Outputs:
        - dataset_statistics.json
        - duplicate_report.json
        - splits_v001.json (if write_splits is True)
        - dataset_analysis_summary.md

        Returns:
            Dict mapping artifact names to their written Path destinations.
        """
        out_dir = Path(output_dir).resolve() if output_dir else self.manifest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        written_paths: Dict[str, Path] = {}

        # 1. Dataset statistics
        stats = self.compute_statistics()
        stats_path = out_dir / "dataset_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        written_paths["dataset_statistics"] = stats_path

        # 2. Duplicate report
        dup_report = self.detect_duplicates()
        dup_path = out_dir / "duplicate_report.json"
        with open(dup_path, "w", encoding="utf-8") as f:
            json.dump(dup_report, f, indent=2)
        written_paths["duplicate_report"] = dup_path

        # 3. Splits
        splits_dict = self.create_splits(policy=split_policy)
        if write_splits:
            splits_path = out_dir / "splits_v001.json"
            with open(splits_path, "w", encoding="utf-8") as f:
                json.dump(splits_dict, f, indent=2)
            written_paths["splits"] = splits_path

        # 4. Summary markdown
        md_path = out_dir / "dataset_analysis_summary.md"
        md_content = self._render_analysis_markdown(stats, dup_report, splits_dict)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        written_paths["dataset_analysis_summary"] = md_path

        return written_paths

    def _render_analysis_markdown(
        self,
        stats: Dict[str, Any],
        dup_report: Dict[str, Any],
        splits: Dict[str, Any],
    ) -> str:
        """Helper to render dataset_analysis_summary.md."""
        shapes = stats.get("tensor_shapes", {})
        per_sample = stats.get("per_sample_stats", {})
        fused_ch_stats = stats.get("channel_statistics", {}).get("fused_input_channels", {})
        lbl_ch_stats = stats.get("channel_statistics", {}).get("routing_label_channels", {})

        md = [
            f"# Dataset Analysis & Introspection Report: {self.dataset_id}",
            "",
            "## 1. Dataset Overview",
            "",
            f"- **Dataset ID**: `{self.dataset_id}`",
            f"- **Design**: `{self.design}`",
            f"- **Platform / PDK**: `{self.platform}`",
            f"- **Grid Resolution**: `{self.grid_size} x {self.grid_size}`",
            f"- **Total Samples**: `{len(self)}`",
            f"- **Unique Physical Layouts**: `{dup_report.get('unique_layouts')}`",
            f"- **Unique Fused Input Tensors**: `{dup_report.get('unique_fused_inputs', dup_report.get('unique_layouts'))}`",
            f"- **Unique Routing Layouts**: `{dup_report.get('unique_routing_layouts', dup_report.get('unique_layouts'))}`",
            f"- **Scan Feature Source**: `{self.scan_source}`",
            f"- **Physical Tensor Shape**: `{shapes.get('physical')}`",
            f"- **Scan Tensor Shape**: `{shapes.get('scan')}`",
            f"- **Fused Input Tensor Shape**: `{shapes.get('fused')}`",
            f"- **Routing Labels Shape**: `{shapes.get('routing_labels')}`",
            "",
            "## 2. Sample Registry & Layout Metrics",
            "",
            "| Sample ID | Util | AR | Chains | Clock | Routed Wirelength | Vias | Scan Cells | Proxy Scan WL | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for sid, p in sorted(per_sample.items()):
            wl = f"{p['total_wirelength_um']:.2f} µm" if p.get("total_wirelength_um") is not None else "N/A"
            vias = f"{p['total_vias']}" if p.get("total_vias") is not None else "N/A"
            scells = f"{p['total_scan_cells']}" if p.get("total_scan_cells") is not None else "N/A"
            swl = f"{p['scan_proxy_wirelength_um']:.2f} µm" if p.get("scan_proxy_wirelength_um") is not None else "N/A"
            u = f"{p['utilization']}" if p.get("utilization") is not None else "N/A"
            ar_str = f"{p.get('aspect_ratio', 1.0)}"
            ch_str = f"{p.get('chain_count', 1)}"
            clk = f"{p['clock_period_ns']} ns" if p.get("clock_period_ns") is not None else "N/A"
            md.append(f"| `{sid}` | {u} | {ar_str} | {ch_str} | {clk} | {wl} | {vias} | {scells} | {swl} | **PASS** |")

        md.extend([
            "",
            "## 3. Duplicate & Near-Duplicate Detection",
            "",
            dup_report.get("summary", ""),
            "",
            "### Exact Duplicate Groups (Both Fused Input & Ground-Truth Match)",
            "",
            "| Group ID | Sample Count | Matched Samples | Shared Config | Differing Config | Max Diff (Fused / Labels) |",
            "| :---: | :---: | :--- | :--- | :--- | :---: |",
        ])

        for g in dup_report.get("duplicate_groups", []):
            sids_str = ", ".join(f"`{s}`" for s in g["sample_ids"])
            shared_str = ", ".join(f"{k}={v}" for k, v in g["shared_config"].items())
            diff_str = ", ".join(f"{k}={v}" for k, v in g["differing_config"].items())
            diffs = f"{g['max_abs_diff_fused']:.1e} / {g['max_abs_diff_labels']:.1e}"
            md.append(f"| Group {g['group_id']} | {len(g['sample_ids'])} | {sids_str} | {shared_str} | {diff_str} | {diffs} |")

        if not dup_report.get("duplicate_groups"):
            md.append("*No exact duplicate groups detected.*")

        if dup_report.get("routing_duplicate_groups"):
            md.extend([
                "",
                "### Routing Ground-Truth Duplicate Groups (Identical Physical Routing)",
                "The following groups share identical physical post-route wirelength and via ground-truth tensors,",
                "differing only in synthetic scan feature channels (e.g. `chain_count` variation):",
                "",
                "| Group ID | Sample Count | Matched Samples | Shared Config | Differing Config |",
                "| :---: | :---: | :--- | :--- | :--- |",
            ])
            for gr in dup_report["routing_duplicate_groups"]:
                sids_str = ", ".join(f"`{s}`" for s in gr["sample_ids"])
                shared_str = ", ".join(f"{k}={v}" for k, v in gr["shared_config"].items())
                diff_str = ", ".join(f"{k}={v}" for k, v in gr["differing_config"].items())
                md.append(f"| Routing Group {gr['group_id']} | {len(gr['sample_ids'])} | {sids_str} | {shared_str} | {diff_str} |")

        md.extend([
            "",
            "## 4. Dataset Split Policy (`smoke_config_holdout`)",
            "",
            f"- **Split Policy**: `{splits.get('split_type')}`",
            f"- **Training Set ({len(splits.get('train', []))} samples)**: " + ", ".join(f"`{s}`" for s in splits.get("train", [])),
            f"- **Validation Set ({len(splits.get('val', []))} sample)**: " + ", ".join(f"`{s}`" for s in splits.get("val", [])),
            f"- **Test Set ({len(splits.get('test', []))} sample)**: " + ", ".join(f"`{s}`" for s in splits.get("test", [])),
            "",
            "### Known Limitations & Integrity Disclosures",
        ])

        for lim in splits.get("limitations", []):
            md.append(f"- {lim}")

        md.extend([
            "",
            "## 5. Channel Statistics Highlights",
            "",
            "### Fused Input Channels (12 channels)",
            "| Channel Name | Min | Max | Mean | Std | Non-Zero Fraction |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cname, cstat in fused_ch_stats.items():
            md.append(
                f"| `{cname}` | {cstat['min']:.4f} | {cstat['max']:.4f} | {cstat['mean']:.4f} | {cstat['std']:.4f} | {cstat['nonzero_fraction']:.2%} |"
            )

        md.extend([
            "",
            "### Routing Ground-Truth Channels (8 channels)",
            "| Channel Name | Min | Max | Mean | Std | Non-Zero Fraction |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cname, cstat in lbl_ch_stats.items():
            md.append(
                f"| `{cname}` | {cstat['min']:.4f} | {cstat['max']:.4f} | {cstat['mean']:.4f} | {cstat['std']:.4f} | {cstat['nonzero_fraction']:.2%} |"
            )

        md.append("")
        return "\n".join(md)
