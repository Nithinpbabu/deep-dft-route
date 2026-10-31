"""
Execution engine for Stage 4/5 multi-run sweeps.
Orchestrates EDA physical design flow, spatial feature extraction,
routing label extraction, synthetic scan metadata generation,
scan feature extraction, sample validation, and fused tensor export.
Supports multi-design sweeps, varying PDK platforms, and floorplan aspect ratios.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from dft_hotspot.config import ConfigManager
from dft_hotspot.data.dataset import PhysicalDesignSample
from dft_hotspot.dft.synthetic_scan import SyntheticScanGenerator
from dft_hotspot.eda import FlowRunner
from dft_hotspot.features.extractor import PhysicalFeatureExtractor
from dft_hotspot.features.scan_features import ScanFeatureExtractor
from dft_hotspot.labels.routing_extractor import RoutingLabelExtractor
from dft_hotspot.sweep.config import SweepConfig, SweepPoint
from dft_hotspot.utils import setup_logger

logger = setup_logger("sweep_runner")


def find_def_file(
    run_dir: Path,
    explicit_def: Optional[str] = None,
    design_name: Optional[str] = None,
) -> Path:
    """Locates target DEF file from run directory."""
    if explicit_def:
        path = Path(explicit_def)
        if not path.is_file():
            path = run_dir / explicit_def
        if path.is_file():
            return path
        raise FileNotFoundError(f"Specified DEF file not found: {explicit_def}")

    candidates: List[Path] = []
    if design_name:
        candidates.extend([
            run_dir / "scripts" / f"{design_name}_final.def",
            run_dir / f"{design_name}_final.def",
        ])
    candidates.extend([
        run_dir / "scripts" / "uart_tx_final.def",
        run_dir / "uart_tx_final.def",
    ])
    for c in candidates:
        if c.is_file():
            return c

    final_defs = list(run_dir.rglob("*_final.def"))
    if final_defs:
        return final_defs[0]

    all_defs = list(run_dir.rglob("*.def"))
    if all_defs:
        return all_defs[0]

    raise FileNotFoundError(f"No .def file found within run directory: {run_dir}")


def export_fused_sample_record(sample_dir: Path) -> Path:
    """Helper to export fused_input.npz and metadata for a processed sample."""
    import numpy as np
    from dft_hotspot.data.dataset import FEATURE_NPZ_NAME, SCAN_NPZ_NAME

    sample = PhysicalDesignSample.load(sample_dir, validate=True, load_scan=True)
    fused_features = sample.get_fused_input_tensor(include_scan=True)
    channel_names = sample.get_fused_channel_names(include_scan=True)

    if sample.has_scan_features:
        assert sample.raw_scan_feature_tensor is not None
        raw_fused_features = np.concatenate(
            [sample.raw_feature_tensor, sample.raw_scan_feature_tensor],
            axis=0,
        ).astype(np.float32)
    else:
        raw_fused_features = sample.raw_feature_tensor.copy().astype(np.float32)

    output_npz_path = sample_dir / "fused_input.npz"
    np.savez_compressed(
        output_npz_path,
        fused_features=fused_features,
        raw_fused_features=raw_fused_features,
        channel_names=channel_names,
    )

    meta_path = sample_dir / "fused_input_metadata.json"
    metadata = {
        "schema_version": "fused_input_v001",
        "sample_id": sample.sample_id,
        "design": sample.design,
        "pdk": sample.pdk,
        "has_scan_features": sample.has_scan_features,
        "scan_source": sample.scan_source,
        "tensor_shape": list(fused_features.shape),
        "channel_count": len(channel_names),
        "channel_names": channel_names,
        "source_sample_dir": str(sample.sample_dir),
        "physical_feature_path": str(sample.sample_dir / FEATURE_NPZ_NAME),
        "scan_feature_path": str(sample.sample_dir / SCAN_NPZ_NAME) if sample.has_scan_features else None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_npz_path


class SweepRunner:
    """
    Orchestrates the multi-run sweep execution.
    """

    def __init__(
        self,
        config: SweepConfig,
        dry_run_plan: bool = False,
        limit: Optional[int] = None,
        skip_existing: bool = False,
        continue_on_error: bool = False,
        force: bool = False,
    ) -> None:
        self.config = config
        self.dry_run_plan = dry_run_plan
        self.limit = limit
        self.skip_existing = skip_existing
        self.continue_on_error = continue_on_error
        self.force = force

    def generate_plan_text(self) -> str:
        """Generates human-readable text for planned sweep points."""
        points = self.config.expand_points()
        if self.limit is not None and self.limit > 0:
            points = points[: self.limit]

        lines = [
            "=" * 75,
            "STAGE 4a: MULTI-RUN PARAMETER SWEEP PLAN"
            if self.config.schema_version == "sweep_config_v001"
            else "STAGE 5a: MULTI-RUN PARAMETER SWEEP PLAN",
            "=" * 75,
            f"Dataset ID          : {self.config.dataset_id}",
            f"Design / Platform   : {self.config.design} / {self.config.platform}",
            f"Base Flow Config    : {self.config.base_flow_config}",
            f"Grid Resolution     : {self.config.grid_size} x {self.config.grid_size}",
            f"Scan Source / Chain : {self.config.scan_source} (Chains: {self.config.chain_count})",
            f"Output Dataset Dir  : {self.config.output_dataset_dir}",
            f"Total Planned Runs  : {len(points)}",
            "",
            "Planned Sweep Configurations:",
        ]
        for p in points:
            ar_info = f" | ar={p.aspect_ratio}" if p.aspect_ratio != 1.0 else ""
            ch_info = f" | chains={p.chain_count}" if p.chain_count != 1 else ""
            lines.append(
                f"  [{p.index}] Sample ID: {p.sample_id:<44} | "
                f"design={p.design:<11} | util={p.utilization:<4} | clk={p.clock_period_ns:<4}ns{ar_info}{ch_info} | run={p.run_id}"
            )
        lines.append("=" * 75)
        return "\n".join(lines)

    def run_sweep(self) -> Dict[str, Any]:
        """
        Executes the planned parameter sweep.
        """
        all_points = self.config.expand_points()
        active_points = all_points[: self.limit] if (self.limit and self.limit > 0) else all_points

        if self.dry_run_plan:
            print(self.generate_plan_text())
            return {
                "dataset_id": self.config.dataset_id,
                "status": "dry_run_plan",
                "planned_points": [p.config_dict for p in active_points],
                "sample_count": len(active_points),
            }

        # Initialize directories
        self.config.output_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.config.samples_dir.mkdir(parents=True, exist_ok=True)
        self.config.configs_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.runs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting sweep execution for dataset '{self.config.dataset_id}' ({len(active_points)} runs)...")

        successful_samples: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for p_idx, point in enumerate(active_points):
            sample_dir = self.config.samples_dir / point.sample_id
            run_dir = self.config.runs_dir / point.run_id

            logger.info(
                f"\n--- [{p_idx + 1}/{len(active_points)}] Running Sweep Point: {point.sample_id} "
                f"(design={point.design}, util={point.utilization}, clk={point.clock_period_ns}ns, ar={point.aspect_ratio}) ---"
            )

            # Check if existing sample can be reused
            if self.skip_existing and not self.force:
                if (sample_dir / "fused_input.npz").is_file() and (sample_dir / "metadata.json").is_file():
                    try:
                        sample = PhysicalDesignSample.load(sample_dir, validate=True, load_scan=True)
                        logger.info(f"Skipping existing valid sample: {point.sample_id}")
                        sample_entry = self._build_sample_entry(point, sample_dir, run_dir, sample)
                        successful_samples.append(sample_entry)
                        continue
                    except Exception as ex:
                        logger.warning(f"Existing sample {point.sample_id} failed validation ({ex}); re-executing...")

            sample_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Step 1: Run Physical Flow
                logger.info(f"[{point.sample_id}] Step 1/7: Running physical design flow...")
                flow_manifest = self._run_physical_flow(point)
                actual_run_dir = Path(flow_manifest["run_dir"])

                # Step 2: Extract Physical Features
                logger.info(f"[{point.sample_id}] Step 2/7: Extracting physical layout features...")
                def_path = find_def_file(actual_run_dir, design_name=point.design)
                feat_extractor = PhysicalFeatureExtractor(grid_size=point.grid_size)
                feat_extractor.process_and_save_sample(
                    def_path=def_path,
                    output_dir=sample_dir,
                    source_run_dir=actual_run_dir,
                    design_name=point.design,
                    pdk_name=point.platform,
                )

                # Step 3: Extract Routing Labels
                logger.info(f"[{point.sample_id}] Step 3/7: Extracting post-route routing labels...")
                lbl_extractor = RoutingLabelExtractor(grid_size=point.grid_size)
                lbl_extractor.process_and_save_sample(
                    def_path=def_path,
                    output_dir=sample_dir,
                    source_run_dir=actual_run_dir,
                    design_name=point.design,
                    pdk_name=point.platform,
                )

                # Step 4: Generate Synthetic Scan Metadata
                logger.info(f"[{point.sample_id}] Step 4/7: Generating synthetic scan metadata...")
                scan_meta = SyntheticScanGenerator.generate(
                    run_dir=actual_run_dir,
                    def_path=def_path,
                    chain_count=point.chain_count,
                    ordering=point.ordering,
                    design=point.design,
                    platform=point.platform,
                )
                scan_meta.save_json(sample_dir / "scan_metadata.json")
                (sample_dir / "scan_metadata_summary.txt").write_text(scan_meta.to_summary_text(), encoding="utf-8")

                # Step 5: Extract Scan Features
                logger.info(f"[{point.sample_id}] Step 5/7: Extracting scan-aware feature tensors...")
                scan_feat_extractor = ScanFeatureExtractor(grid_size=point.grid_size)
                scan_feat_extractor.process_and_save_sample(
                    sample_dir=sample_dir,
                    scan_metadata_path=sample_dir / "scan_metadata.json",
                    output_dir=sample_dir,
                )

                # Step 6: Validate Sample
                logger.info(f"[{point.sample_id}] Step 6/7: Validating paired sample with --require-scan...")
                sample = PhysicalDesignSample.load(sample_dir, validate=True, load_scan=True)
                if not sample.has_scan_features:
                    raise ValueError("Scan features missing under --require-scan contract")

                # Step 7: Export Fused Sample
                logger.info(f"[{point.sample_id}] Step 7/7: Exporting fused 12-channel baseline tensor...")
                export_fused_sample_record(sample_dir)

                # Construct successful sample record
                sample_entry = self._build_sample_entry(point, sample_dir, actual_run_dir, sample)
                successful_samples.append(sample_entry)
                logger.info(f"[{point.sample_id}] COMPLETED SUCCESSFULLY.")

            except Exception as err:
                fail_stage = self._determine_failure_stage(err)
                fail_reason = str(err)
                logger.error(f"[{point.sample_id}] FAILED at stage '{fail_stage}': {fail_reason}", exc_info=True)

                failure_entry = {
                    "config": point.config_dict,
                    "stage": fail_stage,
                    "status": "failed",
                    "reason": fail_reason,
                    "sample_id": point.sample_id,
                    "run_dir": str(run_dir) if run_dir.exists() else None,
                }
                failures.append(failure_entry)

                if not self.continue_on_error:
                    logger.warning("Terminating sweep early (--continue-on-error not specified).")
                    break

        # Post-execution: serialize dataset manifest & summaries
        manifest_data = self._write_dataset_manifest(active_points, successful_samples, failures)
        self._write_dataset_files(successful_samples, failures)

        logger.info("\n" + "=" * 75)
        logger.info(f"SWEEP SUMMARY: {len(successful_samples)} succeeded, {len(failures)} failed")
        logger.info(f"Manifest written to: {self.config.output_dataset_dir / 'dataset_manifest.json'}")
        logger.info("=" * 75)

        return manifest_data

    def _run_physical_flow(self, point: SweepPoint) -> Dict[str, Any]:
        """Generates resolved config and executes FlowRunner."""
        flow_cfg_path = point.base_flow_config if point.base_flow_config else self.config.base_flow_config
        base_cfg = ConfigManager.load_config(flow_cfg_path)

        # Mutate parameters for this sweep point
        base_cfg["design"]["clock_period"] = point.clock_period_ns
        base_cfg["flow"]["target_utilization"] = point.utilization

        if point.aspect_ratio is not None and abs(point.aspect_ratio - 1.0) > 1e-4:
            if abs(point.aspect_ratio - 1.5) < 0.05:
                base_cfg["flow"]["die_area"] = "0 0 116 84"
                base_cfg["flow"]["core_area"] = "10 10 106 74"
            else:
                core_area_val = 6400.0
                h = round((core_area_val / point.aspect_ratio) ** 0.5, 1)
                w = round(h * point.aspect_ratio, 1)
                die_w = round(w + 20.0, 1)
                die_h = round(h + 20.0, 1)
                core_urx = round(10.0 + w, 1)
                core_ury = round(10.0 + h, 1)
                base_cfg["flow"]["die_area"] = f"0 0 {die_w} {die_h}"
                base_cfg["flow"]["core_area"] = f"10 10 {core_urx} {core_ury}"

        resolved_yaml_path = self.config.configs_dir / f"{point.sample_id}.yaml"
        resolved_yaml_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        manifest = FlowRunner.run_flow(
            config_path=resolved_yaml_path,
            dry_run=False,
            output_root=self.config.runs_dir,
            run_name=point.run_id,
        )

        if manifest.get("status") != "success":
            reason = manifest.get("failure_reason") or f"EDA status was '{manifest.get('status')}'"
            raise RuntimeError(f"Physical design flow failed: {reason}")

        return manifest

    def _determine_failure_stage(self, err: Exception) -> str:
        """Infers pipeline stage from error text."""
        msg = str(err).lower()
        if "physical design flow failed" in msg or "yosys" in msg or "openroad" in msg:
            return "run_flow"
        if "physicalfeatureextractor" in msg or "features_spatial" in msg:
            return "features"
        if "routinglabelextractor" in msg or "labels_routing" in msg:
            return "labels"
        if "syntheticscan" in msg or "scan_metadata" in msg:
            return "scan_metadata"
        if "scanfeatureextractor" in msg or "features_scan" in msg:
            return "scan_features"
        if "fused" in msg:
            return "fused_export"
        if "validate" in msg or "require-scan" in msg:
            return "validate"
        return "unknown"

    def _build_sample_entry(
        self,
        point: SweepPoint,
        sample_dir: Path,
        run_dir: Path,
        sample: PhysicalDesignSample,
    ) -> Dict[str, Any]:
        """Constructs manifest entry for a successful sample."""
        fused_shape = list(sample.get_fused_input_tensor().shape)
        scan_shape = list(sample.scan_feature_shape) if sample.scan_feature_shape else []

        return {
            "sample_id": point.sample_id,
            "run_id": point.run_id,
            "run_dir": str(run_dir),
            "sample_dir": str(sample_dir),
            "status": "success",
            "config": point.config_dict,
            "artifacts": {
                "features_spatial": str(sample_dir / "features_spatial.npz"),
                "labels_routing": str(sample_dir / "labels_routing.npz"),
                "scan_metadata": str(sample_dir / "scan_metadata.json"),
                "features_scan": str(sample_dir / "features_scan.npz"),
                "fused_input": str(sample_dir / "fused_input.npz"),
                "metadata": str(sample_dir / "metadata.json"),
                "routing_label_metadata": str(sample_dir / "routing_label_metadata.json"),
                "scan_feature_metadata": str(sample_dir / "scan_feature_metadata.json"),
                "fused_input_metadata": str(sample_dir / "fused_input_metadata.json"),
            },
            "tensor_shapes": {
                "physical": list(sample.feature_shape),
                "routing_labels": list(sample.label_shape),
                "scan": scan_shape,
                "fused": fused_shape,
            },
            "validation": {
                "sample_validation": "pass",
            },
        }

    def _write_dataset_manifest(
        self,
        active_points: List[SweepPoint],
        successful_samples: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Serializes dataset_manifest.json."""
        manifest_path = self.config.output_dataset_dir / "dataset_manifest.json"

        unique_designs = sorted(list(set(p.design for p in active_points)))
        unique_platforms = sorted(list(set(p.platform for p in active_points)))
        design_str = self.config.design if len(unique_designs) <= 1 else (
            self.config.design if self.config.design not in ("unknown", "") else ", ".join(unique_designs)
        )
        platform_str = self.config.platform if len(unique_platforms) <= 1 else ", ".join(unique_platforms)

        manifest_data = {
            "dataset_id": self.config.dataset_id,
            "schema_version": "dataset_manifest_v001",
            "design": design_str,
            "platform": platform_str,
            "designs": unique_designs,
            "platforms": unique_platforms,
            "grid_size": self.config.grid_size,
            "scan_source": self.config.scan_source,
            "sweep_config": str(self.config.config_path),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sample_count": len(successful_samples),
            "success_count": len(successful_samples),
            "failure_count": len(failures),
            "samples": successful_samples,
            "failures": failures,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        return manifest_data

    def _write_dataset_files(
        self,
        successful_samples: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
    ) -> None:
        """Writes samples.txt, failures.json (if any), and dataset_summary.md."""
        # 1. samples.txt
        samples_txt_path = self.config.output_dataset_dir / "samples.txt"
        sample_ids = [s["sample_id"] for s in successful_samples]
        samples_txt_path.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")

        # 2. failures.json
        failures_path = self.config.output_dataset_dir / "failures.json"
        if failures:
            failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        elif failures_path.is_file():
            failures_path.unlink()

        # 3. dataset_summary.md
        summary_md_path = self.config.output_dataset_dir / "dataset_summary.md"
        summary_lines = [
            f"# Dataset Summary: {self.config.dataset_id}",
            "",
            f"- **Design**: `{self.config.design}`",
            f"- **Platform**: `{self.config.platform}`",
            f"- **Spatial Grid**: `{self.config.grid_size} x {self.config.grid_size}`",
            f"- **Scan Source**: `{self.config.scan_source}`",
            f"- **Total Successful Samples**: `{len(successful_samples)}`",
            f"- **Total Failures**: `{len(failures)}`",
            "",
            "## Successful Samples",
            "",
            "| Sample ID | Design | Utilization | Clock (ns) | Aspect Ratio | Physical Shape | Scan Shape | Fused Shape | Validation |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in successful_samples:
            cfg = s["config"]
            shapes = s["tensor_shapes"]
            des = cfg.get("design", self.config.design)
            ar = cfg.get("aspect_ratio", 1.0)
            summary_lines.append(
                f"| `{s['sample_id']}` | `{des}` | {cfg['utilization']} | {cfg['clock_period_ns']} | {ar} | "
                f"`{tuple(shapes['physical'])}` | `{tuple(shapes['scan'])}` | `{tuple(shapes['fused'])}` | PASS |"
            )

        if failures:
            summary_lines.extend([
                "",
                "## Failures",
                "",
                "| Sample ID | Stage | Reason |",
                "| :--- | :---: | :--- |",
            ])
            for f in failures:
                summary_lines.append(f"| `{f.get('sample_id', 'N/A')}` | `{f['stage']}` | {f['reason']} |")

        summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
