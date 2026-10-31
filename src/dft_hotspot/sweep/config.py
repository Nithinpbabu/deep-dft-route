"""
Configuration loader and parameter space expander for Stage 4/5 multi-run sweeps.
Supports single-design and multi-design parameter sweeps with floorplan aspect ratio variations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class SweepPoint:
    """
    Concrete parameter assignment for a single sweep point.
    """
    index: int
    utilization: float
    clock_period_ns: float
    chain_count: int
    grid_size: int
    ordering: str
    sample_id: str
    run_id: str
    design: str = "uart_tx"
    platform: str = "nangate45"
    base_flow_config: Optional[Path] = None
    aspect_ratio: float = 1.0

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Dictionary representation matching dataset manifest specification."""
        d: Dict[str, Any] = {
            "utilization": self.utilization,
            "clock_period_ns": self.clock_period_ns,
            "chain_count": self.chain_count,
            "grid_size": self.grid_size,
        }
        if self.design:
            d["design"] = self.design
        if self.platform:
            d["platform"] = self.platform
        if self.aspect_ratio != 1.0:
            d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class SweepConfig:
    """
    Top-level specification for a multi-run parameter sweep.
    """
    config_path: Path
    dataset_id: str
    schema_version: str
    design: str
    platform: str
    base_flow_config: Path
    grid_size: int
    chain_count: int
    ordering: str
    scan_source: str
    output_dataset_dir: Path
    runs_dir: Path
    samples_dir: Path
    configs_dir: Path
    logs_dir: Path
    sweep_parameters: Dict[str, List[Any]] = field(default_factory=dict)
    designs: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> SweepConfig:
        """
        Loads and validates sweep configuration from YAML file.
        """
        path = Path(config_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Sweep configuration file not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        raw_dict = yaml.safe_load(raw_text)
        if not isinstance(raw_dict, dict):
            raise ValueError(f"Sweep config at {path} must contain a top-level dictionary")

        has_designs_list = "designs" in raw_dict and isinstance(raw_dict["designs"], list) and len(raw_dict["designs"]) > 0

        if has_designs_list:
            required_keys = ["dataset_id"]
        else:
            required_keys = ["dataset_id", "design", "platform", "base_flow_config", "sweep_parameters"]

        missing = [k for k in required_keys if k not in raw_dict]
        if missing:
            raise ValueError(f"Sweep config at {path} missing required keys: {', '.join(missing)}")

        dataset_id = str(raw_dict["dataset_id"])
        schema_version = str(raw_dict.get("schema_version", "sweep_config_v001"))
        design = str(raw_dict.get("design", "multi"))
        platform = str(raw_dict.get("platform", "nangate45"))

        # Base flow config
        raw_base_cfg = raw_dict.get("base_flow_config")
        if raw_base_cfg:
            base_cfg = Path(raw_base_cfg)
            if not base_cfg.is_absolute():
                base_cfg = (path.parent.parent.parent / base_cfg).resolve()
        else:
            base_cfg = (path.parent.parent.parent / "configs/flows/uart_nangate45.yaml").resolve()

        grid_size = int(raw_dict.get("grid_size", 64))
        raw_chain = raw_dict.get("chain_count", 1)
        if isinstance(raw_chain, list):
            chain_count = int(raw_chain[0]) if raw_chain else 1
        else:
            chain_count = int(raw_chain)
        ordering = str(raw_dict.get("ordering", "row_major_by_y_then_x"))
        scan_source = str(raw_dict.get("scan_source", "synthetic_proxy"))

        # Output dataset dir
        out_dataset_raw = raw_dict.get("output_dataset_dir", f"data/processed/datasets/{dataset_id}")
        out_dataset_dir = Path(out_dataset_raw)
        if not out_dataset_dir.is_absolute():
            out_dataset_dir = (path.parent.parent.parent / out_dataset_dir).resolve()

        # Runs dir
        runs_dir_raw = raw_dict.get("runs_dir", "runs")
        runs_dir = Path(runs_dir_raw)
        if not runs_dir.is_absolute():
            runs_dir = (path.parent.parent.parent / runs_dir).resolve()

        samples_dir = out_dataset_dir / "samples"
        configs_dir = out_dataset_dir / "configs"
        logs_dir = out_dataset_dir / "logs"

        sweep_params = raw_dict.get("sweep_parameters", {})
        if not isinstance(sweep_params, dict):
            raise ValueError(f"'sweep_parameters' must be a dictionary in {path}")
        else:
            sweep_params = dict(sweep_params)

        for p_key in [
            "utilization",
            "target_utilization",
            "clock_period_ns",
            "clock_period",
            "aspect_ratio",
            "chain_count",
        ]:
            if p_key in raw_dict and p_key not in sweep_params:
                val = raw_dict[p_key]
                sweep_params[p_key] = val if isinstance(val, list) else [val]

        designs = raw_dict.get("designs", [])
        if not isinstance(designs, list):
            designs = []
        normalized_designs = []
        for d in designs:
            if isinstance(d, str):
                normalized_designs.append({"design": d})
            elif isinstance(d, dict):
                normalized_designs.append(d)
        designs = normalized_designs

        return cls(
            config_path=path,
            dataset_id=dataset_id,
            schema_version=schema_version,
            design=design,
            platform=platform,
            base_flow_config=base_cfg,
            grid_size=grid_size,
            chain_count=chain_count,
            ordering=ordering,
            scan_source=scan_source,
            output_dataset_dir=out_dataset_dir,
            runs_dir=runs_dir,
            samples_dir=samples_dir,
            configs_dir=configs_dir,
            logs_dir=logs_dir,
            sweep_parameters=sweep_params,
            designs=designs,
        )

    def expand_points(self) -> List[SweepPoint]:
        """
        Deterministically expands sweep parameter combinations.
        Supports both multi-design sweeps (via self.designs) and single-design Cartesian product.
        """
        points: List[SweepPoint] = []
        idx = 0

        # Global chain count fallback or sweep
        global_chain_vals = self.sweep_parameters.get("chain_count")
        if global_chain_vals is not None:
            if not isinstance(global_chain_vals, list):
                global_chain_vals = [global_chain_vals]
        else:
            global_chain_vals = [self.chain_count]

        has_global_chain_sweep = len(global_chain_vals) > 1

        if self.designs:
            for d_entry in self.designs:
                if isinstance(d_entry, str):
                    d_entry = {"design": d_entry}
                d_name = str(d_entry["design"])
                d_platform = str(d_entry.get("platform", self.platform))
                d_base_raw = d_entry.get("base_flow_config")
                if d_base_raw:
                    d_base_cfg = Path(d_base_raw)
                    if not d_base_cfg.is_absolute():
                        d_base_cfg = (self.config_path.parent.parent.parent / d_base_cfg).resolve()
                else:
                    candidate = (self.config_path.parent.parent.parent / f"configs/flows/{d_name}_{d_platform}.yaml").resolve()
                    if candidate.is_file():
                        d_base_cfg = candidate
                    elif d_name == "dft_counter":
                        d_base_cfg = (self.config_path.parent.parent.parent / "configs/flows/dft_smoke_nangate45.yaml").resolve()
                    else:
                        d_base_cfg = self.base_flow_config

                d_ordering = str(d_entry.get("ordering", self.ordering))
                d_grid_size = int(d_entry.get("grid_size", self.grid_size))

                # Check if explicit sweep_points are provided
                if "sweep_points" in d_entry and isinstance(d_entry["sweep_points"], list):
                    for sp in d_entry["sweep_points"]:
                        u_flt = float(sp["utilization"])
                        c_flt = float(sp["clock_period_ns"])
                        ar_flt = float(sp.get("aspect_ratio", 1.0))

                        sp_chain = sp.get("chain_count")
                        if sp_chain is not None:
                            sp_chain_vals = sp_chain if isinstance(sp_chain, list) else [sp_chain]
                        else:
                            d_chain = d_entry.get("chain_count")
                            if d_chain is not None:
                                sp_chain_vals = d_chain if isinstance(d_chain, list) else [d_chain]
                            else:
                                sp_chain_vals = global_chain_vals

                        has_chain_sweep = len(sp_chain_vals) > 1 or has_global_chain_sweep

                        for ch_val in sp_chain_vals:
                            ch_int = int(ch_val)
                            u_str = f"{int(round(u_flt * 100)):02d}"
                            c_str = str(c_flt).replace(".", "p")
                            ar_str = str(ar_flt).replace(".", "p")
                            sample_id = f"{d_name}_{d_platform}_u{u_str}_clk{c_str}_ar{ar_str}"
                            if has_chain_sweep:
                                sample_id += f"_chains{ch_int}"
                            run_id = sample_id

                            points.append(
                                SweepPoint(
                                    index=idx,
                                    utilization=u_flt,
                                    clock_period_ns=c_flt,
                                    chain_count=ch_int,
                                    grid_size=d_grid_size,
                                    ordering=d_ordering,
                                    sample_id=sample_id,
                                    run_id=run_id,
                                    design=d_name,
                                    platform=d_platform,
                                    base_flow_config=d_base_cfg,
                                    aspect_ratio=ar_flt,
                                )
                            )
                            idx += 1
                else:
                    # Cartesian product on d_entry sweep_parameters or self.sweep_parameters
                    d_params = d_entry.get("sweep_parameters") or self.sweep_parameters
                    util_vals = d_params.get("utilization") or d_params.get("target_utilization") or [0.60]
                    if not isinstance(util_vals, list):
                        util_vals = [util_vals]
                    clk_vals = d_params.get("clock_period_ns") or d_params.get("clock_period") or [10.0]
                    if not isinstance(clk_vals, list):
                        clk_vals = [clk_vals]
                    ar_vals = d_params.get("aspect_ratio", [1.0])
                    if not isinstance(ar_vals, list):
                        ar_vals = [ar_vals]

                    chain_vals = d_params.get("chain_count")
                    if chain_vals is None:
                        d_chain = d_entry.get("chain_count")
                        chain_vals = d_chain if d_chain is not None else global_chain_vals
                    if not isinstance(chain_vals, list):
                        chain_vals = [chain_vals]

                    has_chain_sweep = len(chain_vals) > 1 or has_global_chain_sweep

                    for util, clk, ar, ch in itertools.product(util_vals, clk_vals, ar_vals, chain_vals):
                        u_flt = float(util)
                        c_flt = float(clk)
                        ar_flt = float(ar)
                        ch_int = int(ch)

                        u_str = f"{int(round(u_flt * 100)):02d}"
                        c_str = str(c_flt).replace(".", "p")
                        ar_str = str(ar_flt).replace(".", "p")
                        sample_id = f"{d_name}_{d_platform}_u{u_str}_clk{c_str}_ar{ar_str}"
                        if has_chain_sweep:
                            sample_id += f"_chains{ch_int}"
                        run_id = sample_id

                        points.append(
                            SweepPoint(
                                index=idx,
                                utilization=u_flt,
                                clock_period_ns=c_flt,
                                chain_count=ch_int,
                                grid_size=d_grid_size,
                                ordering=d_ordering,
                                sample_id=sample_id,
                                run_id=run_id,
                                design=d_name,
                                platform=d_platform,
                                base_flow_config=d_base_cfg,
                                aspect_ratio=ar_flt,
                            )
                        )
                        idx += 1

            return points

        # Fallback to single-design expansion
        util_vals = self.sweep_parameters.get("utilization") or self.sweep_parameters.get("target_utilization")
        if not util_vals:
            util_vals = [0.60]
        if not isinstance(util_vals, list):
            util_vals = [util_vals]

        clk_vals = self.sweep_parameters.get("clock_period_ns") or self.sweep_parameters.get("clock_period")
        if not clk_vals:
            clk_vals = [10.0]
        if not isinstance(clk_vals, list):
            clk_vals = [clk_vals]

        ar_vals = self.sweep_parameters.get("aspect_ratio", [1.0])
        if not isinstance(ar_vals, list):
            ar_vals = [ar_vals]

        chain_vals = self.sweep_parameters.get("chain_count", global_chain_vals)
        if not isinstance(chain_vals, list):
            chain_vals = [chain_vals]

        has_ar_sweep = len(ar_vals) > 1 or (len(ar_vals) == 1 and ar_vals[0] != 1.0)
        has_chain_sweep = len(chain_vals) > 1 or has_global_chain_sweep

        for util, clk, ar, ch in itertools.product(util_vals, clk_vals, ar_vals, chain_vals):
            u_flt = float(util)
            c_flt = float(clk)
            ar_flt = float(ar)
            ch_int = int(ch)

            u_str = f"{int(round(u_flt * 100)):02d}"
            c_str = str(c_flt).replace(".", "p")
            if has_ar_sweep:
                ar_str = str(ar_flt).replace(".", "p")
                sample_id = f"{self.design}_{self.platform}_u{u_str}_clk{c_str}_ar{ar_str}"
            else:
                sample_id = f"{self.design}_{self.platform}_u{u_str}_clk{c_str}"
            if has_chain_sweep:
                sample_id += f"_chains{ch_int}"
            run_id = sample_id

            points.append(
                SweepPoint(
                    index=idx,
                    utilization=u_flt,
                    clock_period_ns=c_flt,
                    chain_count=ch_int,
                    grid_size=self.grid_size,
                    ordering=self.ordering,
                    sample_id=sample_id,
                    run_id=run_id,
                    design=self.design,
                    platform=self.platform,
                    base_flow_config=self.base_flow_config,
                    aspect_ratio=ar_flt,
                )
            )
            idx += 1

        return points
