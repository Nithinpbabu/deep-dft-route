#!/usr/bin/env python3
"""
Orchestration Script for Stage 3b OpenROAD Native DFT Validation.
Runs DFT validation sequence on sequential smoke designs across target platforms,
capturing logs, command transcripts, and structured validation summaries.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from dft_hotspot.config import ConfigManager
from dft_hotspot.dft import DftValidator, DftValidationReport
from dft_hotspot.utils import setup_logger

logger = setup_logger("validate_dft")


def resolve_synthesized_netlist(design_name: str, pdk_name: str, workspace_root: Path) -> Path:
    """Finds the most recent synthesized netlist from runs directory, or synthesizes if missing."""
    runs_dir = workspace_root / "runs"
    candidates = list(runs_dir.glob(f"{design_name}_{pdk_name}_*"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for run in candidates:
        netlist = run / "scripts" / f"synth_{design_name}.v"
        if netlist.exists():
            logger.info(f"Using synthesized netlist from previous run: {netlist}")
            return netlist

    raise FileNotFoundError(f"No existing synthesized netlist found for {design_name} on {pdk_name}. Please run baseline flow first.")


def run_platform_dft_validation(
    platform_name: str,
    config_path: Path,
    output_root: Path,
    workspace_root: Path
) -> Path:
    """Executes OpenROAD DFT validation flow for a specific platform config."""
    cfg = ConfigManager.load_config(config_path)
    design_name = cfg["design"]["name"]
    top_module = cfg["design"]["top_module"]
    pdk_cfg = cfg["pdk"]
    flow_cfg = cfg.get("flow", {})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"dft_smoke_{platform_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized DFT validation output directory: {run_dir}")

    # Copy baseline logs if available
    runs_dir = workspace_root / "runs"
    baseline_runs = sorted(runs_dir.glob(f"{design_name}_{platform_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if baseline_runs:
        latest_base = baseline_runs[0]
        base_synth_log = latest_base / "logs" / "synth.log"
        base_or_log = latest_base / "logs" / "openroad.log"
        if base_synth_log.exists():
            shutil.copy(base_synth_log, run_dir / "baseline_synthesis.log")
        if base_or_log.exists():
            shutil.copy(base_or_log, run_dir / "baseline_openroad.log")

    # Locate synthesized netlist
    synth_netlist = resolve_synthesized_netlist(design_name, platform_name, workspace_root)

    # Resolve platform asset paths
    liberty_file = ConfigManager.load_config(config_path)["pdk"]["liberty_file"]
    liberty_path = Path(os.path.expandvars(liberty_file))
    if not liberty_path.is_absolute():
        liberty_path = (workspace_root / liberty_path).resolve()

    tech_lef = Path(os.path.expandvars(pdk_cfg["tech_lef"]))
    if not tech_lef.is_absolute():
        tech_lef = (workspace_root / tech_lef).resolve()

    stdcell_lef = Path(os.path.expandvars(pdk_cfg["stdcell_lef"]))
    if not stdcell_lef.is_absolute():
        stdcell_lef = (workspace_root / stdcell_lef).resolve()

    tracks_file = pdk_cfg.get("tracks_file", "")
    if tracks_file:
        tracks_path = Path(os.path.expandvars(tracks_file))
        if not tracks_path.is_absolute():
            tracks_path = (workspace_root / tracks_path).resolve()
        tracks_str = str(tracks_path)
    else:
        tracks_str = ""

    # Output artifact targets
    out_verilog = run_dir / f"{design_name}_scan.v"
    out_def = run_dir / f"{design_name}_scan.def"
    out_odb = run_dir / f"{design_name}_scan.odb"

    tcl_script = workspace_root / "flow" / "openroad" / "dft_validation.tcl"
    if not tcl_script.exists():
        raise FileNotFoundError(f"OpenROAD DFT validation script not found: {tcl_script}")

    # Set up environment variables for Tcl script
    env = os.environ.copy()
    env.update({
        "TECH_LEF": str(tech_lef),
        "STDCELL_LEF": str(stdcell_lef),
        "LIB_FILE": str(liberty_path),
        "VERILOG_NETLIST": str(synth_netlist),
        "TOP_MODULE": str(top_module),
        "CLOCK_PORT": str(cfg["design"].get("clock_port", "clk")),
        "CLOCK_PERIOD": str(cfg["design"].get("clock_period", 10.0)),
        "DIE_AREA": str(flow_cfg.get("die_area", "0 0 100 100")),
        "CORE_AREA": str(flow_cfg.get("core_area", "10 10 90 90")),
        "SITE": str(pdk_cfg.get("site", "FreePDK45_38x28_10R_NP_162NW_34O")),
        "TRACKS_FILE": tracks_str,
        "PIN_HOR_LAYER": str(pdk_cfg.get("pin_horizontal_layer", flow_cfg.get("pin_horizontal_layer", "metal5"))),
        "PIN_VER_LAYER": str(pdk_cfg.get("pin_vertical_layer", flow_cfg.get("pin_vertical_layer", "metal6"))),
        "MAX_CHAINS": "1",
        "MAX_LENGTH": "16",
        "OUT_VERILOG": str(out_verilog),
        "OUT_DEF": str(out_def),
        "OUT_ODB": str(out_odb)
    })

    logger.info(f"Executing OpenROAD DFT validation via {tcl_script}...")
    cmd = ["openroad", "-exit", str(tcl_script)]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        combined_log = proc.stdout + "\n" + proc.stderr
    except Exception as e:
        combined_log = f"Execution error: {str(e)}"

    # Write dft_validation.log
    log_file = run_dir / "dft_validation.log"
    log_file.write_text(combined_log, encoding="utf-8")
    logger.info(f"Wrote log to: {log_file}")

    # Write dft_command_transcript.txt
    transcript_file = run_dir / "dft_command_transcript.txt"
    transcript_lines = [
        "============================================================",
        f" OpenROAD Native DFT Command Transcript - {platform_name.upper()}",
        f" Timestamp: {timestamp}",
        "============================================================",
        ""
    ]
    for line in combined_log.splitlines():
        if "=== DFT_STAGE_" in line or "=== DFT_VALIDATION_" in line or "DFT-" in line or "Scan chain" in line or "SCANCHAINS" in line:
            transcript_lines.append(line)
    transcript_file.write_text("\n".join(transcript_lines), encoding="utf-8")
    logger.info(f"Wrote transcript to: {transcript_file}")

    # Analyze run and write dft_validation_summary.json
    summary = DftValidator.analyze_validation_run(
        platform_name=platform_name,
        design_name=design_name,
        log_text=combined_log,
        liberty_path=liberty_path,
        out_def_path=out_def if out_def.exists() else None,
        out_verilog_path=out_verilog if out_verilog.exists() else None
    )
    summary_path = run_dir / "dft_validation_summary.json"
    summary.save_json(summary_path)
    logger.info(f"Wrote summary to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(f" DFT Validation Summary: {platform_name}")
    print("=" * 60)
    print(f" Platform                   : {summary.platform}")
    print(f" OpenROAD Version           : {summary.openroad_version}")
    print(f" Liberty has test_cell()    : {summary.liberty_has_test_cell}")
    print(f" Commands Attempted         : {', '.join(summary.commands_attempted)}")
    print(f" Commands Succeeded         : {', '.join(summary.commands_succeeded)}")
    print(f" scan_replace Succeeded     : {summary.scan_replace_succeeded}")
    print(f" report_dft_plan Succeeded  : {summary.report_dft_plan_succeeded}")
    print(f" execute_dft_plan Succeeded : {summary.execute_dft_plan_succeeded}")
    print(f" Scan Metadata Available    : {summary.scan_metadata_available}")
    print(f" Generated Scan Ports/Nets  : {summary.generated_scan_ports_or_nets}")
    if summary.failure_command:
        print(f" Failure Command            : {summary.failure_command}")
        print(f" Failure Reason             : {summary.failure_reason}")
    if summary.limitations:
        print(f" Limitations                : {summary.limitations}")
    print("=" * 60 + "\n")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Stage 3b OpenROAD Native DFT Validation")
    parser.add_argument(
        "--platform",
        choices=["sky130hd", "nangate45", "all"],
        default="all",
        help="Target platform to validate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/dft_validation",
        help="Directory to store validation runs (default: data/dft_validation)"
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    output_root = (workspace_root / args.output_dir).resolve()

    platforms = ["sky130hd", "nangate45"] if args.platform == "all" else [args.platform]

    for p in platforms:
        cfg_file = workspace_root / "configs" / "flows" / f"dft_smoke_{p}.yaml"
        if not cfg_file.exists():
            logger.error(f"Config file not found: {cfg_file}")
            continue
        try:
            run_platform_dft_validation(p, cfg_file, output_root, workspace_root)
        except Exception as e:
            logger.error(f"Validation failed for platform {p}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
