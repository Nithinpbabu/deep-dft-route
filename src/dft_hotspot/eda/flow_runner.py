"""
EDA Flow Runner and Execution Manager.
Handles workspace preparation, PDK checking, script rendering, EDA environment detection,
execution/blocking logic, and manifest serialization.
"""

import json
import shutil
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from dft_hotspot.config import ConfigManager
from dft_hotspot.eda.script_generator import ScriptGenerator
from dft_hotspot.utils import setup_logger

logger = setup_logger("flow_runner")


class FlowRunner:
    """Manages EDA flow execution, dry-runs, and manifest tracking."""

    @staticmethod
    def check_tool_availability() -> Dict[str, Any]:
        """
        Checks native and WSL availability of required EDA tools (yosys, openroad).
        Returns telemetry dict including status and list of missing tools.
        """
        tools = ["yosys", "openroad"]
        status = {}
        missing = []

        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"
        is_wsl_linux = False
        if is_linux:
            try:
                proc_ver = Path("/proc/version").read_text(encoding="utf-8").lower()
                if "microsoft" in proc_ver or "wsl" in proc_ver:
                    is_wsl_linux = True
            except Exception:
                pass

        has_wsl = False
        if is_windows and shutil.which("wsl"):
            try:
                res = subprocess.run(["wsl", "--status"], capture_output=True, text=True, timeout=5)
                stdout = res.stdout.replace("\x00", "")
                stderr = res.stderr.replace("\x00", "")
                out = stdout + stderr
                if "not installed" not in out.lower():
                    has_wsl = True
            except Exception:
                has_wsl = False

        if is_windows:
            exec_env = "windows"
        elif is_wsl_linux:
            exec_env = "wsl2_linux"
        else:
            exec_env = "native_linux"

        status["wsl_available"] = has_wsl if is_windows else True
        status["os_platform"] = platform.system()
        status["execution_environment"] = exec_env

        for tool in tools:
            tool_found = False
            location = "MISSING"

            native_path = shutil.which(tool)
            if native_path:
                tool_found = True
                location = f"native:{native_path}"
            elif is_windows and has_wsl:
                try:
                    res = subprocess.run(["wsl", tool, "-version"], capture_output=True, text=True, timeout=5)
                    stdout = res.stdout.replace("\x00", "")
                    if res.returncode == 0 and "not installed" not in stdout.lower():
                        tool_found = True
                        location = f"wsl:wsl {tool}"
                except Exception:
                    pass

            status[tool] = location
            if not tool_found:
                missing.append(tool)

        status["missing_tools"] = missing
        status["eda_ready"] = (len(missing) == 0)
        return status

    @classmethod
    def run_flow(
        cls,
        config_path: str | Path,
        dry_run: bool = False,
        output_root: Optional[str | Path] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes or dry-runs an EDA flow according to configuration.
        Always creates run directory and writes manifest.json regardless of execution status.
        """
        config_file = Path(config_path).resolve()
        cfg = ConfigManager.load_config(config_file)

        valid_cfg, cfg_errors = ConfigManager.validate_flow_config(cfg)
        if not valid_cfg:
            raise ValueError(f"Invalid flow configuration in '{config_file}': {', '.join(cfg_errors)}")

        design_name = cfg["design"]["name"]
        pdk_name = cfg["pdk"]["name"]
        backend = cfg.get("flow", {}).get("backend", "openroad_standalone")

        # Determine workspace root and output directory
        workspace_root = None
        for p in [config_file] + list(config_file.parents):
            if (p / "pyproject.toml").is_file() or (p / "pdk").is_dir():
                workspace_root = p
                break
        if workspace_root is None:
            workspace_root = config_file.parents[2] if len(config_file.parents) >= 3 else Path.cwd()
        if output_root is None:
            output_root = workspace_root / "runs"
        else:
            output_root = Path(output_root).resolve()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            run_dir = output_root / run_name
        else:
            run_dir = output_root / f"{design_name}_{pdk_name}_{timestamp}"

        scripts_dir = run_dir / "scripts"
        rtl_dir = run_dir / "rtl"
        logs_dir = run_dir / "logs"
        reports_dir = run_dir / "reports"

        for d in [scripts_dir, rtl_dir, logs_dir, reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized run directory: {run_dir}")

        # Copy RTL files
        copied_rtl = []
        for rtl_rel in cfg["design"]["rtl_files"]:
            src_rtl = workspace_root / rtl_rel
            if not src_rtl.exists():
                src_rtl = Path(rtl_rel)

            if src_rtl.exists():
                dest_rtl = rtl_dir / src_rtl.name
                shutil.copyfile(src_rtl, dest_rtl)
                copied_rtl.append(str(dest_rtl.relative_to(run_dir)))
            else:
                logger.warning(f"RTL file not found: {src_rtl}")

        # Render scripts based on selected backend
        generated_scripts = []
        synth_script_path = None
        or_script_path = None
        if backend == "openroad_standalone":
            synth_script_path = ScriptGenerator.generate_yosys_synth_tcl(cfg, scripts_dir / "synth.tcl")
            or_script_path = ScriptGenerator.generate_openroad_tcl(cfg, scripts_dir / "openroad.tcl")
            generated_scripts.append(str(synth_script_path.relative_to(run_dir)))
            generated_scripts.append(str(or_script_path.relative_to(run_dir)))
        elif backend == "openroad_flow_scripts":
            orfs_cfg = ScriptGenerator.generate_orfs_config(cfg, scripts_dir / "config.mk")
            generated_scripts.append(str(orfs_cfg.relative_to(run_dir)))
        else:
            logger.warning(f"Unknown backend '{backend}', default rendered standalone Tcl scripts.")
            synth_script_path = ScriptGenerator.generate_yosys_synth_tcl(cfg, scripts_dir / "synth.tcl")
            generated_scripts.append(str(synth_script_path.relative_to(run_dir)))

        # Validate PDK files existence
        pdk_valid, missing_pdk_files = ConfigManager.validate_pdk_files(cfg["pdk"], workspace_root=workspace_root)

        # Check tool availability
        tool_status = cls.check_tool_availability()

        # Determine status and failure reason using canonical status taxonomy
        status = "unknown"
        failure_reason = None
        dry_run_completed = False
        synthesis_completed = False

        if dry_run:
            status = "dry_run"
            dry_run_completed = True
            logger.info("Dry-run requested. Rendered scripts and prepared run directory.")
        else:
            if not pdk_valid:
                status = "blocked_missing_pdk_assets"
                failure_reason = f"Execution blocked: Missing PDK physical assets ({', '.join(missing_pdk_files)})"
                logger.warning(f"Run blocked: {failure_reason}")
            elif tool_status["yosys"] == "MISSING":
                status = "blocked_missing_eda_env"
                failure_reason = "Execution blocked: Missing required Yosys synthesis binary."
                logger.warning(f"Run blocked: {failure_reason}")
            else:
                # Run Yosys Synthesis
                logger.info(f"Starting Yosys synthesis execution using script {synth_script_path.name}...")
                yosys_cmd = ["yosys", "-s", synth_script_path.name]
                try:
                    res = subprocess.run(
                        yosys_cmd,
                        cwd=scripts_dir,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    synth_log_file = logs_dir / "synth.log"
                    synth_log_file.write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")

                    if res.returncode != 0:
                        status = "failed"
                        err_snippet = res.stderr.strip().split("\n")[-1] if res.stderr else "Non-zero exit code"
                        failure_reason = f"Yosys synthesis failed: {err_snippet}"
                        logger.error(failure_reason)
                    else:
                        synthesis_completed = True
                        logger.info("Yosys synthesis completed successfully.")

                        # Check OpenROAD physical design tool availability
                        if tool_status["openroad"] == "MISSING":
                            status = "blocked_missing_eda_env"
                            failure_reason = "Yosys synthesis completed successfully (netlist saved). OpenROAD physical design tool is missing for P&R step."
                            logger.warning(f"Run blocked: {failure_reason}")
                        else:
                            # Run OpenROAD P&R
                            logger.info(f"Starting OpenROAD physical design execution using script {or_script_path.name}...")
                            or_cmd = ["openroad", "-exit", or_script_path.name]
                            or_res = subprocess.run(
                                or_cmd,
                                cwd=scripts_dir,
                                capture_output=True,
                                text=True,
                                timeout=600
                            )
                            or_log_file = logs_dir / "openroad.log"
                            or_log_file.write_text(or_res.stdout + "\n" + or_res.stderr, encoding="utf-8")

                            if or_res.returncode == 0:
                                status = "success"
                                logger.info("OpenROAD physical design flow completed successfully.")
                            else:
                                status = "failed"
                                err_snippet = or_res.stderr.strip().split("\n")[-1] if or_res.stderr else "Non-zero exit code"
                                failure_reason = f"OpenROAD physical design execution failed: {err_snippet}"
                                logger.error(failure_reason)
                except Exception as e:
                    status = "failed"
                    failure_reason = f"EDA execution error: {str(e)}"
                    logger.error(failure_reason, exc_info=True)

        # Build Manifest
        manifest = {
            "design": design_name,
            "pdk": pdk_name,
            "flow_backend": backend,
            "status": status,
            "dry_run": dry_run,
            "dry_run_completed": dry_run_completed,
            "synthesis_completed": synthesis_completed,
            "timestamp": timestamp,
            "run_dir": str(run_dir),
            "tool_availability": tool_status,
            "missing_tools": tool_status["missing_tools"],
            "pdk_validation": {
                "valid": pdk_valid,
                "missing_files": missing_pdk_files
            },
            "copied_rtl": copied_rtl,
            "generated_scripts": generated_scripts,
            "failure_reason": failure_reason
        }

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Manifest written to: {manifest_path}")

        return manifest
