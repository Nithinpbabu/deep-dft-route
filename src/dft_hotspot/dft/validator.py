"""
OpenROAD Native DFT Flow Validation Analyzer and Summary Abstraction.
Extracts scan chains, command outcomes, failure reasons, and structured evidence.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class DftValidationReport:
    """Structured report documenting the outcome of an OpenROAD DFT validation run."""
    platform: str
    design: str
    openroad_version: str
    liberty_has_test_cell: bool
    commands_attempted: List[str] = field(default_factory=list)
    commands_succeeded: List[str] = field(default_factory=list)
    failure_command: Optional[str] = None
    failure_reason: Optional[str] = None
    scan_replace_succeeded: bool = False
    report_dft_plan_succeeded: bool = False
    execute_dft_plan_succeeded: bool = False
    scan_metadata_available: bool = False
    generated_scan_ports_or_nets: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load_json(cls, path: Path | str) -> "DftValidationReport":
        in_path = Path(path)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class DftValidator:
    """Parses OpenROAD logs, DEF artifacts, and Liberty files for DFT evidence."""

    @staticmethod
    def check_liberty_test_cell(liberty_path: Path | str) -> bool:
        """Inspects Liberty file for test_cell () blocks and scan signal attributes."""
        p = Path(liberty_path)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return "test_cell ()" in content or "test_cell()" in content or "signal_type : \"test_scan" in content
        except Exception:
            return False

    @staticmethod
    def extract_openroad_version(log_text: str) -> str:
        """Extracts OpenROAD version string from log."""
        m = re.search(r"OpenROAD\s+([^\r\n]+)", log_text)
        if m:
            return m.group(1).strip()
        return "unknown"

    @staticmethod
    def parse_def_scanchains(def_path: Path | str) -> Dict[str, Any]:
        """
        Parses DEF file for SCANCHAINS section.
        Returns dict with has_scanchains, chain_count, and list of chain names.
        """
        p = Path(def_path)
        if not p.exists():
            return {"has_scanchains": False, "chain_count": 0, "chains": []}

        content = p.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"SCANCHAINS\s+(\d+)\s*;(.*?)END\s+SCANCHAINS", content, re.DOTALL)
        if not match:
            return {"has_scanchains": False, "chain_count": 0, "chains": []}

        count = int(match.group(1))
        body = match.group(2)
        chain_names = re.findall(r"-\s+([a-zA-Z0-9_]+)", body)
        return {
            "has_scanchains": True,
            "chain_count": count,
            "chains": chain_names
        }

    @staticmethod
    def parse_verilog_scan_ports(verilog_path: Path | str) -> List[str]:
        """Finds generated scan ports (scan_in_*, scan_out_*, scan_enable_*) in Verilog netlist."""
        p = Path(verilog_path)
        if not p.exists():
            return []

        content = p.read_text(encoding="utf-8", errors="ignore")
        ports = set(re.findall(r"\b(scan_in_\w+|scan_out_\w+|scan_enable_\w+)\b", content))
        return sorted(list(ports))

    @classmethod
    def analyze_validation_run(
        cls,
        platform_name: str,
        design_name: str,
        log_text: str,
        liberty_path: Path | str,
        out_def_path: Optional[Path | str] = None,
        out_verilog_path: Optional[Path | str] = None
    ) -> DftValidationReport:
        """Analyzes OpenROAD validation run output and builds structured DftValidationReport."""
        or_version = cls.extract_openroad_version(log_text)
        has_test_cell = cls.check_liberty_test_cell(liberty_path)

        commands_attempted = [
            "report_dft_config",
            "set_dft_config",
            "scan_replace",
            "report_dft_plan",
            "execute_dft_plan"
        ]
        commands_succeeded: List[str] = []

        # Analyze report_dft_config
        if "DFT Config Report:" in log_text:
            commands_succeeded.append("report_dft_config")

        # Analyze set_dft_config
        if "Scan Architect Config:" in log_text and "- Max Length: 16" in log_text:
            commands_succeeded.append("set_dft_config")

        # Analyze scan_replace
        # DFT-0002 indicates failure to replace cells due to missing scan equivalent
        scan_replace_failed = "[WARNING DFT-0002]" in log_text or "[ERROR DFT-0002]" in log_text
        scan_replace_succeeded = ("=== DFT_STAGE_END: SCAN_REPLACE ===" in log_text) and not scan_replace_failed
        if scan_replace_succeeded:
            commands_succeeded.append("scan_replace")

        # Analyze report_dft_plan
        # Looking for Number of chains > 0
        chain_count_match = re.search(r"Number of chains:\s*(\d+)", log_text)
        chains_found = int(chain_count_match.group(1)) if chain_count_match else 0
        report_dft_plan_succeeded = chains_found > 0
        if report_dft_plan_succeeded:
            commands_succeeded.append("report_dft_plan")

        # Analyze execute_dft_plan
        # execute_dft_plan succeeds if plan reported chains and executed without throwing error
        execute_dft_plan_succeeded = False
        if report_dft_plan_succeeded:
            if "=== DFT_STAGE_END: EXECUTE_DFT_PLAN ===" in log_text:
                if "ERROR" not in log_text.split("=== DFT_STAGE_BEGIN: EXECUTE_DFT_PLAN ===")[-1].split("=== DFT_STAGE_END: EXECUTE_DFT_PLAN ===")[0]:
                    execute_dft_plan_succeeded = True
                    commands_succeeded.append("execute_dft_plan")

        # Scan metadata & artifacts
        scan_ports: List[str] = []
        if out_verilog_path:
            scan_ports.extend(cls.parse_verilog_scan_ports(out_verilog_path))

        def_info = {"has_scanchains": False, "chains": []}
        if out_def_path:
            def_info = cls.parse_def_scanchains(out_def_path)
            if def_info["has_scanchains"] and not scan_ports:
                scan_ports = def_info["chains"]

        scan_metadata_available = def_info["has_scanchains"] or (len(scan_ports) > 0)

        # Failure detection
        failure_command = None
        failure_reason = None
        limitations: List[str] = []

        if not has_test_cell:
            limitations.append(f"Platform '{platform_name}' Liberty library lacks test_cell() annotations required by OpenROAD native scan_replace.")

        if scan_replace_failed:
            failure_command = "scan_replace"
            failure_reason = "No scan equivalent lib cell found (Liberty lacks test_cell annotations with test_scan_in/enable/out attributes)."
        elif not report_dft_plan_succeeded and chains_found == 0:
            failure_command = "report_dft_plan"
            failure_reason = "Reported 0 scan chains because no flip-flops were identified or replaced as scan cells."
        elif not execute_dft_plan_succeeded and len(commands_succeeded) < len(commands_attempted):
            failure_command = "execute_dft_plan"
            failure_reason = "Execute DFT plan failed to stitch scan chains."

        if platform_name == "nangate45" and scan_replace_failed:
            limitations.append("Nangate45 native scan insertion blocked; requires Stage 3d synthetic scan metadata or Liberty test_cell patching.")

        return DftValidationReport(
            platform=platform_name,
            design=design_name,
            openroad_version=or_version,
            liberty_has_test_cell=has_test_cell,
            commands_attempted=commands_attempted,
            commands_succeeded=commands_succeeded,
            failure_command=failure_command,
            failure_reason=failure_reason,
            scan_replace_succeeded=scan_replace_succeeded,
            report_dft_plan_succeeded=report_dft_plan_succeeded,
            execute_dft_plan_succeeded=execute_dft_plan_succeeded,
            scan_metadata_available=scan_metadata_available,
            generated_scan_ports_or_nets=scan_ports,
            limitations=limitations
        )
