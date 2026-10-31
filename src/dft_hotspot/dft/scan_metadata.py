"""
Structured Scan Metadata Extraction from OpenROAD Native DFT Outputs.

Extracts DEF SCANCHAINS and enriches them with placement coordinates (DBU and microns),
cell masters, scan ports, and DFT architect/stitch configuration into a validated
machine-readable representation (schema version: scan_metadata_v001).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class NoScanChainsFoundError(Exception):
    """Raised when no SCANCHAINS block is found in the target DEF file."""
    pass


@dataclass
class ScanCell:
    """Individual scan cell within a scan chain."""
    index: int
    instance: str
    scan_in_pin: Optional[str] = None
    scan_out_pin: Optional[str] = None
    master: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    placed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "instance": self.instance,
            "scan_in_pin": self.scan_in_pin,
            "scan_out_pin": self.scan_out_pin,
            "master": self.master,
            "x": self.x,
            "y": self.y,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "placed": self.placed,
        }


@dataclass
class ScanEndpoint:
    """Start or stop endpoint definition for a scan chain."""
    type: str  # "PIN" or "CELL"
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "name": self.name}


@dataclass
class ScanChain:
    """Structured representation of a single stitched scan chain."""
    chain_id: str
    start: ScanEndpoint
    stop: ScanEndpoint
    partition: str = "default"
    cell_count: int = 0
    cells: List[ScanCell] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "start": self.start.to_dict(),
            "stop": self.stop.to_dict(),
            "partition": self.partition,
            "cell_count": self.cell_count,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass
class ScanConfig:
    """Configuration parameters supplied to or reported by OpenROAD DFT."""
    max_chains: Optional[int] = None
    max_length: Optional[int] = None
    clock_mixing: Optional[str] = None
    scan_enable_name_pattern: Optional[str] = None
    scan_in_name_pattern: Optional[str] = None
    scan_out_name_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_chains": self.max_chains,
            "max_length": self.max_length,
            "clock_mixing": self.clock_mixing,
            "scan_enable_name_pattern": self.scan_enable_name_pattern,
            "scan_in_name_pattern": self.scan_in_name_pattern,
            "scan_out_name_pattern": self.scan_out_name_pattern,
        }


@dataclass
class SyntheticConfig:
    """Configuration parameters for synthetic proxy scan-chain generation."""
    chain_count: int = 1
    ordering: str = "row_major_by_y_then_x"
    source_cell_filter: str = "sequential_master_heuristic"
    notes: List[str] = field(
        default_factory=lambda: [
            "Synthetic proxy generated because Nangate45 Liberty lacks OpenROAD-compatible test_cell annotations."
        ]
    )
    partition_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "chain_count": self.chain_count,
            "ordering": self.ordering,
            "source_cell_filter": self.source_cell_filter,
            "notes": list(self.notes),
        }
        if self.partition_method is not None:
            d["partition_method"] = self.partition_method
        return d


@dataclass
class ScanPorts:
    """Top-level scan ports identified in design netlist and DEF."""
    scan_enable: List[str] = field(default_factory=list)
    scan_in: List[str] = field(default_factory=list)
    scan_out: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_enable": self.scan_enable,
            "scan_in": self.scan_in,
            "scan_out": self.scan_out,
        }


@dataclass
class ScanStatistics:
    """Aggregate statistics for scan chains and placed scan cells."""
    chain_count: int = 0
    total_scan_cells: int = 0
    min_chain_length: int = 0
    max_chain_length: int = 0
    avg_chain_length: float = 0.0
    placed_scan_cells: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_count": self.chain_count,
            "total_scan_cells": self.total_scan_cells,
            "min_chain_length": self.min_chain_length,
            "max_chain_length": self.max_chain_length,
            "avg_chain_length": self.avg_chain_length,
            "placed_scan_cells": self.placed_scan_cells,
        }


@dataclass
class ScanMetadata:
    """Complete scan metadata container conforming to scan_metadata_v001."""
    schema_version: str = "scan_metadata_v001"
    source: str = "openroad_dft"
    design: str = ""
    platform: str = ""
    source_dir: str = ""
    def_path: str = ""
    netlist_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scan_config: Optional[ScanConfig] = None
    synthetic_config: Optional[SyntheticConfig] = None
    scan_ports: ScanPorts = field(default_factory=ScanPorts)
    chains: List[ScanChain] = field(default_factory=list)
    statistics: ScanStatistics = field(default_factory=ScanStatistics)
    limitations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source == "openroad_dft" and self.scan_config is None and self.synthetic_config is None:
            self.scan_config = ScanConfig()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "design": self.design,
            "platform": self.platform,
            "source_dir": self.source_dir,
            "def_path": self.def_path,
            "netlist_path": self.netlist_path,
            "created_at": self.created_at,
        }
        if self.synthetic_config is not None:
            d["synthetic_config"] = self.synthetic_config.to_dict()
        if self.scan_config is not None and self.source != "synthetic_proxy":
            d["scan_config"] = self.scan_config.to_dict()
        d.update({
            "scan_ports": self.scan_ports.to_dict(),
            "chains": [chain.to_dict() for chain in self.chains],
            "statistics": self.statistics.to_dict(),
            "limitations": self.limitations,
        })
        return d

    def save_json(self, output_path: Path | str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    @classmethod
    def load_json(cls, json_path: Path | str) -> "ScanMetadata":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config_data = data.get("scan_config")
        scan_config = ScanConfig(**config_data) if config_data else None

        synth_data = data.get("synthetic_config")
        synthetic_config = SyntheticConfig(**synth_data) if synth_data else None

        ports_data = data.get("scan_ports", {})
        stats_data = data.get("statistics", {})

        chains: List[ScanChain] = []
        for c_dict in data.get("chains", []):
            start = ScanEndpoint(**c_dict.get("start", {"type": "PIN", "name": "unknown"}))
            stop = ScanEndpoint(**c_dict.get("stop", {"type": "PIN", "name": "unknown"}))
            cells = [ScanCell(**cell) for cell in c_dict.get("cells", [])]
            chains.append(
                ScanChain(
                    chain_id=c_dict.get("chain_id", "chain_0"),
                    start=start,
                    stop=stop,
                    partition=c_dict.get("partition", "default"),
                    cell_count=c_dict.get("cell_count", len(cells)),
                    cells=cells,
                )
            )

        return cls(
            schema_version=data.get("schema_version", "scan_metadata_v001"),
            source=data.get("source", "openroad_dft"),
            design=data.get("design", ""),
            platform=data.get("platform", ""),
            source_dir=data.get("source_dir", ""),
            def_path=data.get("def_path", ""),
            netlist_path=data.get("netlist_path"),
            created_at=data.get("created_at", ""),
            scan_config=scan_config,
            synthetic_config=synthetic_config,
            scan_ports=ScanPorts(**ports_data),
            chains=chains,
            statistics=ScanStatistics(**stats_data),
            limitations=data.get("limitations", []),
        )

    def to_summary_text(self) -> str:
        lines = [
            "=" * 60,
            " SCAN METADATA SUMMARY",
            "=" * 60,
            f"Schema Version:     {self.schema_version}",
            f"Source:             {self.source}",
            f"Design:             {self.design}",
            f"Platform:           {self.platform}",
            f"DEF Path:           {self.def_path}",
            f"Netlist Path:       {self.netlist_path or 'N/A'}",
        ]
        if self.synthetic_config:
            lines.append(f"Synthetic Ordering: {self.synthetic_config.ordering}")
            lines.append(f"Cell Filter:        {self.synthetic_config.source_cell_filter}")
            if self.synthetic_config.partition_method:
                lines.append(f"Partition Method:   {self.synthetic_config.partition_method}")
        lines.extend([
            f"Chains Extracted:   {self.statistics.chain_count}",
            f"Total Scan Cells:   {self.statistics.total_scan_cells}",
            f"Placed Scan Cells:  {self.statistics.placed_scan_cells}",
            f"Average Chain Len:  {self.statistics.avg_chain_length}",
            f"Scan In Ports:      {', '.join(self.scan_ports.scan_in) or 'None'}",
            f"Scan Out Ports:     {', '.join(self.scan_ports.scan_out) or 'None'}",
            f"Scan Enable Ports:  {', '.join(self.scan_ports.scan_enable) or 'None'}",
            "-" * 60,
            "Chains Detail:",
        ])
        for c in self.chains:
            lines.append(
                f"  - [{c.chain_id}] {c.cell_count} cells | "
                f"Start: {c.start.type} {c.start.name} -> "
                f"Stop: {c.stop.type} {c.stop.name} (Partition: {c.partition})"
            )
            if c.cells:
                first = c.cells[0]
                last = c.cells[-1]
                lines.append(f"    First Cell: {first.instance} (master: {first.master}, placed: {first.placed})")
                lines.append(f"    Last Cell:  {last.instance} (master: {last.master}, placed: {last.placed})")
        if self.limitations:
            lines.append("-" * 60)
            lines.append("Limitations:")
            for lim in self.limitations:
                lines.append(f"  * {lim}")
        lines.append("=" * 60)
        return "\n".join(lines)


from dft_hotspot.features.def_parser import DefData, DefParser


class ScanMetadataExtractor:
    """
    Parser and extractor for DEF SCANCHAINS, placement enrichment, and scan ports.
    """

    @classmethod
    def parse_def_scanchains(cls, def_content: str, allow_missing: bool = False) -> List[ScanChain]:
        """
        Parses DEF text for SCANCHAINS section and extracts ordered ScanChain objects.
        Preserves cell sequence exactly as written in DEF.
        """
        scanchains_match = re.search(
            r"SCANCHAINS\s+(\d+)\s*;(.*?)END\s+SCANCHAINS",
            def_content,
            re.DOTALL | re.IGNORECASE,
        )
        if not scanchains_match:
            if allow_missing:
                return []
            raise NoScanChainsFoundError("No SCANCHAINS block found in DEF content.")

        body = scanchains_match.group(2)
        # Chain statements begin with '- <chain_name>' and terminate with ';'
        # Split body into statements ending with ';'
        chain_stmts = re.findall(r"-\s+([a-zA-Z0-9_]+)\s*(.*?);", body, re.DOTALL)

        chains: List[ScanChain] = []
        for chain_id, stmt_body in chain_stmts:
            start_ep = cls._parse_endpoint(stmt_body, "START")
            stop_ep = cls._parse_endpoint(stmt_body, "STOP")
            partition = cls._parse_partition(stmt_body)
            cells = cls._parse_ordered_cells(stmt_body)

            chains.append(
                ScanChain(
                    chain_id=chain_id,
                    start=start_ep,
                    stop=stop_ep,
                    partition=partition,
                    cell_count=len(cells),
                    cells=cells,
                )
            )

        return chains

    @classmethod
    def _parse_endpoint(cls, stmt: str, kind: str) -> ScanEndpoint:
        """Parses + START or + STOP endpoint from chain statement."""
        # Pattern: + START [PIN] <name> [<pin>]
        pattern = rf"\+\s*{kind}\s+(?:(PIN)\s+)?([^\s;+]+)(?:\s+([^\s;+]+))?"
        m = re.search(pattern, stmt, re.IGNORECASE)
        if not m:
            return ScanEndpoint(type="PIN", name="unknown")

        has_pin = m.group(1) is not None
        name1 = m.group(2)
        name2 = m.group(3)

        if has_pin:
            return ScanEndpoint(type="PIN", name=name1)
        if name2:
            return ScanEndpoint(type="CELL", name=f"{name1}/{name2}")
        return ScanEndpoint(type="PIN", name=name1)

    @classmethod
    def _parse_partition(cls, stmt: str) -> str:
        """Parses + PARTITION parameter from chain statement."""
        m = re.search(r"\+\s*PARTITION\s+([^\s;+]+)", stmt, re.IGNORECASE)
        return m.group(1) if m else "default"

    @classmethod
    def _parse_ordered_cells(cls, stmt: str) -> List[ScanCell]:
        """
        Parses ordered cells from + ORDERED (or + FLOATING) section.
        Preserves exact order and parses optional ( IN <pin> ) and ( OUT <pin> ) tuples.
        """
        # Match from + ORDERED / + FLOATING until next + property or end of statement
        ordered_match = re.search(
            r"\+\s*(?:ORDERED|FLOATING)\s+(.*?)(?=\+\s*[A-Z]+|$)",
            stmt,
            re.DOTALL | re.IGNORECASE,
        )
        if not ordered_match:
            return []

        cells_text = ordered_match.group(1)
        # Normalize parentheses for robust tokenization
        norm_text = cells_text.replace("(", " ( ").replace(")", " ) ")
        tokens = norm_text.split()

        cells: List[ScanCell] = []
        i = 0
        idx = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("+"):
                break
            inst_name = tok
            i += 1

            in_pin: Optional[str] = None
            out_pin: Optional[str] = None

            # Check for trailing ( IN <pin> ) and/or ( OUT <pin> )
            while i < len(tokens) and tokens[i] == "(":
                i += 1  # consume '('
                pin_dir = tokens[i].upper() if i < len(tokens) else ""
                i += 1  # consume dir
                pin_name = tokens[i] if i < len(tokens) else ""
                i += 1  # consume pin name
                if i < len(tokens) and tokens[i] == ")":
                    i += 1  # consume ')'

                if pin_dir == "IN":
                    in_pin = pin_name
                elif pin_dir == "OUT":
                    out_pin = pin_name

            cells.append(
                ScanCell(
                    index=idx,
                    instance=inst_name,
                    scan_in_pin=in_pin,
                    scan_out_pin=out_pin,
                )
            )
            idx += 1

        return cells

    @classmethod
    def enrich_cells_from_components(
        cls, chains: List[ScanChain], def_data: DefData
    ) -> None:
        """
        Enriches scan cells with placement coordinates and master cell names from DefData COMPONENTS.
        """
        comp_map = {c.name: c for c in def_data.components}
        dbu = float(def_data.dbu_per_micron or 1000)

        for chain in chains:
            for cell in chain.cells:
                comp = comp_map.get(cell.instance)
                if comp is not None:
                    cell.master = comp.master
                    if comp.status.upper() in ("PLACED", "FIXED", "COVER"):
                        cell.x = int(comp.x) if comp.x.is_integer() else comp.x
                        cell.y = int(comp.y) if comp.y.is_integer() else comp.y
                        cell.x_um = round(comp.x / dbu, 4)
                        cell.y_um = round(comp.y / dbu, 4)
                        cell.placed = True
                    else:
                        cell.placed = False
                else:
                    cell.placed = False

    @classmethod
    def extract_scan_ports(
        cls,
        verilog_path: Optional[Path | str] = None,
        def_path: Optional[Path | str] = None,
        chains: Optional[List[ScanChain]] = None,
    ) -> ScanPorts:
        """
        Extracts scan_enable, scan_in, and scan_out ports from Verilog netlist, DEF PINS, or chain endpoints.
        """
        scan_enables: set[str] = set()
        scan_ins: set[str] = set()
        scan_outs: set[str] = set()

        # 1. Parse Verilog netlist ports if available
        if verilog_path and Path(verilog_path).is_file():
            content = Path(verilog_path).read_text(encoding="utf-8", errors="ignore")
            # Extract top module port declarations
            # Look for port identifier names matching scan_*
            for m in re.findall(r"\b(scan_enable_\w+|scan_enable)\b", content):
                scan_enables.add(m)
            for m in re.findall(r"\b(scan_in_\w+|scan_in)\b", content):
                scan_ins.add(m)
            for m in re.findall(r"\b(scan_out_\w+|scan_out)\b", content):
                scan_outs.add(m)

        # 2. Check DEF PINS if available and netlist didn't yield all ports
        if def_path and Path(def_path).is_file():
            def_text = Path(def_path).read_text(encoding="utf-8", errors="ignore")
            pins_match = re.search(r"PINS\s+\d+\s*;(.*?)END\s+PINS", def_text, re.DOTALL | re.IGNORECASE)
            if pins_match:
                pins_body = pins_match.group(1)
                for pin_stmt in re.findall(r"-\s+([^\s+;]+)", pins_body):
                    if pin_stmt.startswith("scan_enable"):
                        scan_enables.add(pin_stmt)
                    elif pin_stmt.startswith("scan_in"):
                        scan_ins.add(pin_stmt)
                    elif pin_stmt.startswith("scan_out"):
                        scan_outs.add(pin_stmt)

        # 3. Check chains start/stop pins
        if chains:
            for chain in chains:
                if chain.start.type == "PIN" and chain.start.name.startswith("scan_in"):
                    scan_ins.add(chain.start.name)
                # If no dedicated scan_out port was found, record chain stop pin if it is a PIN
                if not scan_outs and chain.stop.type == "PIN":
                    scan_outs.add(chain.stop.name)

        return ScanPorts(
            scan_enable=sorted(list(scan_enables)),
            scan_in=sorted(list(scan_ins)),
            scan_out=sorted(list(scan_outs)),
        )

    @classmethod
    def extract_scan_config(cls, dir_path: Path | str) -> ScanConfig:
        """
        Extracts scan configuration parameters from OpenROAD logs or transcripts if available.
        """
        p = Path(dir_path)
        log_files = [
            p / "dft_validation.log",
            p / "dft_command_transcript.txt",
            p / "baseline_openroad.log",
        ]

        text = ""
        for lf in log_files:
            if lf.is_file():
                text += "\n" + lf.read_text(encoding="utf-8", errors="ignore")

        config = ScanConfig()
        if not text:
            return config

        # Scan Architect Config regex
        max_len_m = re.search(r"Max Length:\s*(\d+)", text)
        if max_len_m:
            config.max_length = int(max_len_m.group(1))

        max_chains_m = re.search(r"Max Chains:\s*(\d+)", text)
        if max_chains_m:
            config.max_chains = int(max_chains_m.group(1))

        clock_mix_m = re.search(r"Clock Mixing:\s*([^\r\n*]+)", text)
        if clock_mix_m:
            config.clock_mixing = clock_mix_m.group(1).strip()

        # Scan Stitch Config regex
        en_pat_m = re.search(r"Scan Enable Name Pattern:\s*'([^']*)'", text)
        if en_pat_m:
            config.scan_enable_name_pattern = en_pat_m.group(1)

        in_pat_m = re.search(r"Scan In Name Pattern:\s*'([^']*)'", text)
        if in_pat_m:
            config.scan_in_name_pattern = in_pat_m.group(1)

        out_pat_m = re.search(r"Scan Out Name Pattern:\s*'([^']*)'", text)
        if out_pat_m:
            config.scan_out_name_pattern = out_pat_m.group(1)

        return config

    @classmethod
    def extract_from_directory(
        cls,
        dft_dir: Path | str,
        def_path: Optional[Path | str] = None,
        netlist_path: Optional[Path | str] = None,
        allow_missing_scanchains: bool = False,
    ) -> ScanMetadata:
        """
        Extracts complete ScanMetadata from a DFT validation artifact directory.
        """
        dir_p = Path(dft_dir)
        if not dir_p.is_dir():
            raise FileNotFoundError(f"Directory not found: {dft_dir}")

        # Locate DEF file
        if def_path:
            actual_def_path = Path(def_path)
        else:
            candidates = list(dir_p.glob("*_scan.def")) + list(dir_p.glob("*.def"))
            if not candidates:
                raise FileNotFoundError(f"No DEF file found in {dir_p}")
            actual_def_path = candidates[0]

        # Locate Verilog netlist
        if netlist_path:
            actual_netlist_path: Optional[Path] = Path(netlist_path)
        else:
            v_candidates = list(dir_p.glob("*_scan.v")) + list(dir_p.glob("*.v"))
            actual_netlist_path = v_candidates[0] if v_candidates else None

        # Read DEF content and parse COMPONENTS via DefParser
        def_text = actual_def_path.read_text(encoding="utf-8", errors="replace")
        def_data = DefParser.parse_string(def_text)

        # Detect design and platform
        design_name = ""
        design_m = re.search(r"DESIGN\s+([^\s;]+)", def_text)
        if design_m:
            design_name = design_m.group(1)

        platform_name = ""
        summary_json = dir_p / "dft_validation_summary.json"
        if summary_json.is_file():
            try:
                with open(summary_json, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    platform_name = s_data.get("platform", "")
                    if not design_name:
                        design_name = s_data.get("design", "")
            except Exception:
                pass

        if not platform_name:
            if "sky130" in str(dir_p).lower():
                platform_name = "sky130hd"
            elif "nangate" in str(dir_p).lower():
                platform_name = "nangate45"
            else:
                platform_name = "unknown"

        # Parse chains
        chains = cls.parse_def_scanchains(def_text, allow_missing=allow_missing_scanchains)

        # Enrich with placement
        cls.enrich_cells_from_components(chains, def_data)

        # Scan ports
        scan_ports = cls.extract_scan_ports(
            verilog_path=actual_netlist_path,
            def_path=actual_def_path,
            chains=chains,
        )

        # Scan config
        scan_config = cls.extract_scan_config(dir_p)

        # Statistics
        total_cells = sum(len(c.cells) for c in chains)
        chain_lengths = [len(c.cells) for c in chains]
        placed_cells = sum(sum(1 for cell in c.cells if cell.placed) for c in chains)
        stats = ScanStatistics(
            chain_count=len(chains),
            total_scan_cells=total_cells,
            min_chain_length=min(chain_lengths) if chain_lengths else 0,
            max_chain_length=max(chain_lengths) if chain_lengths else 0,
            avg_chain_length=round(sum(chain_lengths) / len(chains), 4) if chain_lengths else 0.0,
            placed_scan_cells=placed_cells,
        )

        # Limitations
        limitations = [
            "Extracted from OpenROAD DEF SCANCHAINS; no ATPG patterns generated.",
        ]
        has_functional_stop = any(
            c.stop.type == "PIN" and not c.stop.name.startswith("scan_out")
            for c in chains
        )
        if has_functional_stop:
            limitations.append(
                "Scan-out stop may terminate at functional output if OpenROAD DFT uses existing output pin."
            )
        if not chains and allow_missing_scanchains:
            limitations.append("No SCANCHAINS block found in DEF file (negative control or unstitched run).")

        return ScanMetadata(
            schema_version="scan_metadata_v001",
            source="openroad_dft",
            design=design_name or "unknown",
            platform=platform_name,
            source_dir=str(dir_p).replace("\\", "/"),
            def_path=str(actual_def_path).replace("\\", "/"),
            netlist_path=str(actual_netlist_path).replace("\\", "/") if actual_netlist_path else None,
            scan_config=scan_config,
            scan_ports=scan_ports,
            chains=chains,
            statistics=stats,
            limitations=limitations,
        )
