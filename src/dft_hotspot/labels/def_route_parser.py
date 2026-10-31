"""
DEF 5.8 Route Parser.
Extracts routed wire segments, vias, layers, and coordinates from DEF NETS sections.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple



@dataclass
class DefRouteSegment:
    """Represents a single continuous routed wire segment."""
    net_name: str
    layer: str
    x1: float
    y1: float
    x2: float
    y2: float
    length_dbu: float
    length_um: float
    orientation: str  # "horizontal", "vertical", "diagonal", "zero"

    @property
    def is_horizontal(self) -> bool:
        return self.orientation == "horizontal"

    @property
    def is_vertical(self) -> bool:
        return self.orientation == "vertical"

    @property
    def is_diagonal(self) -> bool:
        return self.orientation == "diagonal"


@dataclass
class DefVia:
    """Represents a via placed at a layout coordinate."""
    net_name: str
    via_name: str
    layer: str
    x: float
    y: float

from dft_hotspot.features.def_parser import DefDieArea, DefParser


@dataclass
class DefRouteData:
    """Aggregated routed layout data extracted from DEF."""
    segments: List[DefRouteSegment] = field(default_factory=list)
    vias: List[DefVia] = field(default_factory=list)
    routed_nets_count: int = 0
    total_nets_count: int = 0
    dbu_per_micron: int = 1000
    die_area: Optional[DefDieArea] = None
    skipped_diagonal_count: int = 0
    skipped_syntax_count: int = 0
    layer_wirelengths_um: Dict[str, float] = field(default_factory=dict)
    layer_wirelengths_dbu: Dict[str, float] = field(default_factory=dict)
    layer_segment_counts: Dict[str, int] = field(default_factory=dict)
    layer_via_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def total_vias(self) -> int:
        return len(self.vias)

    @property
    def total_wirelength_um(self) -> float:
        return sum(self.layer_wirelengths_um.values())

    @property
    def total_wirelength_dbu(self) -> float:
        return sum(self.layer_wirelengths_dbu.values())


class DefRouteParser:
    """
    Parser for DEF routed wire structures.
    Extracts wire segment coordinates, layers, lengths, and vias from DEF NETS.
    """

    KNOWN_KEYWORDS = {
        "NEW", "ROUTED", "FIXED", "COVER", "USE", "SIGNAL", "POWER", "GROUND",
        "CLOCK", "TIEOFF", "ANALOG", "SCAN", "RESET", "RECT", "TAPER", "TAPERRULE",
        "STYLE", "MASK", "SHIELDNET", "NOSHIELD"
    }

    @classmethod
    def parse_file(cls, def_path: str | Path) -> DefRouteData:
        path = Path(def_path)
        if not path.is_file():
            raise FileNotFoundError(f"DEF file not found: {def_path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return cls.parse_string(content)

    @classmethod
    def parse_string(cls, content: str) -> DefRouteData:
        route_data = DefRouteData()

        # Parse UNITS and DIEAREA using base DefParser logic
        base_data = DefParser.parse_string(content)
        route_data.dbu_per_micron = base_data.dbu_per_micron
        route_data.die_area = base_data.die_area

        # Find NETS section
        nets_match = re.search(r"\bNETS\s+(\d+)\s*;(.*?)\bEND\s+NETS\b", content, re.DOTALL | re.IGNORECASE)
        if not nets_match:
            return route_data

        route_data.total_nets_count = int(nets_match.group(1))
        nets_body = nets_match.group(2)

        # Split individual net entries delimited by ';'
        raw_net_entries = nets_body.split(";")
        for entry in raw_net_entries:
            entry_strip = entry.strip()
            if not entry_strip or not entry_strip.startswith("-"):
                continue

            cls._parse_net_route_entry(entry_strip, route_data)

        # Compute summary statistics by layer
        cls._compute_layer_summaries(route_data)

        return route_data

    @classmethod
    def _parse_net_route_entry(cls, entry: str, route_data: DefRouteData) -> None:
        tokens = entry.split()
        if len(tokens) < 2:
            return

        net_name = tokens[1]

        # Check for ROUTED or FIXED section
        route_match = re.search(r"\+\s*(ROUTED|FIXED)(.*)", entry, re.DOTALL | re.IGNORECASE)
        if not route_match:
            return

        route_data.routed_nets_count += 1
        route_body = route_match.group(2)

        # Split clauses by 'NEW'
        # First clause comes before any 'NEW'
        # Subsequent clauses are preceded by 'NEW'
        clauses = re.split(r"\bNEW\b", route_body, flags=re.IGNORECASE)

        for clause in clauses:
            clause_clean = clause.strip()
            if not clause_clean:
                continue
            cls._parse_route_clause(net_name, clause_clean, route_data)

    @classmethod
    def _parse_route_clause(cls, net_name: str, clause: str, route_data: DefRouteData) -> None:
        # First token is the layer name
        tokens = clause.split(maxsplit=1)
        if not tokens:
            return

        layer = tokens[0]
        remainder = tokens[1] if len(tokens) > 1 else ""

        # Extract tokens: either parenthesized coordinate '( ... )' or word '[A-Za-z0-9_]+'
        item_matches = re.finditer(r"(\([^)]+\)|[^\s()]+)", remainder)

        prev_pt: Optional[Tuple[float, float]] = None

        for m in item_matches:
            item = m.group(1).strip()
            if item.startswith("(") and item.endswith(")"):
                # Coordinate point
                coord_text = item[1:-1].strip()
                parts = coord_text.split()
                if len(parts) < 2:
                    route_data.skipped_syntax_count += 1
                    continue

                # Parse X
                if parts[0] == "*":
                    if prev_pt is None:
                        route_data.skipped_syntax_count += 1
                        continue
                    curr_x = prev_pt[0]
                else:
                    try:
                        curr_x = float(parts[0])
                    except ValueError:
                        route_data.skipped_syntax_count += 1
                        continue

                # Parse Y
                if parts[1] == "*":
                    if prev_pt is None:
                        route_data.skipped_syntax_count += 1
                        continue
                    curr_y = prev_pt[1]
                else:
                    try:
                        curr_y = float(parts[1])
                    except ValueError:
                        route_data.skipped_syntax_count += 1
                        continue

                curr_pt = (curr_x, curr_y)

                # If we have a previous point, form a segment
                if prev_pt is not None:
                    cls._add_segment(
                        net_name=net_name,
                        layer=layer,
                        p1=prev_pt,
                        p2=curr_pt,
                        route_data=route_data,
                    )

                prev_pt = curr_pt

            else:
                # Word token - could be via name
                if item.upper() not in cls.KNOWN_KEYWORDS:
                    if prev_pt is not None:
                        via = DefVia(
                            net_name=net_name,
                            via_name=item,
                            layer=layer,
                            x=prev_pt[0],
                            y=prev_pt[1],
                        )
                        route_data.vias.append(via)
                    else:
                        route_data.skipped_syntax_count += 1

    @classmethod
    def _add_segment(
        cls,
        net_name: str,
        layer: str,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        route_data: DefRouteData,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx == 0.0 and dy == 0.0:
            orient = "zero"
            length_dbu = 0.0
        elif dy == 0.0:
            orient = "horizontal"
            length_dbu = dx
        elif dx == 0.0:
            orient = "vertical"
            length_dbu = dy
        else:
            orient = "diagonal"
            # Euclidean length in DBU
            length_dbu = math.sqrt(dx * dx + dy * dy)
            route_data.skipped_diagonal_count += 1

        dbu = max(route_data.dbu_per_micron, 1)
        length_um = length_dbu / float(dbu)

        segment = DefRouteSegment(
            net_name=net_name,
            layer=layer,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            length_dbu=length_dbu,
            length_um=length_um,
            orientation=orient,
        )
        route_data.segments.append(segment)

    @classmethod
    def _compute_layer_summaries(cls, route_data: DefRouteData) -> None:
        wirelengths_um: Dict[str, float] = {}
        wirelengths_dbu: Dict[str, float] = {}
        seg_counts: Dict[str, int] = {}
        via_counts: Dict[str, int] = {}

        for seg in route_data.segments:
            layer = seg.layer
            wirelengths_um[layer] = wirelengths_um.get(layer, 0.0) + seg.length_um
            wirelengths_dbu[layer] = wirelengths_dbu.get(layer, 0.0) + seg.length_dbu
            seg_counts[layer] = seg_counts.get(layer, 0) + 1

        for via in route_data.vias:
            layer = via.layer
            via_counts[layer] = via_counts.get(layer, 0) + 1

        route_data.layer_wirelengths_um = wirelengths_um
        route_data.layer_wirelengths_dbu = wirelengths_dbu
        route_data.layer_segment_counts = seg_counts
        route_data.layer_via_counts = via_counts
