"""
Robust DEF (Design Exchange Format) 5.8 parser for physical feature extraction.
Extracts die area, placed components, pins, nets, and blockages from DEF text files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict


@dataclass
class DefDieArea:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    dbu_per_micron: int = 1000

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def width_um(self) -> float:
        return self.width / self.dbu_per_micron

    @property
    def height_um(self) -> float:
        return self.height / self.dbu_per_micron


@dataclass
class DefComponent:
    name: str
    master: str
    x: float
    y: float
    orient: str = "N"
    status: str = "PLACED"


@dataclass
class DefPin:
    name: str
    net_name: str
    x: float
    y: float
    direction: str = "SIGNAL"
    layer: Optional[str] = None


@dataclass
class DefNet:
    name: str
    connections: List[Tuple[str, str]] = field(default_factory=list)  # (inst_name, pin_name)


@dataclass
class DefBlockage:
    blockage_type: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class DefData:
    die_area: Optional[DefDieArea] = None
    components: List[DefComponent] = field(default_factory=list)
    pins: List[DefPin] = field(default_factory=list)
    nets: List[DefNet] = field(default_factory=list)
    blockages: List[DefBlockage] = field(default_factory=list)
    dbu_per_micron: int = 1000

    @property
    def total_components(self) -> int:
        return len(self.components)

    @property
    def total_pins(self) -> int:
        return len(self.pins)

    @property
    def total_nets(self) -> int:
        return len(self.nets)

    def get_instance_pin_counts(self) -> Dict[str, int]:
        """
        Count connected pins for each instance based on net connectivity.
        """
        counts: Dict[str, int] = {c.name: 0 for c in self.components}
        for net in self.nets:
            for inst, _ in net.connections:
                if inst in counts:
                    counts[inst] += 1
                elif inst != "PIN":
                    counts[inst] = counts.get(inst, 0) + 1
        return counts

    def get_instance_locations(self) -> Dict[str, Tuple[float, float]]:
        """
        Map instance name to placed coordinate (x, y).
        """
        return {c.name: (c.x, c.y) for c in self.components}

    def get_pin_locations(self) -> Dict[str, Tuple[float, float]]:
        """
        Map top-level pin name to placed coordinate (x, y).
        """
        return {p.name: (p.x, p.y) for p in self.pins}


class DefParser:
    """
    Parser for DEF files tailored for spatial physical feature extraction.
    """

    @classmethod
    def parse_file(cls, def_path: str | Path) -> DefData:
        path = Path(def_path)
        if not path.is_file():
            raise FileNotFoundError(f"DEF file not found: {def_path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return cls.parse_string(content)

    @classmethod
    def parse_string(cls, content: str) -> DefData:
        data = DefData()
        lines = content.splitlines()

        # Regex patterns
        units_pattern = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;", re.IGNORECASE)
        diearea_pattern = re.compile(r"DIEAREA\s+(.*?);", re.IGNORECASE)
        point_pattern = re.compile(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)")

        in_section: Optional[str] = None
        statement_tokens: List[str] = []

        dbu = 1000

        for line in lines:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#"):
                continue

            # Check UNITS
            if in_section is None:
                units_match = units_pattern.search(line_strip)
                if units_match:
                    dbu = int(units_match.group(1))
                    data.dbu_per_micron = dbu
                    continue

                # Check DIEAREA
                diearea_match = diearea_pattern.search(line_strip)
                if diearea_match:
                    points = point_pattern.findall(diearea_match.group(1))
                    if len(points) >= 2:
                        xs = [float(p[0]) for p in points]
                        ys = [float(p[1]) for p in points]
                        data.die_area = DefDieArea(
                            x_min=min(xs),
                            y_min=min(ys),
                            x_max=max(xs),
                            y_max=max(ys),
                            dbu_per_micron=dbu
                        )
                    continue

            # Section headers
            upper_line = line_strip.upper()
            if upper_line.startswith("COMPONENTS "):
                in_section = "COMPONENTS"
                statement_tokens = []
                continue
            elif upper_line.startswith("END COMPONENTS"):
                in_section = None
                continue
            elif upper_line.startswith("PINS "):
                in_section = "PINS"
                statement_tokens = []
                continue
            elif upper_line.startswith("END PINS"):
                in_section = None
                continue
            elif upper_line.startswith("NETS "):
                in_section = "NETS"
                statement_tokens = []
                continue
            elif upper_line.startswith("END NETS"):
                in_section = None
                continue
            elif upper_line.startswith("BLOCKAGES "):
                in_section = "BLOCKAGES"
                statement_tokens = []
                continue
            elif upper_line.startswith("END BLOCKAGES"):
                in_section = None
                continue

            if in_section is None:
                continue

            # Collect tokens for current section statement until ';'
            tokens = line_strip.split()
            statement_tokens.extend(tokens)

            if ";" in tokens or line_strip.endswith(";"):
                stmt = " ".join(statement_tokens)
                statement_tokens = []

                if in_section == "COMPONENTS":
                    comp = cls._parse_component_stmt(stmt)
                    if comp:
                        data.components.append(comp)
                elif in_section == "PINS":
                    pin = cls._parse_pin_stmt(stmt)
                    if pin:
                        data.pins.append(pin)
                elif in_section == "NETS":
                    net = cls._parse_net_stmt(stmt)
                    if net:
                        data.nets.append(net)
                elif in_section == "BLOCKAGES":
                    blk = cls._parse_blockage_stmt(stmt)
                    if blk:
                        data.blockages.append(blk)

        # Update DIEAREA dbu if die_area was parsed before UNITS
        if data.die_area:
            data.die_area.dbu_per_micron = data.dbu_per_micron

        return data

    @classmethod
    def _parse_component_stmt(cls, stmt: str) -> Optional[DefComponent]:
        # Form: - <name> <master> [optional tokens] + PLACED ( <x> <y> ) <orient> ;
        # or + FIXED ( <x> <y> ) <orient> ;
        if not stmt.startswith("-"):
            return None

        # Clean semicolon
        stmt = stmt.rstrip(";").strip()
        tokens = stmt.split()
        if len(tokens) < 3:
            return None

        name = tokens[1]
        master = tokens[2]

        # Search for placement status and coordinates
        # Pattern: \+\s*(PLACED|FIXED|COVER|UNPLACED)(?:\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*([A-Za-z0-9_]+))?
        place_match = re.search(
            r"\+\s*(PLACED|FIXED|COVER|UNPLACED)(?:\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*([A-Za-z0-9_]+))?",
            stmt,
            re.IGNORECASE,
        )
        if place_match:
            status = place_match.group(1).upper()
            if place_match.group(2) is not None and place_match.group(3) is not None:
                x = float(place_match.group(2))
                y = float(place_match.group(3))
                orient = place_match.group(4) or "N"
                return DefComponent(
                    name=name, master=master, x=x, y=y, orient=orient, status=status
                )
            else:
                return DefComponent(
                    name=name, master=master, x=0.0, y=0.0, orient="N", status=status
                )

        # Fallback coordinate search
        coord_match = re.search(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*([A-Za-z0-9_]+)?", stmt)
        if coord_match:
            return DefComponent(
                name=name,
                master=master,
                x=float(coord_match.group(1)),
                y=float(coord_match.group(2)),
                orient=coord_match.group(3) or "N",
                status="PLACED",
            )

        return None

    @classmethod
    def _parse_pin_stmt(cls, stmt: str) -> Optional[DefPin]:
        # Form: - <pin_name> + NET <net_name> ... + PLACED ( <x> <y> ) <orient> ;
        if not stmt.startswith("-"):
            return None

        stmt = stmt.rstrip(";").strip()
        tokens = stmt.split()
        if len(tokens) < 2:
            return None

        name = tokens[1]
        net_name = name

        net_match = re.search(r"\+\s*NET\s+([^\s+;]+)", stmt, re.IGNORECASE)
        if net_match:
            net_name = net_match.group(1)

        dir_match = re.search(r"\+\s*DIRECTION\s+([^\s+;]+)", stmt, re.IGNORECASE)
        direction = dir_match.group(1) if dir_match else "SIGNAL"

        coord_match = re.search(r"\+\s*(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)", stmt, re.IGNORECASE)
        if not coord_match:
            coord_match = re.search(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", stmt)

        x = float(coord_match.group(1)) if coord_match else 0.0
        y = float(coord_match.group(2)) if coord_match else 0.0

        layer_match = re.search(r"\+\s*LAYER\s+([^\s+;]+)", stmt, re.IGNORECASE)
        layer = layer_match.group(1) if layer_match else None

        return DefPin(name=name, net_name=net_name, x=x, y=y, direction=direction, layer=layer)

    @classmethod
    def _parse_net_stmt(cls, stmt: str) -> Optional[DefNet]:
        # Form: - <net_name> ( <inst> <pin> ) ... [routing details] ;
        if not stmt.startswith("-"):
            return None

        stmt = stmt.rstrip(";").strip()
        tokens = stmt.split()
        if len(tokens) < 2:
            return None

        net_name = tokens[1]
        net = DefNet(name=net_name)

        # Stop looking for pin connections once routing definitions start
        # E.g., '+ ROUTED', '+ USE', '+ SHIELDNET'
        route_split = re.split(r"\+\s*(?:ROUTED|FIXED|COVER|NOSHIELD|USE)", stmt, maxsplit=1, flags=re.IGNORECASE)
        pins_part = route_split[0]

        # Match ( <inst> <pin> )
        conn_matches = re.findall(r"\(\s*([^\s()]+)\s+([^\s()]+)\s*\)", pins_part)
        for inst, pin in conn_matches:
            net.connections.append((inst, pin))

        return net

    @classmethod
    def _parse_blockage_stmt(cls, stmt: str) -> Optional[DefBlockage]:
        # Form: - [LAYER <name>] RECT ( <x1> <y1> ) ( <x2> <y2> ) ;
        stmt = stmt.rstrip(";").strip()
        rect_matches = re.findall(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", stmt)
        if len(rect_matches) >= 2:
            x1, y1 = float(rect_matches[0][0]), float(rect_matches[0][1])
            x2, y2 = float(rect_matches[1][0]), float(rect_matches[1][1])
            btype = "PLACEMENT"
            if "LAYER" in stmt.upper():
                btype = "LAYER"
            return DefBlockage(
                blockage_type=btype,
                x_min=min(x1, x2),
                y_min=min(y1, y2),
                x_max=max(x1, x2),
                y_max=max(y1, y2),
            )
        return None
