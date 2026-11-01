"""
Cell master classification heuristic for physical feature extraction.
Classifies placed cell instances into sequential, combinational, or non-functional categories
based on cell library master names (e.g. Nangate45, TSMC/GF 14nm FinFET, and standard ASIC PDKs).
"""

from __future__ import annotations

import re
from typing import Dict, Literal

CellCategory = Literal["sequential", "combinational", "non_functional"]

# Sequential master name pattern:
# Matches standard flip-flops (DFF, DFFR, DFFS, SDFF, EDFF, DRNQ, DSNQ, etc.) and latches (LATCH, DLH, DLL, LL)
SEQUENTIAL_PATTERN = re.compile(
    r"^(?:[ES]?DFF[A-Z0-9_]*|DRNQ[A-Z0-9_]*|DSNQ[A-Z0-9_]*|DQ[A-Z0-9_]*|LATCH[A-Z0-9_]*|DL[HL][A-Z0-9_]*|LL[A-Z0-9_]*)",
    re.IGNORECASE,
)

# Non-functional dummy cell master pattern:
# In commercial technology nodes (e.g. 14nm FinFET), modern PnR tools insert decoupling capacitor
# and filler cells to satisfy DRC, CMP metal density rules, and power-grid integrity.
# These cells (FDCAP*, FILL*, TAPCELL*, DECAP*, WELLTAP*, ENDCAP*) carry ZERO signal pins
# and ZERO signal nets. If not filtered, they artificially inflate cell density in regions with
# zero routing demand, leading to false hotspot alerts and graph node explosion.
DUMMY_PATTERN = re.compile(
    r"^(?:FDCAP|FILL|TAPCELL|DECAP|WELLTAP|ENDCAP)[A-Z0-9_]*",
    re.IGNORECASE,
)

HEURISTIC_DESCRIPTION = (
    "Sequential classification matches master names starting with (case-insensitive): "
    "DFF, DFFR, DFFS, SDFF, EDFF, DRNQ, DSNQ, DQ, LATCH, DLH, DLL, LL. "
    "Non-functional dummy classification matches (case-insensitive): "
    "FDCAP, FILL, TAPCELL, DECAP, WELLTAP, ENDCAP. "
    "All other active logic gates, inverters, clock buffers, and muxes are classified as combinational."
)


class CellClassifier:
    """
    Standard cell classification engine with support for filtering non-functional dummy cells.
    """

    def __init__(
        self,
        sequential_pattern: re.Pattern = SEQUENTIAL_PATTERN,
        dummy_pattern: re.Pattern = DUMMY_PATTERN,
    ):
        self.pattern = sequential_pattern
        self.dummy_pattern = dummy_pattern
        self._cache: Dict[str, CellCategory] = {}
        self._dummy_cache: Dict[str, bool] = {}

    def is_non_functional(self, master_name: str) -> bool:
        """
        Returns True if cell master is a non-functional dummy cell (FDCAP, FILL, TAPCELL, etc.).
        These cells contain zero signal pins and zero signal nets.
        """
        if master_name in self._dummy_cache:
            return self._dummy_cache[master_name]

        cleaned = master_name.strip()
        is_dummy = bool(self.dummy_pattern.search(cleaned))
        self._dummy_cache[master_name] = is_dummy
        return is_dummy

    def is_dummy(self, master_name: str) -> bool:
        """Alias for is_non_functional."""
        return self.is_non_functional(master_name)

    def classify(self, master_name: str) -> CellCategory:
        """
        Classifies a cell master name as 'sequential', 'non_functional', or 'combinational'.
        """
        if master_name in self._cache:
            return self._cache[master_name]

        cleaned = master_name.strip()

        # 1. Non-functional dummy cells check (FDCAP, FILL, TAPCELL, etc.)
        if self.is_non_functional(cleaned):
            cat: CellCategory = "non_functional"
        # 2. Sequential regex check
        elif self.pattern.search(cleaned):
            cat = "sequential"
        # 3. Secondary keyword check in case of prefix or suffix variation
        elif any(k in cleaned.upper() for k in ("_DFF", "-DFF", "DFFR", "DFFS", "SDFF", "LATCH")):
            cat = "sequential"
        # 4. Standard active combinational logic gate
        else:
            cat = "combinational"

        self._cache[master_name] = cat
        return cat

    def is_sequential(self, master_name: str) -> bool:
        return self.classify(master_name) == "sequential"

    def is_combinational(self, master_name: str) -> bool:
        return self.classify(master_name) == "combinational"


# Global default instance
_DEFAULT_CLASSIFIER = CellClassifier()


def classify_cell(master_name: str) -> CellCategory:
    """Classify cell master name using default heuristic."""
    return _DEFAULT_CLASSIFIER.classify(master_name)


def is_sequential(master_name: str) -> bool:
    """Return True if cell master is classified as sequential."""
    return _DEFAULT_CLASSIFIER.is_sequential(master_name)


def is_combinational(master_name: str) -> bool:
    """Return True if cell master is classified as combinational."""
    return _DEFAULT_CLASSIFIER.is_combinational(master_name)


def is_non_functional(master_name: str) -> bool:
    """Return True if cell master is a non-functional dummy cell (FDCAP, FILL, TAPCELL, etc.)."""
    return _DEFAULT_CLASSIFIER.is_non_functional(master_name)


def is_dummy(master_name: str) -> bool:
    """Alias for is_non_functional."""
    return _DEFAULT_CLASSIFIER.is_dummy(master_name)

