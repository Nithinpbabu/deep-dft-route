"""
Cell master classification heuristic for physical feature extraction.
Classifies placed cell instances into sequential vs combinational categories
based on cell library master names (e.g. Nangate45 and standard ASIC PDKs).
"""

from __future__ import annotations

import re
from typing import Dict, Literal

CellCategory = Literal["sequential", "combinational"]

# Sequential master name pattern:
# Matches standard flip-flops (DFF, DFFR, DFFS, SDFF, EDFF, etc.) and latches (LATCH, DLH, DLL)
SEQUENTIAL_PATTERN = re.compile(
    r"^(?:[ES]?DFF[A-Z0-9_]*|LATCH[A-Z0-9_]*|DL[HL][A-Z0-9_]*)",
    re.IGNORECASE,
)

HEURISTIC_DESCRIPTION = (
    "Sequential classification matches master names starting with (case-insensitive): "
    "DFF, DFFR, DFFS, SDFF, EDFF, LATCH, DLH, DLL. "
    "All other cells (logic gates, inverters, clock buffers, muxes) are classified as combinational."
)


class CellClassifier:
    """
    Standard cell classification engine.
    """

    def __init__(self, sequential_pattern: re.Pattern = SEQUENTIAL_PATTERN):
        self.pattern = sequential_pattern
        self._cache: Dict[str, CellCategory] = {}

    def classify(self, master_name: str) -> CellCategory:
        """
        Classifies a cell master name as 'sequential' or 'combinational'.
        """
        if master_name in self._cache:
            return self._cache[master_name]

        cleaned = master_name.strip()
        # Direct regex check
        if self.pattern.search(cleaned):
            cat: CellCategory = "sequential"
        # Secondary keyword check in case of prefix or suffix variation
        elif any(k in cleaned.upper() for k in ("_DFF", "-DFF", "DFFR", "DFFS", "SDFF", "LATCH")):
            cat = "sequential"
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
