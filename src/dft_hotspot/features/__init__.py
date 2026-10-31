"""
Spatial physical feature extraction module.
Provides DEF parsing, cell classification, spatial grid discretization,
tensor construction, and visualization utilities.
"""

from dft_hotspot.features.cell_classifier import (
    CellClassifier,
    classify_cell,
    is_combinational,
    is_sequential,
)
from dft_hotspot.features.def_parser import (
    DefBlockage,
    DefComponent,
    DefData,
    DefDieArea,
    DefNet,
    DefParser,
    DefPin,
)
from dft_hotspot.features.extractor import (
    CHANNEL_NAMES,
    PhysicalFeatureExtractor,
    compute_net_bounding_box_density,
)
from dft_hotspot.features.grid_mapper import GridMapper
from dft_hotspot.features.scan_features import (
    SCAN_CHANNEL_NAMES,
    ScanFeatureExtractor,
    generate_scan_preview,
)
from dft_hotspot.features.visualization import generate_preview

__all__ = [
    "DefParser",
    "DefData",
    "DefDieArea",
    "DefComponent",
    "DefPin",
    "DefNet",
    "DefBlockage",
    "CellClassifier",
    "classify_cell",
    "is_sequential",
    "is_combinational",
    "GridMapper",
    "PhysicalFeatureExtractor",
    "compute_net_bounding_box_density",
    "CHANNEL_NAMES",
    "generate_preview",
    "ScanFeatureExtractor",
    "SCAN_CHANNEL_NAMES",
    "generate_scan_preview",
]
