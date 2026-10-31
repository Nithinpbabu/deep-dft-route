"""Reporting and benchmark synthesis modules for DFT hotspot prediction."""

from dft_hotspot.reporting.stage4_benchmark import (
    build_stage4_summary_dict,
    generate_stage4_benchmark_report,
    load_stage4_artifacts,
    render_stage4_benchmark_markdown,
)

__all__ = [
    "load_stage4_artifacts",
    "build_stage4_summary_dict",
    "render_stage4_benchmark_markdown",
    "generate_stage4_benchmark_report",
]
