"""
Visualization and preview generation for post-route routing label tensors.
Produces publication-quality multi-panel heatmap figures showing total, horizontal,
vertical wirelength, via count, and metal layer densities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


def generate_routing_preview(
    labels: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str | Path,
    channel_names: Optional[List[str]] = None,
) -> Path:
    """
    Renders an 8-panel heatmap visualization of extracted post-route routing labels.

    Args:
        labels: numpy array of shape (C, H, W)
        metadata: dictionary containing extraction and routing metadata
        output_path: path to write routing_preview.png
        channel_names: optional list of channel names (defaults to metadata['channel_names'])
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if channel_names is None:
        channel_names = metadata.get(
            "channel_names",
            [
                "total_wirelength_density",
                "horizontal_wirelength_density",
                "vertical_wirelength_density",
                "via_count_density",
                "metal1_wirelength_density",
                "metal2_wirelength_density",
                "metal3_wirelength_density",
                "metal4plus_wirelength_density",
            ],
        )

    # 2 rows x 4 columns
    # Row 1: Total, Horizontal, Vertical, Vias
    # Row 2: Metal 1, Metal 2, Metal 3, Metal 4+
    indices = [0, 1, 2, 3, 4, 5, 6, 7]
    cmaps = [
        "turbo",     # Total WL
        "viridis",   # Horizontal WL
        "inferno",   # Vertical WL
        "magma",     # Via count
        "bone",      # Metal 1
        "Blues",     # Metal 2
        "Greens",    # Metal 3
        "copper",    # Metal 4+
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5), facecolor="#14141E")

    design = metadata.get("design", "Design")
    pdk = metadata.get("pdk", "PDK")
    grid_size = metadata.get("grid_size", [64, 64])
    die_area = metadata.get("die_area_um", {})
    die_str = f"{die_area.get('width', 100):.1f} x {die_area.get('height', 100):.1f} µm" if die_area else ""
    r_stats = metadata.get("routing_statistics", {})
    total_wl_um = r_stats.get("total_wirelength_um", 0.0)
    total_vias = r_stats.get("total_vias", 0)

    fig.suptitle(
        f"Post-Route Wirelength & Via Density Labels — {design} ({pdk.upper()}) | "
        f"Grid: {grid_size[0]}x{grid_size[1]} | Die: {die_str} | Total WL: {total_wl_um:.1f} µm | Vias: {total_vias}",
        fontsize=15,
        fontweight="bold",
        color="#F8FAFC",
        y=0.98,
    )

    axes_flat = axes.flatten()

    for ax, idx, cmap in zip(axes_flat, indices, cmaps):
        if idx >= len(labels):
            ax.axis("off")
            continue

        ch_name = channel_names[idx]
        ch_data = labels[idx]

        ax.set_facecolor("#0A0A10")
        im = ax.imshow(
            ch_data,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#94A3B8", labelcolor="#94A3B8")
        cbar.outline.set_edgecolor("#334155")

        formatted_name = ch_name.replace("_", " ").title()
        max_val = float(ch_data.max())
        nonzero = int(np.count_nonzero(ch_data))
        raw_stats = metadata.get("channel_statistics", {}).get(ch_name, {})
        raw_sum = raw_stats.get("raw_sum", float(ch_data.sum()))

        unit_str = "vias" if "via" in ch_name else "µm"
        ax.set_title(
            f"{formatted_name}\n(Norm Max: {max_val:.3f} | Sum: {raw_sum:.1f} {unit_str} | Bins: {nonzero})",
            fontsize=10.5,
            color="#E2E8F0",
            pad=8,
        )

        ax.set_xlabel("Grid X", color="#94A3B8", fontsize=8.5, labelpad=4)
        ax.set_ylabel("Grid Y", color="#94A3B8", fontsize=8.5, labelpad=4)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plt.subplots_adjust(left=0.04, right=0.97, top=0.91, bottom=0.07, wspace=0.25, hspace=0.32)
    fig.savefig(out_file, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return out_file
