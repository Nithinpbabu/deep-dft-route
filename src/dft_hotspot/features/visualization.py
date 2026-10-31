"""
Visualization and preview generation for spatial physical feature tensors.
Produces publication-quality heatmap figures showing cell and pin densities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def generate_preview(
    features: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str | Path,
    channel_names: Optional[list[str]] = None,
) -> Path:
    """
    Renders a multi-panel heatmap visualization of extracted spatial features.
    
    Args:
        features: numpy array of shape (C, H, W)
        metadata: dictionary containing design and extraction metadata
        output_path: path to write preview.png
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
                "cell_density",
                "sequential_cell_density",
                "combinational_cell_density",
                "pin_density",
                "blockage_or_macro_density",
                "normalized_x",
                "normalized_y",
            ],
        )

    # We plot the primary physical channels:
    # 1. cell_density (idx 0)
    # 2. sequential_cell_density (idx 1)
    # 3. combinational_cell_density (idx 2)
    # 4. pin_density (idx 3)
    display_indices = [0, 1, 3]  # cell_density, sequential_cell_density, pin_density
    cmaps = ["viridis", "plasma", "magma"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#181824")
    
    design = metadata.get("design", "Design")
    pdk = metadata.get("pdk", "PDK")
    grid_size = metadata.get("grid_size", [64, 64])
    die_area = metadata.get("die_area_um", {})
    die_str = f"{die_area.get('width', 100):.1f} x {die_area.get('height', 100):.1f} µm" if die_area else ""

    fig.suptitle(
        f"DFT Physical Feature Maps — {design} ({pdk.upper()}) | Grid: {grid_size[0]}x{grid_size[1]} | Die: {die_str}",
        fontsize=14,
        fontweight="bold",
        color="#F0F4F8",
        y=0.98,
    )

    for ax, idx, cmap in zip(axes, display_indices, cmaps):
        ch_name = channel_names[idx]
        ch_data = features[idx]

        ax.set_facecolor("#0F0F17")
        im = ax.imshow(
            ch_data,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#A0AEC0", labelcolor="#A0AEC0")
        cbar.outline.set_edgecolor("#4A5568")

        formatted_name = ch_name.replace("_", " ").title()
        max_val = float(ch_data.max())
        nonzero = int(np.count_nonzero(ch_data))
        ax.set_title(
            f"{formatted_name}\n(Max: {max_val:.3f}, Active bins: {nonzero})",
            fontsize=11,
            color="#E2E8F0",
            pad=10,
        )

        ax.set_xlabel("Grid X", color="#A0AEC0", fontsize=9)
        ax.set_ylabel("Grid Y", color="#A0AEC0", fontsize=9)
        ax.tick_params(colors="#A0AEC0")
        for spine in ax.spines.values():
            spine.set_color("#4A5568")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_file, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return out_file
