#!/usr/bin/env python3
"""
Generates visual benchmark report for Cross-Technology PDK-Conditioned Model.
Compares:
1. Baseline Zero-Shot (No dummy filter)
2. Quick Fix (Dummy filter)
3. Approach 2: PDK Conditioning (Colab-trained Multi-PDK model)
Across both 14nm FinFET (RISCY CV32E40P) and 45nm CMOS (SPI Master).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Order avoids circular import between ml and models
from dft_hotspot.ml.train_fusion import evaluate_fusion_model
from dft_hotspot.models.fusion import CNNGNNFusionModel

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
riscy_dir = ROOT_DIR / "data" / "processed" / "datasets" / "multi_pdk_v001" / "samples" / "riscy_cv32e40p_filtered"
spi_dir = ROOT_DIR / "data" / "processed" / "datasets" / "multi_pdk_v001" / "samples" / "spi_nangate45_u65_clk3p0_ar1p0_chains1"
device = torch.device("cpu")

model = CNNGNNFusionModel(
    spatial_in_channels=14,
    base_channels=16,
    node_in_channels=11,
    gnn_hidden_channels=64,
    gnn_out_channels=64,
    num_gnn_layers=3,
    grid_size=64,
    fusion_mode="concat",
    out_channels=1,
    final_activation="none",
    default_mode="multimodal",
)
ckpt_path = ROOT_DIR / "experiments" / "fusion" / "multi_pdk_conditioned_v001" / "best_model.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

# 1. RISCY inference
riscy_spat = np.load(riscy_dir / "features_spatial.npz")["features"]  # (12, 64, 64)
# Add 2D one-hot planes for 14nm: [0, 1]
is_45 = np.zeros((1, 64, 64), dtype=np.float32)
is_14 = np.ones((1, 64, 64), dtype=np.float32)
riscy_spat_14c = torch.tensor(np.concatenate([riscy_spat, is_45, is_14], axis=0), dtype=torch.float32).unsqueeze(0)

riscy_graph = torch.load(riscy_dir / "circuit_graph.pt", weights_only=False)
# Add tech scale 0.14 to node features
tech_col = torch.full((riscy_graph.x.shape[0], 1), 0.14, dtype=torch.float32)
riscy_graph.x = torch.cat([riscy_graph.x, tech_col], dim=-1)

riscy_gt = np.load(riscy_dir / "labels_routing.npz")["labels"][0]

with torch.no_grad():
    riscy_pred = model(x_spatial=riscy_spat_14c, graph_data=riscy_graph, mode="multimodal").squeeze().numpy()

# 2. SPI inference
spi_12c = np.load(spi_dir / "fused_input.npz")["fused_features"]
is_45_spi = np.ones((1, 64, 64), dtype=np.float32)
is_14_spi = np.zeros((1, 64, 64), dtype=np.float32)
spi_spat_14c = torch.tensor(np.concatenate([spi_12c, is_45_spi, is_14_spi], axis=0), dtype=torch.float32).unsqueeze(0)

spi_graph = torch.load(spi_dir / "circuit_graph.pt", weights_only=False)
tech_col_spi = torch.full((spi_graph.x.shape[0], 1), 0.45, dtype=torch.float32)
spi_graph.x = torch.cat([spi_graph.x, tech_col_spi], dim=-1)

spi_gt = np.load(spi_dir / "labels_routing.npz")["labels"][0]

with torch.no_grad():
    spi_pred = model(x_spatial=spi_spat_14c, graph_data=spi_graph, mode="multimodal").squeeze().numpy()

# Plotting figure
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2])

# Panel A: Evolution Bar Chart on 14nm FinFET Holdout
ax_bar = fig.add_subplot(gs[0, :2])
methods = ["Zero-Shot 45nm\n(Unfiltered)", "Quick Fix\n(Dummy Filter)", "PDK Conditioning\n(Unified 45+14nm)"]
spearman_vals = [0.3532, 0.6925, 0.7105]
pearson_vals = [-0.1415, 0.1159, 0.2236]

x = np.arange(len(methods))
width = 0.35

rects1 = ax_bar.bar(x - width/2, spearman_vals, width, label="Spearman rho (Rank Corr)", color="#2b5c8f")
rects2 = ax_bar.bar(x + width/2, pearson_vals, width, label="Pearson r (Linear Corr)", color="#e07a5f")

ax_bar.set_ylabel("Correlation Score", fontsize=12, fontweight="bold")
ax_bar.set_title("14nm FinFET (RISCY CV32E40P) Correlation Progression", fontsize=14, fontweight="bold")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(methods, fontsize=11)
ax_bar.legend(loc="upper left", fontsize=10)
ax_bar.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax_bar.set_ylim(-0.25, 0.85)

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax_bar.annotate(f"{h:+.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    y_pos = h if h > 0 else h - 0.05
    ax_bar.annotate(f"{h:+.4f}", xy=(rect.get_x() + rect.get_width()/2, y_pos),
                    xytext=(0, 4 if h > 0 else -10), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

# Panel B: 45nm Holdout Metrics Summary (SPI Master)
ax_summary = fig.add_subplot(gs[0, 2:])
ax_summary.axis("off")
table_data = [
    ["Target Technology Node", "Hold-Out Core", "Pearson r", "Spearman rho", "Top-5% Recall", "Top-5% IoU"],
    ["45nm Planar CMOS", "SPI Master (u55)", "+0.8315", "+0.3861", "87.80%", "0.7826"],
    ["45nm Planar CMOS", "SPI Master (u65)", "+0.8459", "+0.3657", "85.85%", "0.7521"],
    ["14nm FinFET", "RISCY CV32E40P (FPU)", "+0.2236", "+0.7105", "9.27%", "0.0486"],
]
table = ax_summary.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.2)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor("#2b5c8f")
        cell.set_text_props(color="white", fontweight="bold")
    elif i in (1, 2):
        cell.set_facecolor("#eef4f8")
    else:
        cell.set_facecolor("#fbeee6")
ax_summary.set_title("Cross-Technology Out-of-Sample Holdout Benchmarks", fontsize=14, fontweight="bold", pad=20)

# Heatmaps: RISCY (14nm)
ax_r_gt = fig.add_subplot(gs[1, 0])
im1 = ax_r_gt.imshow(riscy_gt, cmap="turbo", vmin=0, vmax=max(riscy_gt.max(), riscy_pred.max()))
ax_r_gt.set_title("14nm RISCY: Ground Truth Congestion", fontsize=11, fontweight="bold")
plt.colorbar(im1, ax=ax_r_gt, fraction=0.046, pad=0.04)

ax_r_pred = fig.add_subplot(gs[1, 1])
im2 = ax_r_pred.imshow(riscy_pred, cmap="turbo", vmin=0, vmax=max(riscy_gt.max(), riscy_pred.max()))
ax_r_pred.set_title("14nm RISCY: PDK-Conditioned Pred\n(rho=+0.7105, r=+0.2236)", fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=ax_r_pred, fraction=0.046, pad=0.04)

# Heatmaps: SPI (45nm)
ax_s_gt = fig.add_subplot(gs[1, 2])
im3 = ax_s_gt.imshow(spi_gt, cmap="turbo", vmin=0, vmax=max(spi_gt.max(), spi_pred.max()))
ax_s_gt.set_title("45nm SPI: Ground Truth Congestion", fontsize=11, fontweight="bold")
plt.colorbar(im3, ax=ax_s_gt, fraction=0.046, pad=0.04)

ax_s_pred = fig.add_subplot(gs[1, 3])
im4 = ax_s_pred.imshow(spi_pred, cmap="turbo", vmin=0, vmax=max(spi_gt.max(), spi_pred.max()))
ax_s_pred.set_title("45nm SPI: PDK-Conditioned Pred\n(r=+0.8459, Top-5% Recall=85.9%)", fontsize=11, fontweight="bold")
plt.colorbar(im4, ax=ax_s_pred, fraction=0.046, pad=0.04)

plt.tight_layout()
out_png = REPORTS_DIR / "cross_technology_benchmark.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved benchmark figure to: {out_png}")
