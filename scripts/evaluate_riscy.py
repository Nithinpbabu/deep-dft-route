#!/usr/bin/env python3
"""
Evaluation Pipeline for Real Commercial Design:
PULPino / OpenHW Group CV32E40P (RISCY) 32-bit RISC-V Core

Extracts:
1. 64x64 Spatial Layout & Scan Density Tensors (12 channels)
2. Ground-Truth Post-Route Wirelength Labels (8 channels)
3. Circuit Netlist & Scan Graph (131,670 nodes)
Runs:
4. Multimodal CNN + GNN Neural Inference
5. Detailed Metric Benchmarking against Router Ground Truth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src/ to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# TOP-LEVEL CONFIGURATION VARIABLE
# =============================================================================
# FILTER: Controls whether non-functional dummy cells (FDCAP, FILL, TAPCELL, etc.)
# are filtered out from the 64x64 spatial density tensors and circuit graph nodes.
#
# In commercial 14nm FinFET layouts, ~67% of placed components (88,500+ cells in RISCY)
# are decoupling capacitors, standard cell fillers, and well-tap cells inserted for
# DRC rules and CMP planarization. These cells have ZERO signal pins and ZERO signal nets.
#
# When FILTER = True:
#   - Excludes dummy cells from cell density, combinational density, and pin density channels.
#   - Excludes dummy cells from becoming nodes in the Circuit Graph (avoids GNN dilution).
# Can be overridden via command-line argument:
#   python scripts/evaluate_riscy.py --filter      # Enabled (default)
#   python scripts/evaluate_riscy.py --no-filter   # Disabled (raw DEF with fillers)
FILTER: bool = True

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dft_hotspot.features.def_parser import DefParser
from dft_hotspot.features.extractor import PhysicalFeatureExtractor
from dft_hotspot.features.scan_features import ScanFeatureExtractor
from dft_hotspot.dft.synthetic_scan import SyntheticScanGenerator
from dft_hotspot.labels.routing_extractor import RoutingLabelExtractor
from dft_hotspot.graphs.builder import CircuitGraphBuilder
from dft_hotspot.ml.train_fusion import train_fusion
from dft_hotspot.models.fusion import CNNGNNFusionModel
from dft_hotspot.evaluation.metrics import (
    pearson_corr,
    spearman_corr,
    mae,
    rmse,
    topk_recall,
    iou_topk,
    dice_topk,
)


def main():
    parser = argparse.ArgumentParser(
        description="DEEPDFT-ROUTE: Benchmark Evaluation on Commercial RISC-V Core (OpenHW CV32E40P)"
    )
    # --filter flag: controllable via variable FILTER = True or CLI argument (--filter / --no-filter)
    parser.add_argument(
        "--filter",
        dest="filter_dummy_cells",
        action=argparse.BooleanOptionalAction,
        default=FILTER,
        help="Filter non-functional dummy cells (FDCAP, FILL, TAPCELL) from density and graph tensors (default: True)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (defaults to experiments/fusion/multi_pdk_conditioned_v001/best_model.pt)",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        default=False,
        help="Force re-extraction of spatial features and circuit graph even if cached files exist",
    )
    args = parser.parse_args()
    filter_dummy_cells = args.filter_dummy_cells

    print("=" * 80)
    print(" DEEPDFT-ROUTE: COMMERCIAL DESIGN BENCHMARK EVALUATION")
    print(" Target: OpenHW Group CV32E40P (RISCY) 32-bit RISC-V Core with FPU")
    print(f" Non-Functional Cell Filter: {'ENABLED (Filtering FDCAP, FILL, TAPCELL)' if filter_dummy_cells else 'DISABLED (Raw DEF with Fillers)'}")
    print("=" * 80)

    root_dir = Path(__file__).resolve().parent.parent
    def_path = root_dir / "input-DEF Datasets" / "RISCY - (OpenHW CV32E40P)" / "RISCY_freq_200_mp_1_fpu_50_fpa_1.0_p_0_fi_ap_detailedroute.def"
    tag = "filtered" if filter_dummy_cells else "unfiltered"
    output_sample_dir = root_dir / "data" / "processed" / "samples" / f"riscy_cv32e40p_{tag}"
    output_sample_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not def_path.is_file():
        print(f"[ERROR] DEF file not found: {def_path}", file=sys.stderr)
        return 1

    t0_total = time.time()

    # Check if pre-extracted artifacts exist
    spatial_npz = output_sample_dir / "features_spatial.npz"
    labels_npz = output_sample_dir / "labels_routing.npz"
    graph_pt = output_sample_dir / "circuit_graph.pt"
    scan_json = output_sample_dir / "scan_metadata.json"

    # Reuse routing labels from existing sample dir if available (routing ground-truth does not change)
    existing_labels = root_dir / "data" / "processed" / "samples" / "riscy_cv32e40p_commercial" / "labels_routing.npz"
    if not labels_npz.is_file() and existing_labels.is_file():
        import shutil
        shutil.copy2(existing_labels, labels_npz)
        existing_label_meta = existing_labels.parent / "routing_label_metadata.json"
        if existing_label_meta.is_file():
            shutil.copy2(existing_label_meta, output_sample_dir / "routing_label_metadata.json")

    if not args.force_reextract and spatial_npz.is_file() and labels_npz.is_file() and graph_pt.is_file() and scan_json.is_file():
        print(f"\n[INFO] Loading pre-extracted artifacts from cached sample directory ({output_sample_dir.name})...")
        spatial_12ch = np.load(spatial_npz)["features"]
        labels_routing = np.load(labels_npz)["labels"]
        ground_truth_target = labels_routing[0]
        pyg_data = torch.load(graph_pt, weights_only=False)
        with open(scan_json, "r") as f:
            scan_meta_data = json.load(f)
        total_scan_cells = scan_meta_data.get("statistics", {}).get("total_scan_cells", 9381)
        print(f"  * Loaded Spatial Tensor Shape: {spatial_12ch.shape}")
        print(f"  * Loaded Labels Shape        : {labels_routing.shape}")
        print(f"  * Loaded Graph Nodes         : {pyg_data.num_nodes:,}")
    else:
        # -------------------------------------------------------------------------
        # STAGE 1: Parse DEF & Extract Physical Spatial Layout Features
        # -------------------------------------------------------------------------
        print("\n[1/5] Parsing DEF layout and extracting 64x64 spatial placement features...")
        t0 = time.time()
        def_data = DefParser.parse_file(def_path)
        print(f"  * DEF Parsing complete in {time.time() - t0:.2f}s")
        print(f"  * Total Placed Cells : {len(def_data.components):,}")
        print(f"  * Signal Nets        : {len(def_data.nets):,}")
        print(f"  * Die Area Bounds    : ({def_data.die_area.x_min}, {def_data.die_area.y_min}) to ({def_data.die_area.x_max}, {def_data.die_area.y_max}) DBU")
        print(f"  * Die Dimensions     : {def_data.die_area.width / def_data.dbu_per_micron:.2f} um x {def_data.die_area.height / def_data.dbu_per_micron:.2f} um")

        feat_extractor = PhysicalFeatureExtractor(
            grid_size=64,
            filter_dummy_cells=filter_dummy_cells,
        )
        features_7ch, raw_features_7ch, feat_meta = feat_extractor.extract_from_def_data(
            def_data=def_data,
            source_run_dir=def_path.parent,
            def_path=def_path,
            design_name="riscy_cv32e40p",
            pdk_name="circuitnet_14nm",
            filter_dummy_cells=filter_dummy_cells,
        )
        stats = feat_meta.get("design_statistics", {})
        print(f"  * Active Functional Cells    : {stats.get('active_functional_components', len(def_data.components)):,}")
        print(f"  * Filtered Dummy Cells       : {stats.get('filtered_dummy_components', 0):,}")
        print(f"  * Spatial Tensor Shape       : {features_7ch.shape} (7 channels: cell, seq, comb, pin, blockage, x, y)")

        # -------------------------------------------------------------------------
        # STAGE 2: Synthetic Scan Insertion & Scan Feature Tensor Extraction
        # -------------------------------------------------------------------------
        print("\n[2/5] Synthesizing scan chain architecture & extracting scan features...")
        t0 = time.time()
        scan_meta = SyntheticScanGenerator.generate(
            def_path=def_path,
            chain_count=4,
            ordering="row_major_by_y_then_x",
            design="riscy_cv32e40p",
            platform="circuitnet_14nm",
        )
        print(f"  * Scan Chains Created  : {scan_meta.statistics.chain_count} chains")
        print(f"  * Total Scan Cells     : {scan_meta.statistics.total_scan_cells:,} flip-flops stitched")
        print(f"  * Avg Chain Length     : {scan_meta.statistics.avg_chain_length:.1f} cells/chain")

        scan_extractor = ScanFeatureExtractor(grid_size=64, die_area=def_data.die_area)
        scan_5ch, raw_scan_5ch, scan_meta_dict = scan_extractor.extract_from_scan_metadata(
            scan_metadata=scan_meta,
            sample_dir=output_sample_dir,
        )
        print(f"  * Scan Tensor Shape   : {scan_5ch.shape} (5 channels: cell_density, endpoint, segment, order, wirelength_proxy)")

        # Combine into 12-channel spatial tensor
        spatial_12ch = np.concatenate([features_7ch, scan_5ch], axis=0) # (12, 64, 64)
        np.savez_compressed(output_sample_dir / "features_spatial.npz", features=spatial_12ch)
        scan_meta.save_json(output_sample_dir / "scan_metadata.json")
        total_scan_cells = scan_meta.statistics.total_scan_cells
        print(f"  * Combined Spatial Tensor Shape: {spatial_12ch.shape} (Saved to features_spatial.npz)")

        # -------------------------------------------------------------------------
        # STAGE 3: Extract Post-Route Ground-Truth Routing Labels
        # -------------------------------------------------------------------------
        print("\n[3/5] Extracting ground-truth post-route detailed wirelength density...")
        t0 = time.time()
        if not labels_npz.is_file():
            label_extractor = RoutingLabelExtractor(grid_size=64)
            label_artifacts = label_extractor.process_and_save_sample(
                def_path=def_path,
                output_dir=output_sample_dir,
                source_run_dir=def_path.parent,
                design_name="riscy_cv32e40p",
                pdk_name="circuitnet_14nm",
            )
        labels_routing = np.load(output_sample_dir / "labels_routing.npz")["labels"] # (8, 64, 64)
        ground_truth_target = labels_routing[0] # Channel 0: total_wirelength_density
        print(f"  * Routing Labels Ready (Shape: {labels_routing.shape})")
        print(f"  * Target Total Wirelength Map: Min={ground_truth_target.min():.4f}, Mean={ground_truth_target.mean():.4f}, Max={ground_truth_target.max():.4f}")

        # -------------------------------------------------------------------------
        # STAGE 4: Build Circuit Netlist & Scan Graph
        # -------------------------------------------------------------------------
        print("\n[4/5] Building topological circuit netlist and scan graph...")
        t0 = time.time()
        graph_builder = CircuitGraphBuilder(
            grid_size=64,
            max_net_degree=30,
            include_scan_edges=True,
            bidirectional_scan=True,
            filter_dummy_cells=filter_dummy_cells,
        )
        circuit_graph = graph_builder.build(
            def_data=def_data,
            scan_metadata=scan_meta,
            design="riscy_cv32e40p",
            sample_id=f"riscy_cv32e40p_{tag}",
            pdk="circuitnet_14nm",
            filter_dummy_cells=filter_dummy_cells,
        )
        print(f"  * Graph Built in {time.time() - t0:.2f}s")
        print(f"  * Node Count (Cells) : {circuit_graph.num_nodes:,}")
        print(f"  * Edge Count (Links) : {circuit_graph.num_edges:,}")

        pyg_data = circuit_graph.to_pyg()
        torch.save(pyg_data, output_sample_dir / "circuit_graph.pt")
        circuit_graph.save_npz(output_sample_dir / "circuit_graph.npz")
        print(f"  * Saved circuit_graph.pt and circuit_graph.npz")

    # -------------------------------------------------------------------------
    # STAGE 5: Run Multimodal Neural Inference & Benchmarking
    # -------------------------------------------------------------------------
    print("\n[5/5] Executing Multimodal CNN + GNN neural inference...")
    multi_pdk_ckpt = root_dir / "experiments" / "fusion" / "multi_pdk_conditioned_v001" / "best_model.pt"
    legacy_ckpt = root_dir / "experiments" / "fusion" / "expanded_v002_fusion_primary" / "best_model.pt"

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (multi_pdk_ckpt if multi_pdk_ckpt.is_file() else legacy_ckpt)
    if not checkpoint_path.is_file():
        print(f"[ERROR] Model checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  * Compute Device: {device}")
    print(f"  * Using Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Detect whether model is PDK-conditioned (14 channels / 11 node features) or legacy
    first_weight = state_dict.get("spatial_inc.double_conv.0.weight", None)
    if first_weight is not None and first_weight.shape[1] == 14:
        is_pdk_conditioned = True
        spatial_in_channels = 14
        node_in_channels = 11
    else:
        is_pdk_conditioned = False
        spatial_in_channels = 12
        node_in_channels = 10

    print(f"  * Architecture: {'PDK-Conditioned (14ch spatial, 11ch graph)' if is_pdk_conditioned else 'Standard (12ch spatial, 10ch graph)'}")

    model = CNNGNNFusionModel(
        spatial_in_channels=spatial_in_channels,
        base_channels=16,
        node_in_channels=node_in_channels,
        gnn_hidden_channels=64,
        gnn_out_channels=64,
        num_gnn_layers=3,
        grid_size=64,
        fusion_mode="concat",
        out_channels=1,
        final_activation="none",
        default_mode="multimodal",
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Prepare inputs
    if is_pdk_conditioned:
        # Append 2D one-hot planes: is_45nm = 0, is_14nm = 1
        is_45 = np.zeros((1, 64, 64), dtype=np.float32)
        is_14 = np.ones((1, 64, 64), dtype=np.float32)
        spatial_input_np = np.concatenate([spatial_12ch, is_45, is_14], axis=0)

        # Append tech scale 0.14 to graph node features
        if pyg_data.x.shape[1] == 10:
            tech_col = torch.full((pyg_data.x.shape[0], 1), 0.14, dtype=torch.float32)
            pyg_data.x = torch.cat([pyg_data.x, tech_col], dim=-1)
    else:
        spatial_input_np = spatial_12ch

    x_spatial_t = torch.from_numpy(spatial_input_np).float().unsqueeze(0).to(device) # (1, C, 64, 64)
    pyg_data = pyg_data.to(device)

    # 1. Multimodal Inference
    t0 = time.time()
    with torch.no_grad():
        pred_mm = model(x_spatial=x_spatial_t, graph_data=pyg_data, mode="multimodal")
    inference_time_ms = (time.time() - t0) * 1000.0
    pred_mm_np = pred_mm.squeeze().cpu().numpy()

    # 2. Spatial-Only Ablation Inference
    with torch.no_grad():
        pred_spatial = model(x_spatial=x_spatial_t, graph_data=None, mode="spatial_only")
    pred_spatial_np = pred_spatial.squeeze().cpu().numpy()

    # 3. Circuit GNN-Only Ablation Inference
    with torch.no_grad():
        pred_gnn = model(x_spatial=None, graph_data=pyg_data, mode="graph_only")
    pred_gnn_np = pred_gnn.squeeze().cpu().numpy()

    # Heuristic baseline for comparison (Cell Density)
    pred_heuristic = spatial_12ch[0] # cell_density normalized

    print(f"  * Multimodal Inference Latency: {inference_time_ms:.1f} ms (< 1 second!)")

    # -------------------------------------------------------------------------
    # METRICS EVALUATION
    # -------------------------------------------------------------------------
    def calc_metrics(p: np.ndarray, t: np.ndarray) -> dict:
        return {
            "pearson": float(pearson_corr(p, t)),
            "spearman": float(spearman_corr(p, t)),
            "mae": float(mae(p, t)),
            "rmse": float(rmse(p, t)),
            "recall_top5": float(topk_recall(p, t, percent=0.05)),
            "iou_top5": float(iou_topk(p, t, percent=0.05)),
            "dice_top5": float(dice_topk(p, t, percent=0.05)),
            "recall_top10": float(topk_recall(p, t, percent=0.10)),
            "iou_top10": float(iou_topk(p, t, percent=0.10)),
        }

    metrics_mm = calc_metrics(pred_mm_np, ground_truth_target)
    metrics_spatial = calc_metrics(pred_spatial_np, ground_truth_target)
    metrics_gnn = calc_metrics(pred_gnn_np, ground_truth_target)
    metrics_heuristic = calc_metrics(pred_heuristic, ground_truth_target)

    # Hotspot masks for visualization
    q_gt = np.percentile(ground_truth_target, 95)
    q_pred = np.percentile(pred_mm_np, 95)
    gt_mask = ground_truth_target >= q_gt
    pred_mask = pred_mm_np >= q_pred

    tp = np.logical_and(gt_mask, pred_mask)
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask)
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask))

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS ON COMMERCIAL HOLD-OUT CORE (CV32E40P)")
    print("=" * 80)
    print(f"{'Model Architecture':<30} | {'Pearson r':<10} | {'MAE':<8} | {'RMSE':<8} | {'Top-5% Recall':<14} | {'Top-5% IoU':<10}")
    print("-" * 88)
    print(f"{'Heuristic (Placement Density)':<30} | {metrics_heuristic['pearson']:<10.4f} | {metrics_heuristic['mae']:<8.4f} | {metrics_heuristic['rmse']:<8.4f} | {metrics_heuristic['recall_top5']*100:<13.2f}% | {metrics_heuristic['iou_top5']:<10.4f}")
    print(f"{'Ablation: GraphSAGE GNN Only':<30} | {metrics_gnn['pearson']:<10.4f} | {metrics_gnn['mae']:<8.4f} | {metrics_gnn['rmse']:<8.4f} | {metrics_gnn['recall_top5']*100:<13.2f}% | {metrics_gnn['iou_top5']:<10.4f}")
    print(f"{'Ablation: Spatial CNN Only':<30} | {metrics_spatial['pearson']:<10.4f} | {metrics_spatial['mae']:<8.4f} | {metrics_spatial['rmse']:<8.4f} | {metrics_spatial['recall_top5']*100:<13.2f}% | {metrics_spatial['iou_top5']:<10.4f}")
    print(f"{'DeepDFT-Route Multimodal':<30} | {metrics_mm['pearson']:<10.4f} | {metrics_mm['mae']:<8.4f} | {metrics_mm['rmse']:<8.4f} | {metrics_mm['recall_top5']*100:<13.2f}% | {metrics_mm['iou_top5']:<10.4f}")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # GENERATE 4-PANEL VISUALIZATION (Matching DeepDFT Presentation Style)
    # -------------------------------------------------------------------------
    print("\nGenerating presentation 4-panel visual validation figure...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    fig.suptitle(
        f"Pre-Route DFT Hotspot Prediction vs Ground Truth Router\nDesign: riscy_cv32e40p_14nm_finfet (OpenHW CV32E40P)",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Pre-Route Input (Standard Cell Placement)
    cell_density = spatial_12ch[0]
    im0 = axes[0].imshow(cell_density, cmap="Blues", origin="lower")
    axes[0].set_title("1. Pre-Route Input\n(Standard Cell Placement)", fontsize=11, fontweight="semibold")
    axes[0].set_xlabel("Die Grid X (0-63)")
    axes[0].set_ylabel("Die Grid Y (0-63)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Cell Count")

    # Panel 2: Predicted Congestion (Model Output)
    # Dynamic range normalization for display:
    # Column 0 contains a peripheral IO/power trunk (up to 1.0) which visually compresses
    # the functional core (range 0.00 - 0.087) into the bottom 8% dark blue/purple zone.
    # We clip display vmax to the core's peak range so congestion hotspots glow warm (orange/yellow).
    core_vmax = float(np.percentile(ground_truth_target[:, 1:], 99.5)) * 1.25
    im1 = axes[1].imshow(pred_mm_np, cmap="inferno", origin="lower", vmin=0.0, vmax=core_vmax)
    axes[1].set_title(
        f"2. AI Predicted Congestion\n(Pearson r = {metrics_mm['pearson']:.3f})",
        fontsize=11,
        fontweight="semibold",
    )
    axes[1].set_xlabel("Die Grid X (0-63)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized Wirelength")

    # Panel 3: Ground Truth Post-Route Wirelength (Actual Router Result)
    im2 = axes[2].imshow(ground_truth_target, cmap="inferno", origin="lower", vmin=0.0, vmax=core_vmax)
    axes[2].set_title("3. Actual Router Result\n(Post-Route Detailed Wires)", fontsize=11, fontweight="semibold")
    axes[2].set_xlabel("Die Grid X (0-63)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Actual Wirelength")

    # Panel 4: Hotspot Alert Classification (Top 5%)
    alert_rgb = np.ones((64, 64, 3), dtype=np.float32) * 0.94  # background light gray
    alert_rgb[tp] = [0.12, 0.65, 0.23]   # Green: True Hotspot Caught
    alert_rgb[fn] = [0.85, 0.15, 0.15]   # Red: Missed Hotspot
    alert_rgb[fp] = [0.20, 0.45, 0.85]   # Blue: False Alarm

    axes[3].imshow(alert_rgb, origin="lower")
    axes[3].set_title(
        f"4. Hotspot Detection Alert\n(Top-5% Recall: {metrics_mm['recall_top5'] * 100:.1f}%)",
        fontsize=11,
        fontweight="semibold",
    )
    axes[3].set_xlabel("Die Grid X (0-63)")

    # Legend for Panel 4
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.12, 0.65, 0.23], label=f"True Hotspot Caught ({np.sum(tp)})"),
        Patch(facecolor=[0.85, 0.15, 0.15], label=f"Missed Hotspot ({np.sum(fn)})"),
        Patch(facecolor=[0.20, 0.45, 0.85], label=f"False Alarm ({np.sum(fp)})"),
    ]
    axes[3].legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    fig_path = reports_dir / f"riscy_cv32e40p_evaluation_{tag}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    # Also update the primary report figure
    plt.savefig(reports_dir / "riscy_cv32e40p_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  * Saved figure to: {fig_path}")

    # Save JSON metrics
    metrics_summary = {
        "design": "OpenHW Group CV32E40P (RISCY)",
        "module": "pulpino_top",
        "technology": "CircuitNet 14nm FinFET",
        "filter_dummy_cells": filter_dummy_cells,
        "total_instances": 131670,
        "active_graph_nodes": int(pyg_data.num_nodes) if "pyg_data" in locals() else 43158,
        "total_nets": 44369,
        "total_scan_cells": int(total_scan_cells) if "total_scan_cells" in locals() else 9381,
        "inference_latency_ms": inference_time_ms,
        "metrics_multimodal": metrics_mm,
        "metrics_spatial_only": metrics_spatial,
        "metrics_gnn_only": metrics_gnn,
        "metrics_heuristic": metrics_heuristic,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = reports_dir / f"riscy_cv32e40p_metrics_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    # Also update the primary metrics summary
    with open(reports_dir / "riscy_cv32e40p_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  * Saved metrics summary to: {json_path}")
    print(f"\n[DONE] Full commercial evaluation pipeline completed in {time.time() - t0_total:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
