#!/usr/bin/env python3
"""
Batch Feature Extractor for Batch 2 14nm FinFET Training Cores (CircuitNet-N14).
Extracts:
1. 12-channel spatial layout + scan tensors (with dummy filter)
2. 8-channel ground-truth routing wirelength labels
3. Circuit netlist graphs (with dummy filter)
4. Metadata JSON
for 7 new 14nm FinFET layouts across zero-riscy, Vortex GPGPU, and NVDLA AI accelerator.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from dft_hotspot.dft.synthetic_scan import SyntheticScanGenerator
from dft_hotspot.features.def_parser import DefParser
from dft_hotspot.features.extractor import PhysicalFeatureExtractor
from dft_hotspot.features.scan_features import ScanFeatureExtractor
from dft_hotspot.graphs.builder import CircuitGraphBuilder
from dft_hotspot.labels.routing_extractor import RoutingLabelExtractor

DEF_JOBS = [
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "zero-riscy - (OpenHW CV32E20)" / "zero-riscy_freq_200_mp_1_fpu_50_fpa_1.0_p_0_fi_ar_detailedroute.def",
        "sample_id": "zero_riscy_14nm_ar1p0_p0_ar",
        "design": "zero_riscy",
    },
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "zero-riscy - (OpenHW CV32E20)" / "zero-riscy_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar_detailedroute.def",
        "sample_id": "zero_riscy_14nm_ar1p0_p1_ar",
        "design": "zero_riscy",
    },
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "zero-riscy - (OpenHW CV32E20)" / "zero-riscy_freq_200_mp_1_fpu_50_fpa_1.0_p_4_fi_ap_detailedroute.def",
        "sample_id": "zero_riscy_14nm_ar1p0_p4_ap",
        "design": "zero_riscy",
    },
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "zero-riscy - (OpenHW CV32E20)" / "zero-riscy_freq_200_mp_1_fpu_50_fpa_1.0_p_4_fi_ar_detailedroute.def",
        "sample_id": "zero_riscy_14nm_ar1p0_p4_ar",
        "design": "zero_riscy",
    },
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "Vortex-small - (RISC-V GPGPU)" / "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar_detailedroute.def",
        "sample_id": "vortex_small_14nm_ar1p0_p1_ar",
        "design": "vortex_small",
    },
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "Vortex-small - (RISC-V GPGPU)" / "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap_detailedroute.def",
        "sample_id": "vortex_small_14nm_ar1p0_p2_ap",
        "design": "vortex_small",
    },
    {
        "def_path": ROOT_DIR / "input-DEF Datasets" / "nvdla-small - (NVIDIA Deep Learning Accelerator)" / "nvdla-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap_detailedroute.def",
        "sample_id": "nvdla_small_14nm_ar1p0_p1_ap",
        "design": "nvdla_small",
    },
]


def process_def(job: dict) -> None:
    def_path: Path = job["def_path"]
    sample_id: str = job["sample_id"]
    design: str = job["design"]

    if not def_path.is_file():
        print(f"Error: DEF file not found: {def_path}")
        return

    output_dir = ROOT_DIR / "data" / "processed" / "samples" / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"EXTRACTING 14nm FinFET DESIGN: {sample_id}")
    print(f"Source DEF: {def_path.name} ({def_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 80)

    # 1. Parse DEF
    t0 = time.time()
    def_data = DefParser.parse_file(def_path)
    print(f"  * Parsed DEF in {time.time() - t0:.2f}s ({len(def_data.components):,} components, {len(def_data.nets):,} nets)")

    # 2. Extract Physical Features (Dummy Filter Active)
    feat_extractor = PhysicalFeatureExtractor(grid_size=64, filter_dummy_cells=True)
    features_7ch, raw_7ch, meta = feat_extractor.extract_from_def_data(
        def_data=def_data,
        source_run_dir=def_path.parent,
        def_path=def_path,
        design_name=design,
        pdk_name="circuitnet_14nm",
        filter_dummy_cells=True,
    )
    active_cells = meta["design_statistics"]["active_functional_components"]
    filtered_cells = meta["design_statistics"]["filtered_dummy_components"]
    print(f"  * Physical features extracted. Functional cells: {active_cells:,}, Dummy filtered: {filtered_cells:,}")

    # 3. Synthetic Scan Features
    scan_meta = SyntheticScanGenerator.generate(
        def_path=def_path,
        chain_count=4,
        ordering="row_major_by_y_then_x",
        design=design,
        platform="circuitnet_14nm",
    )
    scan_extractor = ScanFeatureExtractor(grid_size=64, die_area=def_data.die_area)
    scan_5ch, raw_scan_5ch, scan_meta_dict = scan_extractor.extract_from_scan_metadata(
        scan_metadata=scan_meta,
        sample_dir=output_dir,
    )
    spatial_12ch = np.concatenate([features_7ch, scan_5ch], axis=0)
    np.savez_compressed(output_dir / "features_spatial.npz", features=spatial_12ch)
    scan_meta.save_json(output_dir / "scan_metadata.json")
    print(f"  * Saved 12-ch spatial layout tensor: {spatial_12ch.shape}")

    # 4. Extract Post-Route Ground-Truth Routing Labels
    label_extractor = RoutingLabelExtractor(grid_size=64)
    label_extractor.process_and_save_sample(
        def_path=def_path,
        output_dir=output_dir,
        source_run_dir=def_path.parent,
        design_name=design,
        pdk_name="circuitnet_14nm",
    )
    print(f"  * Saved 8-ch routing labels")

    # 5. Build Graph
    graph_builder = CircuitGraphBuilder(
        grid_size=64,
        max_net_degree=30,
        include_scan_edges=True,
        bidirectional_scan=True,
        filter_dummy_cells=True,
    )
    circuit_graph = graph_builder.build(
        def_data=def_data,
        scan_metadata=scan_meta,
        design=design,
        sample_id=sample_id,
        pdk="circuitnet_14nm",
        filter_dummy_cells=True,
    )
    pyg_data = circuit_graph.to_pyg()
    torch.save(pyg_data, output_dir / "graph.pt")
    print(f"  * Saved circuit graph: {pyg_data.num_nodes:,} nodes, {pyg_data.num_edges:,} edges")

    # 6. Save design sample metadata
    design_meta = {
        "sample_id": sample_id,
        "design": design,
        "pdk": "circuitnet_14nm",
        "def_file": def_path.name,
        "grid_size": 64,
        "die_area_um": {
            "width": def_data.die_area.width / def_data.dbu_per_micron,
            "height": def_data.die_area.height / def_data.dbu_per_micron,
        },
        "components_count": len(def_data.components),
        "functional_components": active_cells,
        "filtered_dummy_components": filtered_cells,
        "nets_count": len(def_data.nets),
        "spatial_features_file": "features_spatial.npz",
        "routing_labels_file": "labels_routing.npz",
        "graph_file": "graph.pt",
        "scan_metadata_file": "scan_metadata.json",
    }
    with open(output_dir / "design_sample.json", "w", encoding="utf-8") as f:
        json.dump(design_meta, f, indent=2)
    print(f"  * Successfully exported sample to {output_dir}")


def main():
    print("=" * 80)
    print("BATCH 2: EXTRACTING 7 NEW 14nm FinFET BENCHMARKS (CircuitNet-N14)")
    print(f"Targets: zero-riscy (4 layouts), Vortex GPGPU (2 layouts), NVDLA AI accelerator (1 layout)")
    print("=" * 80)

    for i, job in enumerate(DEF_JOBS, 1):
        print(f"\n[{i}/{len(DEF_JOBS)}] Processing {job['sample_id']}...")
        process_def(job)

    print("\n" + "=" * 80)
    print("BATCH 2 FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
