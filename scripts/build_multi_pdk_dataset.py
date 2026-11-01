#!/usr/bin/env python3
"""
Builder for Multi-PDK Cross-Technology Benchmark Dataset (Stage 8).
Unifies 45nm planar CMOS (UART, DFT Counter, SPI) and 14nm FinFET (zero-riscy, RISCY)
into an organized dataset with strict out-of-sample splits.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    print("=" * 80)
    print("BUILDING MULTI-PDK DATASET (45nm Planar + 14nm FinFET)")
    print("=" * 80)

    dataset_dir = ROOT_DIR / "data" / "processed" / "datasets" / "multi_pdk_v001"
    samples_dir = dataset_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    expanded_samples_dir = ROOT_DIR / "data" / "processed" / "datasets" / "expanded_v002" / "samples"
    processed_samples_dir = ROOT_DIR / "data" / "processed" / "samples"

    # Define the exact samples to include
    train_45nm = [
        "uart_tx_nangate45_u45_clk3p0_ar1p0_chains1",
        "uart_tx_nangate45_u45_clk3p0_ar1p0_chains2",
        "uart_tx_nangate45_u55_clk3p0_ar1p0_chains1",
        "uart_tx_nangate45_u55_clk3p0_ar1p0_chains2",
        "dft_counter_nangate45_u45_clk3p0_ar1p0_chains1",
        "dft_counter_nangate45_u45_clk3p0_ar1p0_chains2",
        "dft_counter_nangate45_u55_clk3p0_ar1p0_chains1",
        "dft_counter_nangate45_u55_clk3p0_ar1p0_chains2",
    ]

    val_45nm = [
        "uart_tx_nangate45_u65_clk3p0_ar1p0_chains1",
        "dft_counter_nangate45_u65_clk3p0_ar1p0_chains1",
    ]

    test_45nm = [
        "spi_nangate45_u55_clk3p0_ar1p0_chains1",
        "spi_nangate45_u65_clk3p0_ar1p0_chains1",
    ]

    train_14nm = [
        "zero_riscy_14nm_train",
        "zero_riscy_14nm_ar1p0_p1",
        "zero_riscy_14nm_ar1p0_p2",
        "zero_riscy_14nm_ar1p0_p3",
        "zero_riscy_14nm_ar1p5_p0",
        "zero_riscy_14nm_ar1p5_p1",
        "vortex_small_14nm_ar1p0_p1",
        "zero_riscy_14nm_ar1p0_p0_ar",
        "zero_riscy_14nm_ar1p0_p1_ar",
        "zero_riscy_14nm_ar1p0_p4_ap",
        "zero_riscy_14nm_ar1p0_p4_ar",
        "vortex_small_14nm_ar1p0_p1_ar",
        "vortex_small_14nm_ar1p0_p2_ap",
        "nvdla_small_14nm_ar1p0_p1_ap",
    ]

    test_14nm = [
        "riscy_cv32e40p_filtered",
    ]

    all_sample_ids = train_45nm + val_45nm + test_45nm + train_14nm + test_14nm
    print(f"Total samples to organize: {len(all_sample_ids)}")

    # Copy/link sample files
    manifest_records = []
    for sid in all_sample_ids:
        dst = samples_dir / sid
        dst.mkdir(parents=True, exist_ok=True)

        # Source resolution
        if (expanded_samples_dir / sid).is_dir():
            src = expanded_samples_dir / sid
        elif (processed_samples_dir / sid).is_dir():
            src = processed_samples_dir / sid
        else:
            raise FileNotFoundError(f"Could not locate source directory for sample: {sid}")

        # 1. Spatial layout tensor (standardize to 12 channels)
        if (src / "fused_input.npz").is_file():
            shutil.copy2(src / "fused_input.npz", dst / "fused_input.npz")
            fused_d = np.load(src / "fused_input.npz")
            feat_arr = fused_d["fused_features"]
            np.savez_compressed(dst / "features_spatial.npz", features=feat_arr)
        elif (src / "features_spatial.npz").is_file():
            spat_d = np.load(src / "features_spatial.npz")
            feat_arr = spat_d["features"]
            if feat_arr.shape[0] < 12:
                pad = np.zeros((12 - feat_arr.shape[0], feat_arr.shape[1], feat_arr.shape[2]), dtype=feat_arr.dtype)
                feat_arr = np.concatenate([feat_arr, pad], axis=0)
            np.savez_compressed(dst / "features_spatial.npz", features=feat_arr)

        # 2. Routing labels tensor
        if (src / "labels_routing.npz").is_file():
            shutil.copy2(src / "labels_routing.npz", dst / "labels_routing.npz")

        # 3. Circuit netlist graph (standardize to circuit_graph.pt)
        if (src / "circuit_graph.pt").is_file():
            shutil.copy2(src / "circuit_graph.pt", dst / "circuit_graph.pt")
        elif (src / "graph.pt").is_file():
            shutil.copy2(src / "graph.pt", dst / "circuit_graph.pt")

        # 4. Metadata (standardize to metadata.json)
        if (src / "metadata.json").is_file():
            shutil.copy2(src / "metadata.json", dst / "metadata.json")
        elif (src / "design_sample.json").is_file():
            shutil.copy2(src / "design_sample.json", dst / "metadata.json")

        # Clean up any redundant large files to ensure compact package under Colab upload limits
        for redundant in ["graph.pt", "scan_metadata.json", "design_sample.json"]:
            if (dst / redundant).is_file():
                (dst / redundant).unlink()

        # Read metadata to check PDK
        meta_f = dst / "metadata.json"
        pdk = "nangate45" if "nangate45" in sid or "spi" in sid or "uart" in sid or "counter" in sid else "circuitnet_14nm"
        if meta_f.is_file():
            with open(meta_f, "r") as f:
                meta = json.load(f)
            meta["pdk"] = pdk
            with open(meta_f, "w") as f:
                json.dump(meta, f, indent=2)

        manifest_records.append({
            "sample_id": sid,
            "pdk": pdk,
            "sample_dir": f"samples/{sid}",
        })
        print(f"  * Synchronized [{pdk:15s}]: {sid}")

    # Build splits_v001.json
    splits_doc = {
        "schema_version": "splits_v001",
        "dataset_id": "multi_pdk_v001",
        "split_type": "cross_technology_holdout",
        "train": train_45nm + train_14nm,
        "val": val_45nm,
        "test": test_45nm + test_14nm,
        "descriptions": {
            "train_45nm": "45nm Planar (UART + DFT Counter across 45%, 55% utilization)",
            "train_14nm": "14nm FinFET (zero-riscy 2-stage microcontroller core)",
            "val_45nm": "45nm Planar (high utilization 65% runs)",
            "test_45nm": "45nm Planar (SPI controller - completely unseen circuit architecture)",
            "test_14nm": "14nm FinFET (OpenHW CV32E40P 131k-cell core - completely unseen)",
        }
    }
    splits_path = dataset_dir / "splits_v001.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_doc, f, indent=2)
    print(f"\nSaved splits configuration: {splits_path}")

    # Create dataset_manifest.json
    manifest_doc = {
        "dataset_id": "multi_pdk_v001",
        "schema_version": "dataset_manifest_v001",
        "platforms": ["nangate45", "circuitnet_14nm"],
        "sample_count": len(all_sample_ids),
        "samples": manifest_records,
    }
    manifest_path = dataset_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)
    print(f"Saved dataset manifest: {manifest_path}")

    # Build compressed tar.gz archive for Google Colab upload
    archive_path = ROOT_DIR / "data" / "dataset_multi_pdk.tar.gz"
    print(f"\nPackaging dataset into compact archive for Google Colab ({archive_path.name})...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dataset_dir, arcname="multi_pdk_v001")
        tar.add(ROOT_DIR / "src", arcname="src")
        tar.add(ROOT_DIR / "scripts", arcname="scripts")
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: Created {archive_path.name} ({archive_size_mb:.2f} MB)")
    print(f"This lightweight archive contains the complete 45nm + 14nm paired dataset, ready to upload to Colab in seconds!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
