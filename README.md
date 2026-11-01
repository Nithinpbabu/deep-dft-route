# DeepDFT-Route

**Multimodal Neural Prediction of DFT-Induced Physical Routing Congestion**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GNN-3C2179.svg)](https://pyg.org/)
[![EDA Toolchain](https://img.shields.io/badge/EDA-Yosys%20%7C%20OpenROAD-orange.svg)](https://theopenroadproject.org/)
[![PDK](https://img.shields.io/badge/PDK-Nangate45-green.svg)](https://si2.org/open-cell-library/)
[![Tests](https://img.shields.io/badge/Tests-193%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-orange.svg)](LICENSE)

A source-available, reproducible research framework combining **2D Spatial Layout Convolutions (CNN)** and **Topological Circuit Graph Message Passing (GNN)** to predict physical routing congestion and DRC hotspots caused by **Design-for-Test (DFT)** scan chains prior to detailed routing.

---

## 1. Executive Summary & Research Motivation

Modern digital ASIC flows incorporate dedicated DFT structures—such as scan chains, scan-enable distributions, test clocks, and decompressors—to ensure silicon defect testability. However, scan insertion creates severe physical design stress:
- **Long scan-chain stitch routes** that traverse already congested core regions;
- **Local cell density spikes** from scan-muxed flip-flops (`SDFF`);
- **High fanout routing pressure** on scan-enable (`SE`) and test-clock trees;
- **Routing detours** introduced by sub-optimal scan chain ordering.

Generic congestion prediction models typically ask:
> *"Can we predict routing congestion from standard placement features?"*

**DeepDFT-Route** investigates a narrower, research-critical question:
> ***"Can a multimodal CNN + GNN architecture using early physical design layout tensors and netlist/scan graphs predict post-route physical hotspots specifically attributable to DFT/scan implementation, and generalize inductively to unseen RTL designs?"***

---

## 2. Industry Impact: Why This Matters to Semiconductor Companies

In modern advanced-node ASIC flows, physical design is bottle-necked by late-stage routing iterations. **DeepDFT-Route** acts as a fast **AI surrogate model** that bridges the communication gap between Frontend/DFT engineers and Backend Physical Design (PD) teams.

In a production semiconductor environment, this provides high-value utility in three specific areas:

### A. Fast "Shift-Left" Iteration & Turnaround Time (TAT)
* **Traditional Detailed Route**: Takes **12 to 48+ hours** on high-performance compute clusters for large hierarchical blocks.
* **DeepDFT-Route Inference**: Takes **sub-second to seconds** on a single GPU (or standard workstation CPU).
* **Value to Chip Companies**: Enables true "shift-left" physical design. Physical designers and DFT engineers receive near-instantaneous feedback immediately after placement. If a specific scan partition creates an unroutable congestion hotspot, teams detect and rectify it within minutes rather than discovering it after a weekend-long routing job.

### B. Early DFT Architecture & Scan-Chain Co-Optimization
* In conventional flows, scan chains are stitched based on schematic/logical proximity or simple geometric heuristics without insight into upper-metal routing pressure.
* **Value to Chip Companies**: PD engineers can feed early surrogate congestion maps back into DFT tools to re-order scan chains, adjust scan-chain partitioning across core blocks, or insert test-clock repeaters *before* detailed routing begins.

### C. Compute Farm & EDA License Cost Optimization
* Running commercial PnR suites (Cadence Innovus / Synopsys ICC2 / OpenROAD) at scale consumes substantial compute farm resources and expensive license tokens.
* **Value to Chip Companies**: Acts as a rapid **pre-screening gate** in automated design sweeps. Unroutable placement configurations are pruned early, preventing wasted compute cycles and freeing up EDA tool licenses for viable designs.

---

### ⏱️ Traditional EDA Flow vs. DeepDFT-Route Surrogate Flow

![Enterprise ASIC Workflow: Traditional PnR vs DeepDFT-Route AI Surrogate](docs/assets/enterprise_workflow_comparison.svg)

| Dimension | Traditional Full Detailed Route | DeepDFT-Route (AI Surrogate) |
| :--- | :--- | :--- |
| **Execution Speed** | **12 – 48+ hours** (Multi-core compute cluster) | **< 1 second** (Single workstation / GPU) |
| **Feedback Stage** | **Late Stage** (Post-detailed routing & DRC) | **Shift-Left** (Early global / legal placement) |
| **Failure Penalty** | 2–3 days lost per failed iteration | Seconds to re-adjust placement/scan chains |
| **Compute Footprint**| Heavy CPU/memory farm workload | Lightweight neural inference pass |
| **Primary Output** | Final wire geometry & DRC reports | High-resolution pre-route hotspot heatmaps & alert masks |

---

## 3. Technology Stack

This project is built entirely on free, open-source EDA tools and permissive PDKs without proprietary cloud or tool dependencies.

| Layer | Tool / Framework | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Physical Implementation** | [OpenROAD](https://theopenroadproject.org/) | 26Q4 | Floorplanning, global/detailed placement, CTS, global & detailed routing |
| **Logic Synthesis** | [Yosys](https://yosyshq.net/yosys/) | 0.52 | RTL elaboration and logic mapping to Nangate45 standard cells |
| **PDK** | [Nangate45](https://si2.org/open-cell-library/) | Open Cell Lib | 45nm standard cell PDK (Liberty `.lib`, Tech/Macro `.lef`, track templates) |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | 2.0+ | Convolutional U-Net branches, training loops, metric pipelines |
| **Graph Neural Networks** | [PyTorch Geometric (PyG)](https://pyg.org/) | 2.5+ | Circuit netlist and scan chain graph message passing (GraphSAGE) |
| **Physical Layout Parsing** | Custom Python Engine | DEF 5.8 | High-throughput geometry parsing, wirelength interval discretizer, die binning |
| **Host Environment** | WSL2 Ubuntu | 24.04 / 26.04 | Isolated Linux execution on Windows hosts |

---

## 4. End-to-End System Flow

```mermaid
flowchart TD
    subgraph S1["1. Open-Source EDA Implementation Flow"]
        RTL["RTL Benchmarks<br/>(uart_tx, dft_counter, spi)"] --> SYN["Yosys Synthesis<br/>(Logic Optimization)"]
        SYN --> DFT["DFT / Scan Insertion<br/>(Chains, Muxes, Stitches)"]
        DFT --> PNR["OpenROAD Physical Design<br/>(Floorplan + Placement)"]
    end

    subgraph S2["2. Dual-Modal Feature & Label Extraction (64x64 Die Grid)"]
        PNR --> FEAT_SPATIAL["Pre-Route Spatial Layout Tensors<br/>(7 channels: Cell, Pin, Seq Density)"]
        PNR --> FEAT_GRAPH["Circuit Netlist & Scan Graph<br/>(Cell Nodes + Bidirectional Scan Links)"]
        PNR --> ROUTE["OpenROAD Detailed Route<br/>(Post-Route Ground Truth)"]
        ROUTE --> LABELS["Empirical Routing Labels<br/>(8 channels: Wirelength & Via Densities)"]
    end

    subgraph S3["3. Multimodal CNN + GNN Fusion Architecture"]
        FEAT_SPATIAL --> CNN["Spatial CNN Branch<br/>(U-Net Layout Convolutions)"]
        FEAT_GRAPH --> GNN["Topological GNN Branch<br/>(GraphSAGE Message Passing)"]
        CNN --> FUSION["Cross-Modal Fusion Neck<br/>(Channel Concat + Conv Refinement)"]
        GNN --> FUSION
    end

    subgraph S4["4. Pre-Route Hotspot Prediction & Verification"]
        FUSION --> PRED["Predicted Congestion Heatmap<br/>(Normalized Wirelength)"]
        PRED --> EVAL["Metric Evaluation vs Ground Truth<br/>(Top-5% Hotspot Recall, IoU, Pearson r)"]
        LABELS --> EVAL
    end
```

---

## 5. Actual Generated Outputs & Visual Proofs

The framework outputs spatial prediction maps, discrete hotspot alert masks, and multi-channel physical representations.

### A. Pre-Route Hotspot Prediction vs OpenROAD Router
Below is the evaluation result generated on an unseen holdout test design (`dft_counter` at $u=0.65$ utilization, aspect ratio 1.5):

![Pre-Route DFT Hotspot Prediction vs Ground Truth Router](docs/assets/demo_prediction_comparison.png)

- **Panel 1 (Pre-Route Input)**: Standard cell placement density before global/detailed routing.
- **Panel 2 (AI Predicted Congestion)**: Pre-route wirelength density predicted by DeepDFT-Route (**Pearson $r = 0.915$**).
- **Panel 3 (Actual Router Result)**: Ground-truth post-route detailed wirelength density from OpenROAD.
- **Panel 4 (Hotspot Alert Mask)**: Critical congestion localization (**Green = True Positive Hotspot captured at 92.1% Recall**, Red = Missed, Blue = False Alarm).

### B. Multi-Channel Feature Representation
Layout tensors are extracted into synchronized $64 \times 64$ spatial die grids:

| Input Placement Density | Scan Chain Stitch Ordering | Routed Ground Truth |
| :---: | :---: | :---: |
| ![Placement Density](docs/assets/sample_placement_density.png) | ![Scan Features](docs/assets/sample_scan_features.png) | ![Routing Labels](docs/assets/sample_routing_labels.png) |
| Standard cell instance counts & sequential cell distribution | Scan flip-flop bounding boxes & stitch link routing proxies | Metal layer wirelength & cut via densities |

### C. Uncertainty Calibration & Distribution
The model output demonstrates linear alignment with true continuous wirelength distributions:

<p align="center">
  <img src="docs/assets/calibration_curve.png" alt="Calibration Curve" width="48%" />
  <img src="docs/assets/prediction_target_histograms.png" alt="Histograms" width="48%" />
</p>

---

## 6. Empirical Benchmark Results & Ablation Proofs

### Final Evaluation on Unseen Holdout Test Circuits (`expanded_v002`)
Evaluated strictly on holdout test circuits (`dft_counter` and `spi` at high utilization $u=0.65$) across 36 parameter-perturbed Nangate45 physical layouts. All deep learning models are controlled to the same parameter budget (~484k – 519k parameters) for fair evaluation:

| Model Architecture | Category | Parameters | Test MAE $\downarrow$ | Test RMSE $\downarrow$ | Pearson $r \uparrow$ | Top-5% Recall $\uparrow$ | IoU @ 5% $\uparrow$ | Dice @ 5% $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | Empirical | 0 | 0.0361 | 0.0876 | NaN | 0.00% | 0.0000 | 0.0000 |
| **Heuristic (`simple_weighted_sum`)** | Non-learning | 0 | 0.0125 | 0.0661 | 0.6750 | 57.85% | 0.4070 | 0.5785 |
| **GraphSAGE Baseline (Stage 6e)** | Topological GNN | 42,321 | 0.0131 | 0.0514 | 0.8128 | 68.10% | 0.5163 | 0.6810 |
| **TinyUNet Baseline (Stage 5c)** | Spatial CNN | 484,033 | 0.0095 | 0.0399 | 0.8921 | 83.16% | 0.8316 | 0.9081 |
| **Primary Multimodal Fusion (Concat)** | **CNN + GNN** | **519,281** | **0.0090** | **0.0355** | **0.9151** | **92.11%** | **0.8537** | **0.9211** |
| *Ablation: Spatial CNN Only* | Ablation | 519,281 | 0.0105 | 0.0395 | 0.8929 | 90.07% | 0.8194 | 0.9007 |
| *Ablation: Circuit GNN Only* | Ablation | 519,281 | 0.0078 | 0.0360 | 0.9149 | 91.62% | 0.8453 | 0.9162 |
| *Ablation: Physical-Only (No Scan)* | Ablation | 518,561 | 0.0089 | 0.0385 | 0.8999 | 90.72% | 0.8302 | 0.9072 |
| *Ablation: Gated Spatial Attention* | Ablation | 519,265 | 0.0082 | 0.0366 | 0.9088 | 90.72% | 0.8302 | 0.9072 |

### Key Scientific Findings
1. **Multimodal Synergy**: Primary Multimodal Fusion delivers the highest Pearson correlation (**0.9151**), highest Top-5% Recall (**92.11%**), highest hotspot IoU (**0.8537**), and lowest RMSE (**0.0355**).
2. **Topological Value**: Bypassing the circuit graph message passing branch drops hotspot IoU from **0.8537 to 0.8194** and increases MAE by **+17%**, confirming that graph-level netlist connectivity and scan chain stitch order provide complementary non-local signal to 2D convolutions.
3. **DFT/Scan Contribution**: Bypassing scan feature channels drops Pearson correlation from **0.9151 to 0.8999** and increases RMSE from **0.0355 to 0.0385**, confirming that scan metadata directly improves congestion prediction.
4. **Physical Conservation Law**: Wirelength discretization strictly satisfies continuous spatial conservation:
   $$\sum_{\text{cells}} \text{cell\_wirelength} = \text{continuous\_segment\_length}$$
5. **No Spatial Data Leakage**: Train, validation, and test splits are partitioned strictly by entire RTL design entity (no cross-tile contamination).

---

## 7. Repository Structure

```text
DeepDFT-Route/
├── configs/                  # Physical flow and sweep YAML configurations
│   ├── flows/                # Design flow configs (uart_nangate45.yaml, spi, etc.)
│   └── sweeps/               # Multi-run parameter sweep definitions
├── docs/                     # Visual assets and presentation diagrams
│   └── assets/               # Previews, demo plots, and calibration curves
├── pdk/                      # Open PDK library assets
│   └── nangate45/            # Nangate45 .lib, tech LEF, macro LEF, track templates
├── reports/                  # Final benchmark report and demonstration images
│   ├── demo_prediction_comparison.png
│   └── stage7c_multimodal_ablation_benchmark.md
├── rtl/                      # Open-source Verilog benchmarks
│   ├── dft_smoke/            # Scalable scan counter benchmark (dft_counter.v)
│   ├── spi/                  # SPI controller core (spi.v)
│   └── uart/                 # UART transmitter core (uart_tx.v)
├── scripts/                  # CLI orchestration & extraction entry points
│   ├── run_flow.py           # Synthesis & OpenROAD flow runner
│   ├── extract_features.py   # Spatial layout feature tensor extractor
│   ├── extract_labels.py     # Post-route wirelength & via label extractor
│   ├── extract_design_sample.py # Unified paired sample builder
│   ├── validate_design_sample.py # 7-tier validation contract checker
│   └── show_demo.py          # Interactive 4-panel visual comparison demo CLI
├── src/dft_hotspot/          # Core Python package
│   ├── data/                 # PhysicalDesignSample abstractions & dataset loaders
│   ├── eda/                  # Yosys & OpenROAD script generators and telemetry
│   ├── evaluation/           # Continuous & discrete hotspot metrics (IoU, Dice, Recall)
│   ├── features/             # DEF 5.8 parser, grid mapper, scan chain extractors
│   ├── labels/               # Length-conserving wirelength & cut via discretizer
│   └── models/               # PyTorch CNN (TinyUNet), GNN (GraphSAGE), and Fusion
└── tests/                    # Automated unit & integration test suites
    └── unit/                 # 193 unit tests covering full pipeline (100% pass)
```

---

## 8. Quick Start & Reproducibility Guide

Run all commands inside WSL2 Ubuntu from the repository root:

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Nithinpbabu/deep-dft-route.git
cd deep-dft-route

# Activate virtual environment and install dependencies
python3 -m venv ~/.venv_wsl
source ~/.venv_wsl/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

### 2. Verify Pipeline Integrity
Run the automated unit test suite (193 tests):
```bash
PYTHONPATH=src python3 -m unittest discover tests/unit
```

### 3. Run the Interactive Visual Demonstration
Generate the 4-panel AI vs Ground Truth comparison on an unseen circuit:
```bash
PYTHONPATH=src python3 scripts/show_demo.py \
  --sample-id dft_counter_nangate45_u65_clk3p0_ar1p5_chains1 \
  --output-image reports/demo_prediction_comparison.png
```

### 4. Execute Synthesis & Physical Flow
Dry-run or execute the full OpenROAD flow:
```bash
# Dry run: generate scripts and manifest
PYTHONPATH=src python3 scripts/run_flow.py --config configs/flows/uart_nangate45.yaml --dry-run

# Full physical flow execution (Synthesis + Floorplan + Placement + CTS + Route)
PYTHONPATH=src python3 scripts/run_flow.py --config configs/flows/uart_nangate45.yaml
```

### 5. Extract Paired Layout & Routing Tensors
```bash
PYTHONPATH=src python3 scripts/extract_design_sample.py \
  --run-dir runs/uart_tx_nangate45_20261011_064013 \
  --grid-size 64 \
  --output-dir data/processed/samples/sample_demo
```

---

## 9. License & Usage

Copyright © 2026 Nithin P B.

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

### Permitted Use
- Personal learning, experimentation, and hobby projects.
- Academic and non-commercial research.
- Educational review and recruiter evaluation.
- Modification and redistribution under the terms of the license.

### Commercial Restrictions
- Commercial use, incorporation into commercial products, integration into proprietary EDA tools, or other commercial exploitation is strictly prohibited without a separate commercial license and prior written permission from the author.

### Commercial Licensing & Inquiries
For commercial licensing, enterprise partnerships, or permissions beyond the scope of this license, please contact:
- **Author**: Nithin P B
- **Email**: [pbnitin3@gmail.com](mailto:pbnitin3@gmail.com)
- **Official License Terms**: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
