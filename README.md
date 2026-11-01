# DeepDFT-Route

**Multimodal Neural Prediction of DFT-Induced Physical Routing Congestion**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GNN-3C2179.svg)](https://pyg.org/)
[![EDA Toolchain](https://img.shields.io/badge/EDA-Yosys%20%7C%20OpenROAD%20%7C%20Innovus-orange.svg)](https://theopenroadproject.org/)
[![PDK](https://img.shields.io/badge/PDK-Nangate45%20%7C%20CircuitNet--14nm-green.svg)](https://circuitnet.github.io/)
[![Commercial Holdout](https://img.shields.io/badge/Holdout%20Core-OpenHW%20CV32E40P%20(RISCY)-purple.svg)](https://www.openhwgroup.org/)
[![Cloud Runner](https://img.shields.io/badge/Cloud%20Runner-Google%20Colab%20Pro%20(T4)-yellow.svg)]()
[![Pearson r](https://img.shields.io/badge/Pearson%20r-%2B0.987-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-orange.svg)](LICENSE)

A source-available, reproducible research framework combining **2D Spatial Layout Convolutions (CNN)** and **Topological Circuit Graph Message Passing (GNN)** to predict physical routing congestion and DRC hotspots caused by **Design-for-Test (DFT)** scan chains prior to detailed routing, supporting both **45nm Planar CMOS** and **14nm FinFET** technology nodes.

---

## 1. Executive Summary & Research Motivation

Modern digital ASIC flows incorporate dedicated DFT structures—such as scan chains, scan-enable distributions, test clocks, and decompressors—to ensure silicon defect testability. However, scan insertion creates severe physical design stress:
- **Long scan-chain stitch routes** that traverse already congested core regions;
- **Local cell density spikes** from scan-muxed flip-flops (`SDFF`);
- **High fanout routing pressure** on scan-enable (`SE`) and test-clock trees;
- **Routing detours** introduced by sub-optimal scan chain ordering.

Generic congestion prediction models typically ask:
> *"Can we predict routing congestion from standard placement features?"*

**DeepDFT-Route** investigates two deeper, research-critical questions:
1. > ***"Can a multimodal CNN + GNN architecture using early physical design layout tensors and netlist/scan graphs predict post-route physical hotspots specifically attributable to DFT/scan implementation?"***
2. > ***"Can PDK-conditioned representations generalize inductively across process technology nodes—transferring from planar 45nm CMOS to commercial 14nm FinFET layouts without retraining from scratch?"***

---

## 2. Industry Impact: Why This Matters to Semiconductor Companies

In modern advanced-node ASIC flows, physical design is bottle-necked by late-stage routing iterations. **DeepDFT-Route** acts as a fast **AI surrogate model** that bridges the communication gap between Frontend/DFT engineers and Backend Physical Design (PD) teams.

In a production semiconductor environment, this provides high-value utility in three specific areas:

### A. Fast "Shift-Left" Iteration & Turnaround Time (TAT)
* **Traditional Detailed Route**: Takes **12 to 48+ hours** on high-performance compute clusters for large hierarchical blocks.
* **DeepDFT-Route Inference**: Takes **sub-second to seconds (< 10 ms on GPU)** on a single workstation.
* **Value to Chip Companies**: Enables true "shift-left" physical design. Physical designers and DFT engineers receive near-instantaneous feedback immediately after placement. If a specific scan partition creates an unroutable congestion hotspot, teams detect and rectify it within minutes rather than discovering it after a weekend-long routing job.

### B. Early DFT Architecture & Scan-Chain Co-Optimization
* In conventional flows, scan chains are stitched based on schematic/logical proximity or simple geometric heuristics without insight into upper-metal routing pressure.
* **Value to Chip Companies**: PD engineers can feed early surrogate congestion maps back into DFT tools to re-order scan chains, adjust scan-chain partitioning across core blocks, or insert test-clock repeaters *before* detailed routing begins.

### C. Compute Farm & EDA License Cost Optimization
* Running commercial PnR suites (Cadence Innovus / Synopsys ICC2 / OpenROAD) at scale consumes substantial compute farm resources and expensive license tokens.
* **Value to Chip Companies**: Acts as a rapid **pre-screening gate** in automated design sweeps. Unroutable placement configurations are pruned early, preventing wasted compute cycles and freeing up EDA tool licenses for viable designs.

---

### Traditional EDA Flow vs. DeepDFT-Route Surrogate Flow

![Enterprise ASIC Workflow: Traditional PnR vs DeepDFT-Route AI Surrogate](docs/assets/enterprise_workflow_comparison.svg)

| Dimension | Traditional Full Detailed Route | DeepDFT-Route (AI Surrogate) |
| :--- | :--- | :--- |
| **Execution Speed** | **12 – 48+ hours** (Multi-core compute cluster) | **< 10 ms** (Single workstation / GPU) |
| **Feedback Stage** | **Late Stage** (Post-detailed routing & DRC) | **Shift-Left** (Early global / legal placement) |
| **Failure Penalty** | 2–3 days lost per failed iteration | Seconds to re-adjust placement/scan chains |
| **Compute Footprint**| Heavy CPU/memory farm workload | Lightweight neural inference pass |
| **Primary Output** | Final wire geometry & DRC reports | High-resolution pre-route hotspot heatmaps & alert masks |

---

## 3. Milestone 5: Cross-Technology Generalization (14nm FinFET + 45nm Planar)

DeepDFT-Route incorporates **PDK-Conditioned Multimodal Learning** and a custom **Correlation-Aware Loss Function** that jointly optimizes Mean Squared Error and Pearson ranking contrast. Trained on a multi-PDK pool across 45nm planar CMOS and 14nm FinFET layouts, the model generalizes zero-shot to commercial, large-scale tapeout-ready silicon.

### Visual Benchmark on Commercial 14nm Holdout: OpenHW Group CV32E40P (RISCY)
*Evaluated strictly out-of-sample on the 131,670-instance commercial RISC-V core with hardware FPU (never seen during training).*

![OpenHW CV32E40P 14nm Benchmark](docs/assets/riscy_cv32e40p_evaluation_filtered.png)

- **Panel 1 (Pre-Route Placement):** 64×64 cell placement density filtered of non-functional dummy filler and decap cells.
- **Panel 2 (AI Predicted Congestion):** Pre-route wirelength congestion predicted by DeepDFT-Route (**Pearson $r = +0.9873$**).
- **Panel 3 (Router Ground Truth):** True detailed post-route wirelength density from Cadence Innovus detailed router.
- **Panel 4 (Hotspot Alert Mask):** **59.02% of all critical Top-5% routing bottlenecks caught** (**Green = True Positive (121 caught)**, Red = Missed, Blue = False Alarm). Inference latency: **< 10 ms**.

---

### Cross-Technology Comparative Heatmaps (45nm SPI vs 14nm RISCY)
*Demonstrating inductive cross-node transfer across planar and FinFET physics.*

![Cross-Technology Benchmark](docs/assets/cross_technology_benchmark.png)

---

## 4. Benchmark Progression on 14nm FinFET Holdout (`CV32E40P`)

The table below documents the complete milestone evolution on the unseen commercial core (**OpenHW CV32E40P with FPU**):

| Phase / Configuration | Training Pool | Loss Function | Pearson $r \uparrow$ | Spearman $\rho \uparrow$ | Top-5% Recall $\uparrow$ | Top-10% Recall $\uparrow$ | MAE $\downarrow$ | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Zero-Shot Baseline** | 45nm Planar only (No filter) | Standard MSE | $-0.1415$ | $+0.3532$ | $4.8\%$ | $10.0\%$ | $0.2183$ | Broken |
| **2. Physical Filtering** | 45nm Planar only (Dummy filter) | Standard MSE | $+0.1159$ | $+0.6925$ | $8.3\%$ | $19.0\%$ | $0.0612$ | Recovered |
| **3. PDK Conditioning** | 45nm + 1 single 14nm core | Standard MSE | $+0.2236$ | $+0.7105$ | $9.3\%$ | $26.8\%$ | $0.0540$ | Adapted |
| **4. Multi-PDK Scaling** | 45nm + 7 14nm cores (30 ep) | Standard MSE | $+0.7524$ | $+0.8220$ | $42.44\%$ | $54.88\%$ | $0.0414$ | Strong |
| **5. DeepDFT-Route Final** | **45nm + 14 14nm cores (60 ep)** | **Correlation-Aware** | **$+0.9873$** | **$+0.8979$** | **$59.02\%$** | **$66.10\%$** | **$0.0077$** | **State-of-the-Art** |

### Key Scientific Highlights:
1. **Near-Perfect Linear Tracking ($r = +0.9873$):** Pearson correlation reached $+0.9873$, confirming near-perfect linear proportionality between pre-route neural outputs and true post-route detailed wirelength density.
2. **High-Congestion Recall ($59.02\%$ Top-5%, $66.10\%$ Top-10%):** The model captures nearly 6 out of 10 extreme congestion bins and two-thirds of all high-congestion routing channels.
3. **Sub-1% Mean Absolute Error ($\text{MAE} = 0.0077$):** Across all 4,096 die bins on CV32E40P, average prediction error is under 0.8% of the normalized dynamic range.
4. **Preservation of 45nm Accuracy:** On the unseen 45nm SPI Master holdout, Pearson correlation improved to **$+0.8447$** with **$86.59\%$ Top-5% Recall**, demonstrating zero catastrophic forgetting.

---

## 5. Anti-Leakage Partition Protocol (Dataset Structure)

To guarantee scientific rigor, all training, validation, and holdout test partitions are isolated strictly by entire chip microarchitecture:

```mermaid
flowchart TD
    subgraph POOL["Consolidated 27-Layout Multi-PDK Repository"]
        N45["12 Nangate45 (45nm Planar CMOS)"]
        N14["15 CircuitNet-N14 (14nm FinFET)"]
    end

    subgraph TRAIN["1. Training Pool (22 Layouts - Backpropagation Gradients)"]
        TR1["45nm: uart_tx, dft_counter (u=45%, 55%)"]
        TR2["14nm: zero-riscy CV32E20 (10 layouts across seeds/freqs)"]
        TR3["14nm: Vortex-small RISC-V GPGPU (3 layouts, 110k cells)"]
        TR4["14nm: nvdla-small NVIDIA AI Accelerator (1 layout, 280k cells)"]
    end

    subgraph VAL["2. Validation Pool (2 Layouts - Early Stopping Only)"]
        VA1["45nm: uart_tx, dft_counter (u=65% high-stress runs)"]
        VA2["*Used solely to tune learning rate decay and early stopping*"]
    end

    subgraph TEST["3. Strictly Unseen Out-of-Sample Holdouts (3 Layouts - Zero Leakage)"]
        TE1["14nm FinFET Holdout: OpenHW CV32E40P (RISCY Core + FPU, 131k cells)"]
        TE2["45nm CMOS Holdout: SPI Master Controller (spi_nangate45)"]
        TE3["*Zero tile overlap, zero RTL reuse, completely unseen pipeline architectures*"]
    end

    N45 --> TRAIN
    N45 --> VAL
    N45 --> TEST
    N14 --> TRAIN
    N14 --> TEST
```

- **No Tile-Level Sharing:** Splits are partitioned by entire RTL design entities. No spatial tiles or cropped regions from test designs are ever present in training.
- **Microarchitecture Diversity:** The training set features distinct microarchitectures (UART, simple counters, 2-stage RISC-V, SIMT GPGPU compute, and deep learning systolic arrays).
- **Out-of-Distribution Generalization:** The commercial 14nm test holdout (**OpenHW CV32E40P**) features a 4-stage pipeline with hardware FPU and 131k placed cells, testing true inductive transfer rather than transductive interpolation.

---

## 6. Multimodal vs Single-Modality Ablation on 14nm FinFET

| Architecture Variant | Inputs Used | Pearson $r \uparrow$ | Top-5% Recall $\uparrow$ | Top-5% IoU $\uparrow$ | MAE $\downarrow$ |
|---|---|:---:|:---:|:---:|:---:|
| **Heuristic Baseline** | Standard Cell Density Only | $+0.1456$ | $18.54\%$ | $0.1022$ | $0.1194$ |
| **Ablation: Graph Only** | GraphSAGE GNN (Netlist Topology) | $+0.1393$ | $14.63\%$ | $0.0789$ | $0.0758$ |
| **Ablation: Spatial Only** | 2D UNet CNN (Placement Layout) | $+0.4098$ | $23.90\%$ | $0.1357$ | $0.0306$ |
| **DeepDFT-Route Multimodal** | **PDK-Conditioned CNN + GNN Fusion** | **$+0.9873$** | **$59.02\%$** | **$0.4187$** | **$0.0077$** |

*The multimodal fusion model decisively outperforms single modalities. Fusing continuous 2D layout geometry with discrete topological netlist constraints provides the required inductive bias to decode FinFET routing behavior.*

---

## 7. Technology Stack

| Layer | Tool / Framework | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Physical Implementation** | [OpenROAD](https://theopenroadproject.org/) / Cadence Innovus | 26Q4 | Floorplanning, placement, CTS, global & detailed routing |
| **Logic Synthesis** | [Yosys](https://yosyshq.net/yosys/) | 0.52 | RTL elaboration and logic mapping |
| **PDKs** | Nangate45 & CircuitNet-N14 | 45nm / 14nm | Planar standard cell library & commercial 14nm FinFET PDK |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | 2.5+ | Convolutional branches, correlation loss, metric pipelines |
| **Graph Neural Networks** | [PyTorch Geometric (PyG)](https://pyg.org/) | 2.5+ | Circuit netlist and scan graph message passing (GraphSAGE) |
| **Physical Layout Parsing** | Custom Python Engine | DEF 5.8 | High-throughput geometry parsing, non-functional dummy cell filter |
| **Cloud Acceleration** | Google Colab Pro CLI (`colab`) | T4 / A100 | Automated remote cloud training runner with chunked archive streaming |

---

## 8. Quick Start & Reproducibility Guide

Run commands inside WSL2 Ubuntu from the repository root:

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Nithinpbabu/deep-dft-route.git
cd deep-dft-route

# Create virtual environment and install dependencies
python3 -m venv ~/.venv_ml
source ~/.venv_ml/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

### 2. Verify Pipeline Integrity
Run the automated test suite (193 tests):
```bash
PYTHONPATH=src python3 -m unittest discover tests/unit
```

### 3. Evaluate the Trained Model on the Commercial 14nm Core
Benchmark the included pre-trained checkpoint on OpenHW CV32E40P:
```bash
python3 scripts/evaluate_riscy.py --filter
```
This generates the 4-panel diagnostic figure in `reports/riscy_cv32e40p_evaluation_filtered.png` and logs metrics to `reports/riscy_cv32e40p_metrics_filtered.json`.

### 4. Cloud Training on Google Colab Pro (Automated CLI)
To reproduce the 60-epoch cloud training on Google Colab Pro GPU:
```powershell
# In PowerShell (Windows):
.\scripts\train_on_colab.ps1 -Gpu T4 -Epochs 60
```
Or directly from Linux / WSL:
```bash
python3 scripts/run_colab_training.py --gpu T4 --epochs 60
```

---

## 9. License & Usage

Copyright © 2026 Nithin P B.

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

### Permitted Use
- Personal learning, experimentation, and research projects.
- Academic, non-commercial research, and university teaching.
- Educational review, portfolio demonstration, and technical evaluation.
- Modification and redistribution under the terms of the license.

### Commercial Licensing & Inquiries
For enterprise partnerships, commercial use, or integration into proprietary EDA tools, please contact:
- **Author**: Nithin P B
- **Email**: [pbnitin3@gmail.com](mailto:pbnitin3@gmail.com)
- **Official License Terms**: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
