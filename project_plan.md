# Project Plan: DFT-Aware Physical Design Hotspot Prediction using CNN + GNN

## 1. Executive Summary

This project builds an open-source, reproducible research system for predicting DFT-aware physical-design hotspots before final detailed routing. The intended final project is:

**DFT-Aware Physical Design Hotspot Prediction using CNN + GNN**

The project studies whether machine learning can predict physical regions that will become problematic because of scan/DFT-related implementation effects, using only information available from RTL, synthesized netlists, DFT/scan information, and early physical-design stages.

The target flow is:

```text
RTL
  -> synthesis
  -> DFT/scan insertion or scan metadata generation
  -> OpenROAD physical implementation
  -> pre-route / early physical feature extraction
  -> spatial feature tensors + circuit/scan graph
  -> baseline ML
  -> CNN / U-Net
  -> GNN / GraphSAGE
  -> CNN+GNN fusion
  -> DFT hotspot prediction
  -> post-route ground truth and evaluation
```

The project should use only free and open-source tools. Paid EDA tools, commercial cloud-only dependencies, proprietary PDKs, and proprietary datasets are out of scope. The recommended host environment is WSL2 Ubuntu on a Windows laptop, optionally using Docker for reproducibility and Kaggle free GPU when model training needs more GPU memory.

The practical aim is not to invent a novel deep-learning architecture. The aim is to build a credible, end-to-end EDA + ML research pipeline where the novelty comes from the DFT-aware problem formulation, dataset construction, feature engineering, ablation studies, and careful evaluation.

The project should be designed so future LLM coding agents can help implement it quickly while preserving scientific discipline. LLMs may generate boilerplate, parsers, orchestration scripts, Tcl, Python extractors, training loops, and documentation, but they must not invent EDA semantics, silently change the methodology, or claim unsupported tool behavior without validation.

## 2. Motivation

Modern digital chips are not implemented as generic logic only. They also include design-for-test structures such as scan chains, scan-enable logic, test clocks, compression logic, and ATPG-related structures. These DFT structures affect physical implementation.

DFT can introduce or worsen physical-design issues such as:

- long scan-chain routing;
- scan-chain crossings through congested regions;
- concentrated scan connections around specific modules;
- extra muxes or test logic increasing local cell density;
- scan-clock or scan-enable fanout stress;
- timing pressure on scan or test-mode paths;
- routing detours caused by scan-chain ordering;
- post-route congestion or DRC risk near scan-heavy regions.

Generic congestion prediction projects usually ask:

> Can we predict routing congestion or DRC hotspots from placement/layout features?

This project asks a narrower and more research-interesting question:

> Can we predict physical hotspots that are specifically associated with DFT/scan implementation, using early physical and scan-aware information before final routing?

This matters because DFT is often treated as a late-stage requirement, but scan-chain structure can materially affect physical design. If scan-related hotspot risk can be estimated earlier, engineers could adjust scan-chain partitioning, ordering, floorplanning, placement constraints, or routing strategy before expensive late-stage iterations.

## 3. Precise Problem Statement

Given:

- an RTL design;
- synthesized gate-level netlist;
- scan/DFT information, if available;
- early physical-design data such as floorplan, placement, cell locations, macro/blockage information, netlist connectivity, and estimated congestion;
- optional timing estimates from static timing analysis;

predict:

- a spatial heatmap over the chip area indicating regions likely to become DFT/scan-related physical hotspots after detailed routing;
- optionally, a binary hotspot mask identifying the highest-risk grid cells;
- optionally, a ranked list of top-k hotspot regions for engineer review.

The initial target should be:

> Predict post-route DFT/scan routing hotspots from pre-route or early physical-design plus DFT/scan information.

This should be implemented first as a continuous heatmap regression task. Binary classification can be derived from the continuous hotspot score using percentile or threshold rules.

## 4. Final Research Question

The final research question is:

> Can a CNN+GNN model using both spatial physical-design tensors and circuit/scan graph features predict post-route DFT/scan physical hotspots more accurately than physical-only models, graph-only models, generic congestion heuristics, and classical ML baselines, while generalizing to unseen RTL designs?

Secondary questions:

- Do explicit DFT/scan features improve hotspot prediction compared with physical-only features?
- Does graph structure improve generalization to unseen designs?
- Does fusion of CNN spatial context and GNN connectivity context outperform either model alone?
- Which feature groups matter most: physical density, scan-chain topology, timing estimates, or graph centrality?
- How robust are predictions across utilization, clock, aspect-ratio, and scan-configuration sweeps?

## 5. Scope

The project includes:

- setting up a free/open-source EDA flow;
- choosing small-to-medium open RTL designs;
- running synthesis and physical implementation;
- validating feasible open-source DFT/scan integration options;
- generating a dataset from repeated implementation sweeps;
- extracting spatial tensor features;
- extracting circuit graph and scan graph features;
- defining post-route DFT/scan hotspot ground truth;
- training classical ML baselines;
- training CNN/U-Net models;
- training GNN/GraphSAGE models;
- training a CNN+GNN fusion model;
- running ablation studies;
- producing benchmark tables, heatmaps, and a final report.

The expected project level is suitable for a strong undergraduate or graduate-level EDA/ML project, a resume project, a GitHub research artifact, or a stepping stone toward a publication-quality experiment.

## 6. Non-Goals

This project does not aim to:

- build a production EDA tool;
- replace commercial DFT, ATPG, placement, routing, or signoff tools;
- guarantee manufacturable chips;
- use paid tools, licensed PDKs, proprietary foundry data, or proprietary benchmarks;
- train very large neural networks;
- create a new exotic ML architecture;
- optimize scan-chain ordering in v1;
- perform full ATPG quality evaluation unless open-source tooling makes it practical;
- predict every physical-design failure mode at once;
- claim open-source tool capabilities before validating them locally.

Post-v1 extensions are listed later.

## 7. Target End-State Architecture

The final system should look like this:

```text
                      +----------------+
                      |      RTL       |
                      +-------+--------+
                              |
                              v
                      +----------------+
                      |   Synthesis    |
                      |     Yosys      |
                      +-------+--------+
                              |
                              v
                  +----------------------+
                  | DFT / Scan Handling  |
                  | Fault / OpenROAD DFT |
                  | capability validated |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | OpenROAD Physical    |
                  | Implementation       |
                  | floorplan/place/CTS/ |
                  | route/report         |
                  +----------+-----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +----------------------+       +----------------------+
   | Spatial Feature      |       | Circuit + Scan Graph |
   | Tensor Extraction    |       | Extraction           |
   | grid channels        |       | nodes/edges/features |
   +----------+-----------+       +----------+-----------+
              |                             |
              v                             v
   +----------------------+       +----------------------+
   | Classical Baselines  |       | Graph ML / GNN       |
   | linear/logistic/RF/  |       | GraphSAGE            |
   | XGBoost              |       +----------+-----------+
   +----------+-----------+                  |
              |                              |
              v                              v
       +--------------+              +--------------+
       | CNN / U-Net  |              | Graph Embeds |
       +------+-------+              +------+-------+
              |                             |
              +--------------+--------------+
                             |
                             v
                  +----------------------+
                  | CNN + GNN Fusion     |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | DFT Hotspot Heatmap  |
                  | and/or Hotspot Mask  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Post-route Ground    |
                  | Truth + Evaluation   |
                  +----------------------+
```

The implementation should be modular. EDA flow orchestration, feature extraction, dataset building, model training, evaluation, and visualization should be separate layers with stable file formats between them.

## 8. How This Differs from Generic Congestion and DRC Prediction

Many ML-for-EDA projects predict generic routing congestion, DRC hotspots, timing violations, or IR-drop regions. Those projects often use physical images or layout-derived tensors and train CNN-like models to predict post-route outcomes.

This project differs in the following ways:

1. **DFT-aware target**

   The label is not simply total routing congestion. The target should isolate or approximate scan/DFT-related routing stress.

2. **Explicit scan features**

   The model should include features such as scan-chain membership, scan-chain segment density, scan net crossings, scan path length estimates, scan endpoint distribution, scan-enable/test-clock fanout, and DFT cell density where available.

3. **Graph plus spatial representation**

   The chip layout is naturally spatial, but scan chains and circuit connectivity are naturally graph-structured. The final architecture should use both.

4. **Ablation-centered methodology**

   The project must prove whether DFT features matter by comparing physical-only, physical+DFT, graph-only, CNN-only, and fusion models.

5. **Dataset generated from open RTL and sweeps**

   Public datasets can be used for reference, but the core dataset should be generated through a reproducible open-source flow where the scan/DFT metadata and labels are controlled.

6. **Practical open-source EDA integration**

   The project must document what can and cannot be done with available open-source DFT/ATPG and physical-design tools.

## 9. Difference from `Pranavk098/chip-congestion-predictor`

The project `Pranavk098/chip-congestion-predictor` is understood here as a generic chip congestion prediction project. This plan must not copy its scope directly. This project should be different in research framing, data generation, feature design, and evaluation.

Expected differences:

- This project is DFT-aware, not only generic congestion-aware.
- It includes scan/DFT metadata as first-class features.
- It separates generic congestion labels from DFT/scan-specific hotspot labels as much as the open-source flow allows.
- It uses both spatial tensors and circuit/scan graphs.
- It evaluates whether DFT-aware features improve prediction.
- It emphasizes train/validation/test splitting by entire design, not random grid cells only.
- It includes open-source EDA flow reproducibility and data provenance as a core deliverable.
- It aims for a CNN+GNN fusion architecture, not only image-style prediction.

The project may still use generic congestion-prediction ideas as inspiration, such as grid-based tensors, heatmap labels, and CNN/U-Net prediction. But the final contribution should be the DFT-aware formulation and dataset.

## 10. Toolchain Requirements

Use only free and open-source tools.

Recommended tools:

- **Yosys** for RTL synthesis.
- **OpenROAD** and **OpenROAD-flow-scripts** for floorplanning, placement, CTS, routing, extraction, and reports.
- **OpenDB / ODB** for database access and physical data extraction.
- **OpenSTA** for timing reports and timing-derived features.
- **KLayout** and/or **Magic** for optional layout viewing and DRC-related inspection where appropriate.
- **Fault** and/or **OpenROAD DFT** for open-source DFT/scan/ATPG exploration, after capability validation.
- **Python** for orchestration, extraction, ML, and analysis.
- **pandas** and **numpy** for tabular data.
- **scikit-learn** for classical baselines and metrics.
- **XGBoost** for strong tabular baselines if installable in the environment.
- **PyTorch** for neural models.
- **PyTorch Geometric** for GraphSAGE and graph learning, if installable.
- **matplotlib**, **seaborn**, and optionally **plotly** for plots and heatmaps.
- **PyYAML** or **OmegaConf/Hydra** for configuration.
- **pytest** for tests.
- **DVC** is optional for dataset versioning if it remains manageable.
- **MLflow** or a lightweight local experiment tracker is optional; simple JSON/CSV experiment logs are acceptable for v1.

Important rule:

> Do not claim that a specific DFT or scan capability exists in a tool until it has been validated with the installed version and a minimal test case.

DFT integration is the highest-risk part of the project. The plan must allow fallback labels and fallback metadata if full scan insertion is not available.

## 11. Environment Recommendation

The user is on a Windows laptop. Recommended environment:

```text
Windows host
  -> WSL2 Ubuntu
      -> OpenROAD / OpenROAD-flow-scripts
      -> Yosys
      -> Python virtual environment or Conda/Mamba
      -> PyTorch CPU/GPU where practical
```

Optional:

- Docker inside WSL2 for reproducible OpenROAD builds.
- Kaggle free GPU for heavier model training.
- Local RTX 2050 GPU for small PyTorch experiments if CUDA setup is stable.

Avoid spending too much time fighting GPU setup early. The first models should run on CPU. GPU becomes useful after dataset generation and CNN/GNN training are stable.

## 12. Hardware Constraints and Practical Implications

User hardware:

- CPU: 13th Gen Intel Core i5-13420H
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 2050, 4 GB VRAM
- Integrated GPU: Intel UHD Graphics
- Storage: about 214 GB free from a 477 GB drive
- OS: Windows 64-bit, recommended WSL2 Ubuntu

Practical implications:

- EDA runtime and RAM are bigger bottlenecks than ML.
- Use small and medium RTL designs first.
- Avoid very large SoCs in v1.
- Keep placement/routing sweeps controlled.
- Prefer 64x64 or 128x128 spatial grids initially.
- Store compressed artifacts.
- Do not keep every intermediate OpenROAD build directory forever.
- Use CPU-friendly baseline models first.
- Keep CNN/U-Net small.
- Use mini-batching and graph sampling for GNNs if needed.
- Kaggle GPU can help for neural training, but the dataset should be portable.

Recommended first resource profile:

- Grid size: 64x64 for early experiments.
- Designs: 3 to 8 small designs.
- Sweeps per design: 10 to 50 initially.
- Dataset samples: 50 to 300 early, later 500+ if storage and runtime permit.
- CNN input channels: 8 to 15 initially.
- Batch size: tune based on memory; start small.

## 13. Dataset Strategy

The dataset strategy has two parts:

1. **Reference/benchmarking**

   Use public datasets such as CircuitNet only as reference material for generic congestion prediction, tensor formats, metrics, and baseline comparison ideas. CircuitNet should not be treated as sufficient for the project’s DFT-aware target unless it contains the required DFT/scan metadata, which must be verified.

2. **Primary project dataset**

   Generate a project-specific dataset using open RTL designs, open-source EDA flows, and controlled parameter sweeps.

The generated dataset is the core contribution. It should include physical implementation artifacts, scan/DFT metadata where available, extracted tensor features, graph features, labels, and metadata manifests.

## 14. Candidate RTL Designs

Candidate open RTL designs should be small enough for a laptop and documented enough to run through Yosys/OpenROAD.

Possible design categories:

- small RISC-V cores;
- microcontroller-style cores;
- AES/SHA/crypto accelerators;
- DSP blocks such as FIR filters;
- UART/SPI/I2C controllers;
- simple SoC peripherals;
- open benchmark circuits;
- OpenROAD-flow-scripts example designs;
- designs from educational or permissively licensed RTL repositories.

Selection criteria:

- permissive or clearly usable license;
- synthesizable with Yosys;
- no proprietary IP;
- manageable runtime under 16 GB RAM;
- enough sequential elements to make scan meaningful;
- can be run across multiple physical configurations;
- produces routable designs in at least one open PDK/platform.

Avoid designs that require large external memories, proprietary generators, complex software build systems, or unclear licenses.

## 15. Parameter Sweeps

To create enough dataset diversity, run sweeps across implementation parameters.

Recommended sweep dimensions:

- target utilization;
- clock period;
- aspect ratio;
- placement density;
- floorplan size;
- random seed, if supported;
- scan-chain count;
- scan-chain ordering mode, if available;
- scan-enable/test-clock buffering settings, if available;
- DFT insertion mode, if available;
- optional macro/blockage settings for designs with macros.

Initial sweep example:

```yaml
designs:
  - aes
  - ibex_small
  - uart
platforms:
  - nangate45
sweeps:
  utilization: [0.45, 0.55, 0.65, 0.75]
  clock_period_ns: [2.0, 3.0, 5.0]
  aspect_ratio: [1.0, 1.5]
  seed: [1, 2, 3]
  scan_chains: [1, 2, 4, 8]
```

Do not start with the full Cartesian product. Start small and grow after the flow works end to end.

## 16. Dataset Schema

Each implementation run should produce a sample with this structure:

```text
data/
  raw/
    runs/
      <run_id>/
        config.yaml
        manifest.json
        rtl/
        netlist/
        reports/
        openroad/
        dft/
        logs/
  processed/
    samples/
      <sample_id>/
        metadata.json
        features_spatial.npz
        graph.pt
        labels.npz
        masks.npz
        preview.png
  splits/
    train_designs.txt
    val_designs.txt
    test_designs.txt
    train_samples.txt
    val_samples.txt
    test_samples.txt
```

Every sample must have:

- unique `run_id`;
- design name;
- RTL source identifier;
- platform/PDK name;
- tool versions;
- synthesis configuration;
- physical-flow configuration;
- DFT configuration;
- grid configuration;
- feature-channel list;
- label definition version;
- extraction script version;
- timestamps;
- status: success, failed, skipped, partial;
- failure reason if not successful.

## 17. Spatial Grid and Tensor Design

Represent the chip as a fixed-size 2D grid. Each grid cell stores one value per feature channel.

Initial grid choices:

- 64x64 for v0/v1;
- 128x128 for final experiments if runtime and memory allow;
- keep die boundary and coordinate mapping in metadata.

Coordinate mapping:

- Use die/core area coordinates from OpenDB/OpenROAD.
- Convert physical coordinates into grid indices.
- Store physical bounds per grid cell when possible.
- Use consistent orientation across all samples.
- Record whether grid is over die area or core area.

Recommended tensor shape:

```text
C x H x W
```

where:

- `C` = feature channels;
- `H` = grid height;
- `W` = grid width.

Labels should use:

```text
1 x H x W
```

for continuous hotspot score and optionally:

```text
1 x H x W
```

for binary hotspot mask.

## 18. Recommended Initial 8-15 Spatial Feature Channels

Start with a compact channel list. Avoid creating 100 features before the pipeline works.

Recommended initial channels:

1. **Cell density**

   Number or area of standard cells per grid cell, normalized by grid-cell area.

2. **Sequential cell density**

   Density of flip-flops/latches per grid cell. Important because scan is related to sequential elements.

3. **Combinational cell density**

   Density of non-sequential standard cells.

4. **Estimated routing congestion**

   Pre-route global-routing estimate or congestion report mapped to the grid, if available.

5. **Pin density**

   Number of cell pins or net terminals per grid cell.

6. **Net bounding-box density**

   Count or weighted count of net bounding boxes passing through each grid cell.

7. **Scan cell density**

   Density of cells that are scan flops or DFT-related cells, if identifiable. If not available initially, use sequential cell density as a temporary proxy and mark this clearly.

8. **Scan net segment density**

   Estimated scan-chain wire demand through each grid cell based on scan-chain endpoints and approximate rectilinear routes. Requires scan-chain metadata.

9. **Scan endpoint density**

   Count of scan-chain endpoints or scan-connected sequential cells per grid cell.

10. **Scan crossing count**

   Count of estimated scan-chain connections crossing each grid cell.

11. **Clock/test-clock sink density**

   Density of clocked cells, and test-clock sink density if test-clock metadata is available.

12. **Scan-enable fanout density**

   Density or estimated fanout load for scan-enable/test-mode nets if identifiable.

13. **Timing slack pressure**

   Aggregated negative slack, near-critical endpoints, or timing pressure per grid cell from OpenSTA reports. If scan-mode timing is unavailable, use functional timing pressure and mark it as such.

14. **Macro/blockage density**

   Fraction of grid cell blocked by macros, obstructions, or routing blockages if present.

15. **Distance-to-boundary or normalized coordinates**

   Optional coordinate channels such as normalized x/y or distance to core boundary. Use with care because models may overfit to design-specific layout patterns.

For the first milestone, channels 1-8 are enough if DFT metadata exists. If DFT metadata is not yet available, start with physical channels plus sequential density, then add scan channels later.

## 19. Graph Schema

The graph should represent circuit and scan connectivity.

Possible graph levels:

1. **Cell-level graph**

   Nodes are cells. Edges are net connectivity. This may become large.

2. **Sequential-cell graph**

   Nodes are flops/latches/scan cells. Edges include scan-chain order and optionally functional connectivity summaries. This is smaller and directly relevant to DFT.

3. **Hybrid graph**

   Nodes are important cells, modules, or grid regions. Edges represent scan connections, functional connectivity, and spatial adjacency.

Recommended v1 graph:

- nodes: sequential cells and scan-related cells;
- scan edges: ordered scan-chain connections;
- functional edges: optional summarized connectivity between sequential cells;
- spatial edges: optional k-nearest-neighbor edges based on placement coordinates;
- node features: physical location, cell type, scan-chain id, scan index, fanin/fanout summary, timing features, local density values sampled from the spatial tensor.

Node feature examples:

- normalized x/y location;
- cell area;
- is_scan_cell;
- is_sequential;
- scan_chain_id encoded or embedded;
- normalized scan_index_in_chain;
- distance to previous scan cell;
- distance to next scan cell;
- functional fanin count;
- functional fanout count;
- clock domain id if available;
- local cell density;
- local congestion estimate;
- local timing slack pressure.

Edge feature examples:

- edge type: scan, functional, spatial;
- Manhattan distance;
- estimated wirelength;
- same_chain flag;
- scan order delta;
- net fanout;
- timing criticality if available.

Graph output options:

- node-level hotspot risk, later aggregated to grid;
- graph-level embedding fused into CNN features;
- grid-cell embeddings generated by pooling node embeddings into spatial cells.

For v1, the most practical fusion method is:

1. Run GraphSAGE over scan/sequential graph.
2. Pool node embeddings into grid cells based on node placement.
3. Concatenate pooled graph-embedding channels with CNN feature maps.
4. Predict heatmap using a small CNN head.

## 20. Labels and Ground Truth

The label must represent post-route DFT/scan physical hotspot severity.

The cleanest possible label would be:

```text
DFT hotspot score per grid cell =
  post-route routing demand or congestion caused by scan/DFT nets
```

However, open-source tool support may limit direct attribution. The project must validate what data is available.

Possible label definitions, from strongest to weakest:

1. **Routed scan-net congestion density**

   Identify scan/DFT nets after routing and compute routed wire length, via count, overflow, or congestion contribution per grid cell.

2. **Scan-net routing demand estimate after global/detail routing**

   Use routing guides or routed segments associated with scan nets and map them to the grid.

3. **Difference label**

   Compare implementation with scan/DFT enabled vs similar implementation without scan/DFT, and label the extra congestion or routing pressure attributable to DFT. This is scientifically attractive but may be noisy because implementation changes can cascade.

4. **Proxy scan-route hotspot**

   Use scan-net wirelength density, scan-chain crossings, scan-cell concentration, and post-route congestion overlap to approximate DFT-specific hotspots.

5. **Generic congestion with DFT overlay**

   Use generic post-route congestion labels and evaluate DFT features as predictors. This is a fallback only, not the desired final target.

Recommended initial label:

```text
continuous_hotspot_score[g] =
  normalized scan/DFT routed-wire density in grid cell g
  + alpha * normalized post-route congestion/overflow in g where scan nets are present
```

This should be adjusted after tool capability validation.

Binary label:

```text
hotspot_mask[g] = 1 if continuous_hotspot_score[g] >= percentile_threshold
```

Suggested threshold:

- top 5% grid cells per sample for early experiments;
- also report top 10% sensitivity;
- consider absolute thresholds if scores become comparable across designs.

Important:

- Store the label formula and version in metadata.
- Do not compare experiments that use different label formulas without marking it.
- Keep generic congestion labels separately from DFT-specific labels.

## 21. Distinguishing Generic Congestion from DFT-Specific Congestion

The project must carefully avoid claiming generic congestion prediction as DFT prediction.

Definitions:

- **Generic routing congestion**: congestion or overflow caused by all nets.
- **DFT/scan routing hotspot**: region where scan/DFT-related nets, scan cells, scan ordering, test control signals, or scan-related routing demand contribute strongly to physical difficulty.

Recommended stored labels:

- `label_total_congestion`: all-net post-route/global-route congestion.
- `label_scan_wire_density`: scan/DFT routed wire density.
- `label_scan_congestion_overlap`: total congestion weighted by scan/DFT net presence.
- `label_dft_hotspot_score`: final selected DFT-aware target.
- `label_binary_top5`: binary top-5% hotspot mask.
- `label_binary_top10`: optional top-10% mask.

The final report should show:

- examples where total congestion and DFT hotspot overlap;
- examples where they differ;
- limitations of the chosen label.

## 22. Metadata and Manifest Format

Each run should include a manifest:

```json
{
  "run_id": "aes_nangate45_u065_clk3_seed2_scan4",
  "sample_id": "sample_000123",
  "design": "aes",
  "design_source": {
    "name": "example-aes",
    "url": null,
    "commit": null,
    "license": "TBD"
  },
  "platform": "nangate45",
  "tool_versions": {
    "yosys": "validated-locally",
    "openroad": "validated-locally",
    "openroad_flow_scripts": "validated-locally",
    "python": "validated-locally"
  },
  "config": {
    "utilization": 0.65,
    "clock_period_ns": 3.0,
    "aspect_ratio": 1.0,
    "seed": 2,
    "scan_chains": 4
  },
  "dft": {
    "tool": "TBD",
    "validated": false,
    "mode": "TBD",
    "notes": "Do not assume capability until tested."
  },
  "grid": {
    "height": 64,
    "width": 64,
    "region": "core",
    "coordinate_units": "database_units_or_microns"
  },
  "features": {
    "version": "features_v001",
    "channels": [
      "cell_density",
      "sequential_density",
      "pin_density"
    ]
  },
  "labels": {
    "version": "labels_v001",
    "target": "dft_hotspot_score",
    "binary_threshold": "top_5_percent_per_sample"
  },
  "status": "success",
  "created_at": "ISO-8601",
  "notes": []
}
```

## 23. Train/Validation/Test Splitting

Avoid leakage.

Do not randomly split grid cells from the same chip into train and test. That makes results look better than they are.

Primary split:

- split by entire design;
- all runs/sweeps for a given design belong to one split.

Example:

```text
train: aes, uart, spi, fir
val:   sha
test:  ibex_small
```

Secondary split:

- within train designs, split by sweep configuration for validation during early debugging;
- final reported results should include unseen-design generalization.

Report both:

- within-design performance, clearly marked as easier;
- unseen-design performance, treated as the main research result.

## 24. Versioning and Reproducibility

The project must be reproducible.

Minimum requirements:

- all configs saved as YAML;
- all tool versions recorded;
- all dataset samples have manifests;
- generated features and labels include version IDs;
- random seeds recorded;
- train/val/test split files committed or versioned;
- no silent overwriting of datasets;
- no untracked manual edits to generated artifacts;
- scripts should be rerunnable from clean inputs.

Recommended version tags:

- `flow_v001`
- `features_v001`
- `labels_v001`
- `splits_v001`
- `baseline_v001`
- `cnn_v001`
- `gnn_v001`
- `fusion_v001`

## 25. Model Progression

Use a staged model progression. Each model exists for a reason.

### 25.1 Classical Baselines

Purpose:

- validate data quality;
- establish simple baselines;
- provide feature importance;
- catch leakage or label bugs.

Models:

- linear regression for continuous score;
- logistic regression for binary hotspots;
- random forest;
- XGBoost if available;
- simple heuristic baselines.

Inputs:

- per-grid-cell tabular features;
- optional neighborhood-aggregated features;
- optional design-level metadata.

Outputs:

- continuous hotspot score per grid cell;
- binary hotspot probability per grid cell.

Simple heuristics to compare against:

- predict hotspots from cell density only;
- predict hotspots from sequential density only;
- predict hotspots from scan endpoint density only;
- predict hotspots from estimated congestion only;
- predict top-k cells by scan segment density.

Acceptance:

- baseline pipeline trains and evaluates end to end;
- metrics are saved;
- feature importance is plotted for tree models;
- results are plausible but not overclaimed.

### 25.2 CNN / U-Net

Purpose:

- model spatial patterns;
- predict heatmaps from grid tensors;
- capture local and regional physical-design context.

Recommended architecture:

- small U-Net or encoder-decoder CNN;
- input: `C x H x W` tensor;
- output: `1 x H x W` continuous hotspot score;
- optional second output head for binary mask.

Why U-Net:

- well documented;
- common for dense prediction;
- easy to train;
- suitable for heatmaps;
- not exotic.

Loss functions:

- MSE or MAE for regression;
- weighted BCE or focal loss for binary classification;
- combined regression + classification loss if useful.

Important:

- handle class imbalance for binary hotspots;
- normalize input channels using train-set statistics only;
- avoid using post-route features as input.

### 25.3 GNN / GraphSAGE

Purpose:

- model circuit and scan-chain connectivity;
- capture long-range scan relationships that spatial CNNs may miss;
- learn embeddings for scan/sequential cells or grid regions.

Recommended architecture:

- GraphSAGE;
- 2 to 4 message-passing layers;
- hidden dimension 32 to 128;
- node embeddings pooled into grid cells or used for node-level labels.

Inputs:

- graph nodes: sequential/scan cells;
- graph edges: scan order, optional functional edges, optional spatial kNN edges;
- node/edge features as described earlier.

Outputs:

- node hotspot risk, pooled to grid; or
- grid-cell graph embedding for fusion.

Why GraphSAGE:

- well documented;
- scalable compared with full graph convolution on large graphs;
- supports inductive learning on unseen graphs;
- available in PyTorch Geometric.

### 25.4 CNN + GNN Fusion

Purpose:

- combine spatial layout features and scan/circuit graph context.

Recommended fusion method:

1. CNN encoder processes spatial tensor.
2. GraphSAGE processes scan/circuit graph.
3. Node embeddings are mapped to grid cells using placement coordinates.
4. Grid-cell graph embeddings are concatenated with CNN feature maps.
5. CNN decoder/head predicts DFT hotspot heatmap.

Alternative simpler fusion:

- compute graph-derived channels offline;
- append them to spatial tensor;
- train a normal U-Net.

Start with the simpler fusion if implementation time is limited. Move to end-to-end fusion after baselines are stable.

## 26. Experimental and Ablation Plan

The project must include ablations. Without ablations, it becomes just another heatmap model.

Core experiments:

1. **Physical-only baseline**

   Input features exclude DFT/scan-specific features.

2. **Physical + DFT features**

   Add scan density, scan endpoint density, scan segment density, test-control features.

3. **Heuristic comparison**

   Compare against density-only, congestion-only, and scan-density-only ranking.

4. **Classical ML vs CNN**

   Compare tabular per-grid models with U-Net heatmap prediction.

5. **CNN vs GNN**

   Compare spatial-only and graph-only approaches.

6. **CNN+GNN fusion**

   Test whether fusion improves unseen-design performance.

7. **Generalization to unseen designs**

   Main test split should be by design.

8. **Generalization to unseen configurations**

   Hold out some utilization/clock/scan settings.

9. **Feature importance**

   Use tree models and feature-drop ablations.

10. **Label sensitivity**

   Compare top-5% vs top-10% hotspot masks.

11. **Grid resolution sensitivity**

   Compare 64x64 and 128x128 if resources permit.

12. **DFT fallback sensitivity**

   If full scan insertion is not available, compare proxy labels and clearly discuss limitations.

## 27. Evaluation Metrics

Use metrics appropriate to both continuous heatmap regression and imbalanced hotspot classification.

Regression metrics:

- MAE;
- RMSE;
- normalized MAE;
- Pearson/Spearman correlation between predicted and true heatmaps;
- per-design averaged error.

Classification metrics:

- precision;
- recall;
- F1;
- AUROC;
- PR-AUC;
- balanced accuracy;
- calibration error if probabilities are reported.

Spatial map metrics:

- IoU/Jaccard for binary masks;
- Dice score;
- top-k hotspot recall;
- precision@k;
- recall@k;
- spatially weighted error if useful.

Recommended main metrics:

- MAE/RMSE for continuous score;
- PR-AUC for binary hotspot classification;
- top-5% hotspot recall;
- IoU or Dice for binary masks;
- unseen-design performance as the primary table.

Class imbalance:

- Most grid cells are not hotspots.
- Accuracy alone is not meaningful.
- PR-AUC is more informative than AUROC when hotspots are rare.
- Report calibration if using probability maps.

## 28. Milestone Roadmap: Stage 0 to Stage 7

### Stage 0: Project Setup and Method Freeze

Deliverables:

- repository initialized;
- environment setup notes;
- initial toolchain selected;
- this `project_plan.md` committed;
- minimal configuration structure created;
- decision log started.

Acceptance criteria:

- a new contributor or LLM agent can understand the project from the README and project plan;
- no paid/proprietary tools required;
- unknown tool capabilities are listed as validation tasks.

### Stage 1: Environment and One RTL Through Basic Flow

Goal:

- run one small RTL design through synthesis and OpenROAD physical flow.

Deliverables:

- WSL2 Ubuntu setup;
- Yosys working;
- OpenROAD/OpenROAD-flow-scripts working;
- one open RTL selected;
- one successful place-and-route run;
- logs and reports saved;
- basic manifest generated.

Acceptance criteria:

- command can be rerun from config;
- final DEF/GDS/reports produced, depending on flow support;
- tool versions recorded;
- runtime and storage noted.

### Stage 2: Physical Feature Extractor

Goal:

- extract grid-based physical features from pre-route or early physical data.

Deliverables:

- coordinate/grid mapper;
- cell density extraction;
- sequential density extraction;
- pin density extraction;
- net bounding-box density extraction;
- optional estimated congestion extraction;
- `.npz` feature tensor output;
- preview heatmaps.

Acceptance criteria:

- extracted tensors have expected shape;
- channels are documented;
- preview plots look physically plausible;
- extractor has unit tests for coordinate mapping.

### Stage 3: DFT/Scan Integration Validation

Goal:

- validate what open-source DFT/scan capability is available.

Candidate tools:

- Fault;
- OpenROAD DFT features, if available in installed version;
- any other fully open-source, locally runnable DFT/scan utility after validation.

Deliverables:

- minimal scan/DFT test design;
- validated commands and limitations;
- scan metadata format;
- fallback plan if full scan insertion is not practical;
- updated decision log.

Acceptance criteria:

- do not claim success unless a test case works;
- scan-chain metadata can be extracted or a documented proxy is created;
- DFT limitations are written clearly.

Fallback paths:

- use sequential-cell chain construction as a proxy scan chain;
- label it as synthetic scan metadata;
- focus first on scan-chain routing demand rather than full ATPG.

### Stage 4: Dataset Generator

Goal:

- automate multiple design/configuration runs and produce reproducible samples.

Deliverables:

- YAML sweep configs;
- run orchestration CLI;
- run manifest generation;
- processed sample generation;
- train/val/test split generation by design;
- dataset validation script.

Acceptance criteria:

- can generate at least 30 to 50 samples from small sweeps;
- failed runs are recorded, not silently ignored;
- splits avoid design leakage;
- data can be loaded by ML scripts.

### Stage 5: Classical ML Baselines

Goal:

- train simple baseline models and validate label/feature usefulness.

Deliverables:

- tabular dataset loader;
- logistic/linear baselines;
- random forest baseline;
- XGBoost baseline if installable;
- heuristic baselines;
- metrics table;
- feature importance plots.

Acceptance criteria:

- baselines run end to end;
- metrics saved in experiment directory;
- results are compared against heuristics;
- obvious leakage checks performed.

### Stage 6: CNN/U-Net and GNN/GraphSAGE

Goal:

- implement spatial deep learning and graph deep learning models.

Deliverables:

- PyTorch dataset class;
- U-Net model;
- training loop;
- evaluation and heatmap visualization;
- PyTorch Geometric graph dataset;
- GraphSAGE model;
- graph-to-grid pooling or graph-only evaluation;
- ablation results.

Acceptance criteria:

- CNN trains on CPU or GPU;
- GNN trains on manageable graphs;
- predictions can be visualized;
- validation metrics beat trivial baselines or failure is analyzed.

### Stage 7: CNN+GNN Fusion, Final Evaluation, and Report

Goal:

- complete final architecture and produce a polished research artifact.

Deliverables:

- fusion model;
- final ablation table;
- unseen-design evaluation;
- sample prediction heatmaps;
- architecture diagram;
- dataset card;
- reproducibility instructions;
- limitations section;
- GitHub-ready README;
- final report or paper-style document.

Acceptance criteria:

- full pipeline demonstrated from RTL to prediction;
- final results are reproducible from documented commands;
- DFT-specific contribution is clearly argued;
- limitations and fallback choices are honest;
- project is presentable on GitHub/resume.

## 29. Repository Structure

Recommended structure:

```text
dft-hotspot-prediction/
  README.md
  project_plan.md
  LICENSE
  pyproject.toml
  requirements.txt
  environment.yml

  configs/
    flows/
    sweeps/
    features/
    labels/
    models/
    experiments/

  scripts/
    setup/
    run_flow.py
    run_sweep.py
    extract_features.py
    build_dataset.py
    train_baseline.py
    train_cnn.py
    train_gnn.py
    train_fusion.py
    evaluate.py
    visualize_sample.py

  src/
    dft_hotspot/
      __init__.py
      config/
      eda/
      dft/
      odb/
      features/
      labels/
      graphs/
      data/
      models/
      training/
      evaluation/
      visualization/
      utils/

  rtl/
    external/
    manifests/

  flow/
    openroad/
    yosys/
    tcl/

  data/
    raw/
    processed/
    splits/
    manifests/

  experiments/
    baselines/
    cnn/
    gnn/
    fusion/

  reports/
    figures/
    tables/
    final/

  tests/
    unit/
    integration/
    data_validation/
    ml/

  docs/
    tool_validation/
    dataset_card.md
    decision_log.md
    limitations.md
```

Directory responsibilities:

- `configs/`: all user-editable experiment and flow settings.
- `scripts/`: thin CLI entry points.
- `src/`: reusable Python package.
- `rtl/`: references or checked-in small RTL where license permits.
- `flow/`: EDA tool scripts and templates.
- `data/`: generated datasets, preferably excluded or partially excluded from Git.
- `experiments/`: model outputs and logs.
- `reports/`: final human-facing outputs.
- `tests/`: automated validation.
- `docs/`: methodology, tool validation, decisions, limitations.

## 30. Automation and CLI Conventions

Use consistent CLI commands.

Examples:

```bash
python scripts/run_flow.py --config configs/flows/aes_nangate45.yaml
python scripts/run_sweep.py --config configs/sweeps/small_v001.yaml
python scripts/extract_features.py --run-id aes_nangate45_u065_clk3_seed2
python scripts/build_dataset.py --config configs/features/features_v001.yaml
python scripts/train_baseline.py --config configs/models/xgb_v001.yaml
python scripts/train_cnn.py --config configs/models/unet_v001.yaml
python scripts/train_gnn.py --config configs/models/graphsage_v001.yaml
python scripts/train_fusion.py --config configs/models/fusion_v001.yaml
python scripts/evaluate.py --experiment experiments/fusion/run_001
```

Rules:

- every command takes a config file;
- every run writes a manifest;
- output directories include timestamps or run IDs;
- scripts fail loudly on missing required files;
- failed EDA runs are recorded;
- no script silently overwrites a previous dataset or experiment;
- all random seeds are recorded;
- all generated metrics are saved as JSON/CSV.

## 31. Experiment Tracking

Minimum tracking:

Each experiment directory should contain:

```text
experiment.yaml
metrics.json
metrics.csv
training_log.csv
model.pt
predictions/
figures/
environment.json
git_commit.txt
notes.md
```

Track:

- model type;
- input feature version;
- label version;
- dataset split version;
- hyperparameters;
- random seed;
- training runtime;
- hardware used;
- metrics per design;
- aggregate metrics;
- failure notes.

Optional:

- MLflow;
- TensorBoard;
- Weights & Biases only if free and acceptable, but local-first tracking is preferred.

## 32. Testing Strategy

Testing is essential because EDA and ML pipelines can silently produce wrong data.

### Unit Tests

Test:

- coordinate-to-grid mapping;
- grid-cell area calculations;
- tensor channel ordering;
- normalization;
- label thresholding;
- graph construction;
- manifest parsing;
- config validation.

### Integration Tests

Test:

- tiny RTL through synthesis flow;
- feature extraction on a known small example;
- graph extraction from a tiny netlist;
- dataset sample loading;
- training one epoch on a tiny dataset.

### Flow Sanity Tests

Check:

- all expected EDA output files exist;
- reports are parseable;
- die/core dimensions are nonzero;
- cell count is nonzero;
- grid tensors contain finite values;
- labels are not all zero unless expected;
- binary hotspot mask has expected positive fraction.

### Data Validation Tests

Check:

- no NaNs/Infs in tensors;
- channel count matches metadata;
- split files contain valid sample IDs;
- no design appears in multiple final splits;
- labels do not use post-route features accidentally included as inputs.

### ML Tests

Check:

- model forward pass shape;
- loss computes without NaN;
- one training step updates parameters;
- evaluation metrics compute on toy data;
- saved model can be loaded.

## 33. Resource Management and Storage Rules

Storage is limited. Use rules from the beginning.

Recommended:

- compress processed tensors using `.npz`;
- keep only important OpenROAD artifacts for most runs;
- keep full logs for failed runs and representative successful runs;
- remove bulky intermediates after processed samples are validated;
- store plots at reasonable resolution;
- track large data outside Git;
- use `.gitignore` carefully;
- create a cleanup script with dry-run mode.

Suggested cleanup categories:

- keep: configs, manifests, reports, final DEF, key logs, processed tensors, labels, graph files;
- optional: full routed database for selected samples;
- discard after validation: temporary build files, duplicated logs, very large intermediate directories.

Never delete raw data automatically without:

- a manifest;
- processed sample validation;
- explicit cleanup command;
- dry-run preview.

## 34. LLM Usage Rules

LLMs such as Codex, Antigravity, or ChatGPT may be used heavily to accelerate development. This is encouraged, but controlled.

LLMs may:

- write Python boilerplate;
- write parsers after seeing real report examples;
- write Tcl templates after validating tool command syntax;
- create training loops;
- create plotting scripts;
- refactor code;
- write tests;
- draft documentation;
- help debug errors.

LLMs must not:

- invent EDA tool capabilities;
- assume scan insertion works without validation;
- silently change label definitions;
- silently include post-route features in pre-route inputs;
- change train/test split methodology without documenting it;
- claim results are meaningful without checking leakage;
- hide failed runs;
- fabricate benchmark numbers;
- use paid tools or proprietary PDKs;
- copy code or data without respecting licenses.

Every LLM-assisted change should preserve:

- reproducibility;
- explicit assumptions;
- stable configs;
- clear manifests;
- honest limitations.

When an LLM writes code that parses tool output, the parser must be tested against real output files from the installed tool version.

## 35. Coding Standards

Python standards:

- use clear module boundaries;
- type hints for public functions where useful;
- small functions for parsing and transformations;
- dataclasses or pydantic-style models for structured metadata if practical;
- no hardcoded absolute paths;
- config-driven behavior;
- deterministic random seeds where possible;
- readable errors;
- logging instead of print spam for scripts.

Data standards:

- every tensor channel documented;
- every label documented;
- every sample has metadata;
- every split has version;
- no hidden global state.

EDA script standards:

- keep Tcl scripts minimal and documented;
- record tool versions;
- keep generated and hand-written scripts separate;
- validate commands against installed tool documentation or help output.

Documentation standards:

- README gives quickstart and project story;
- `project_plan.md` gives full context;
- `docs/tool_validation/` records DFT and EDA capability tests;
- `docs/dataset_card.md` describes dataset creation and limitations;
- `docs/decision_log.md` records major decisions;
- `docs/limitations.md` states known weaknesses honestly.

## 36. Risks, Blockers, and Fallback Paths

### Risk 1: Open-source DFT integration is limited

Problem:

- full scan insertion or ATPG may not work cleanly for selected designs/tools.

Fallback:

- construct synthetic scan chains from sequential cells;
- clearly label them as synthetic;
- focus on scan-chain physical routing demand;
- still evaluate DFT-aware features as scan-aware physical features.

### Risk 2: DFT-specific ground truth is hard to isolate

Problem:

- router reports may not directly attribute congestion to scan nets.

Fallback:

- identify scan nets and map routed scan wires to grid;
- use scan-wire density plus overlap with generic congestion;
- compare scan-enabled vs no-scan flows cautiously;
- document label limitations.

### Risk 3: EDA runtime too high

Problem:

- physical implementation sweeps may be slow on 16 GB RAM.

Fallback:

- reduce design size;
- reduce sweep count;
- use 64x64 grids;
- generate dataset incrementally;
- use OpenROAD example designs;
- avoid large SoCs.

### Risk 4: PyTorch Geometric installation friction

Problem:

- PyG can be difficult to install with matching CUDA versions.

Fallback:

- use CPU PyG;
- use pure PyTorch message passing for a small graph;
- delay GNN until CNN and baseline pipeline are stable.

### Risk 5: Model overfits

Problem:

- small dataset, strong spatial biases.

Fallback:

- split by design;
- reduce model size;
- use feature ablations;
- report generalization honestly;
- increase dataset diversity.

### Risk 6: Labels are too noisy

Problem:

- noisy proxy labels may limit model performance.

Fallback:

- improve label extraction;
- use top-k ranking metrics;
- smooth labels spatially only if documented;
- compare multiple label definitions;
- frame results as proxy hotspot prediction.

### Risk 7: Licensing uncertainty

Problem:

- some open RTL may have unclear licenses.

Fallback:

- use only clearly licensed designs;
- record source and license in manifest;
- avoid redistributing RTL if license is unclear.

## 37. Decision Log

Current decisions:

1. Project title: **DFT-Aware Physical Design Hotspot Prediction using CNN + GNN**.
2. Use only free/open-source tools.
3. Recommended host environment: WSL2 Ubuntu on Windows.
4. Use OpenROAD/OpenROAD-flow-scripts for physical implementation.
5. Use Yosys for synthesis.
6. Validate open-source DFT/scan capabilities before relying on them.
7. Generate primary dataset from open RTL and parameter sweeps.
8. Use CircuitNet only as reference/benchmarking inspiration unless DFT-relevant metadata is validated.
9. Use grid-based spatial tensors for physical features.
10. Use graph representation for scan/circuit connectivity.
11. Initial target: post-route DFT/scan routing hotspot heatmap from pre-route/early physical + DFT information.
12. Model progression: classical baseline, CNN/U-Net, GraphSAGE, CNN+GNN fusion.
13. Main evaluation must include unseen-design split.
14. User hardware requires small-to-medium designs and careful storage management.

Items intentionally left open:

1. Exact open RTL design list.
2. Exact open PDK/platform choice for first complete flow.
3. Whether Fault, OpenROAD DFT, or another open tool will provide scan insertion.
4. Exact DFT label formula after tool capability validation.
5. Final grid resolution.
6. Whether PyTorch Geometric will run locally or on Kaggle.
7. Whether DVC/MLflow will be used or a lightweight local tracker is enough.

## 38. Final GitHub and Resume Deliverables

The final project should include:

- clear README;
- architecture diagram;
- reproducibility instructions;
- environment setup guide;
- tool validation notes;
- dataset card;
- data manifest examples;
- feature-channel documentation;
- model architecture descriptions;
- benchmark tables;
- ablation results;
- sample predicted vs true heatmaps;
- comparison against heuristic baselines;
- discussion of DFT-specific vs generic congestion labels;
- limitations section;
- future work section;
- final report or paper-style PDF/Markdown;
- short resume-ready summary.

Resume summary example:

```text
Built an open-source EDA+ML pipeline to predict DFT-aware physical-design hotspots from RTL-to-route implementation data. Generated a reproducible dataset using Yosys/OpenROAD sweeps, extracted spatial layout tensors and scan/circuit graphs, and compared classical ML, U-Net, GraphSAGE, and CNN+GNN fusion models on unseen-design hotspot prediction.
```

Do not include inflated claims. If DFT integration uses synthetic scan chains or proxy labels, say so clearly.

## 39. Suggested Future Extensions

These are post-v1 and should not distract from Stage 7.

Possible extensions:

- scan-chain ordering optimization using predicted hotspots;
- reinforcement learning or heuristic search for scan reordering;
- multi-task learning for congestion, DRC, timing, and DFT hotspots;
- timing-aware scan hotspot prediction;
- DRC hotspot prediction around DFT structures;
- explainability maps showing which scan features drive predictions;
- active learning for choosing new EDA sweeps;
- transfer learning across PDK/platforms;
- hierarchical graph models for module-level and cell-level prediction;
- integration with an interactive layout viewer;
- suggestion engine that recommends physical-design fixes.

Future extension rule:

> Do not start optimization or multi-task learning until the v1 prediction pipeline, dataset, and evaluation are stable.

## 40. Immediate Next Steps

Recommended next tasks:

1. Create repository skeleton.
2. Set up WSL2 Ubuntu and install/validate Yosys and OpenROAD-flow-scripts.
3. Run one tiny RTL design through the flow.
4. Record tool versions and logs.
5. Build the first grid mapper and physical feature extractor.
6. Validate open-source DFT/scan options on a minimal sequential design.
7. Choose the first 3 candidate RTL designs.
8. Create `docs/decision_log.md` and `docs/tool_validation/dft_validation.md`.
9. Generate the first 10-sample toy dataset.
10. Train the first trivial baseline and visualize one prediction/label heatmap.

The project should grow from a small working vertical slice, not from a large unvalidated architecture. The correct first success is one RTL design producing one valid tensor, one valid label, one manifest, and one baseline prediction.
