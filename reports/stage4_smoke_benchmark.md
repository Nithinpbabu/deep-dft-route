# Stage 4 Smoke Benchmark Report and Readiness Review

**Target Dataset**: `uart_nangate45_smoke_v001`  
**Schema Version**: `stage4_smoke_benchmark_v001`  
**Evaluation Date**: 2026-10-14  
**Status**: Complete — Ready for Dataset Scaling (Stage 5a)  

---

## 1. Executive Summary

Stage 4 established the complete multi-run dataset generation, introspection, and baseline evaluation infrastructure for DFT-aware physical design hotspot prediction. Across four sequential substages (4a through 4d), the framework progressed from a single validated prototype to an automated 6-run flow sweep, formalized dataset collection and duplicate detection, spatial heuristic baselines, and classical machine learning models trained on 12-channel fused feature tensors.

> [!IMPORTANT]
> **End-to-End Smoke Benchmark Only — Not a Scientific Generalization Claim**
>
> The results documented in this report represent an end-to-end software, numerical, and machine learning pipeline validation on a small standard cell design (`uart_tx`, 155 gates) across 6 flow executions. Because ample timing slack caused identical physical placements between 3.0 ns and 5.0 ns clock constraints, the 6 samples collapse into **3 unique physical layouts**. All reported metrics demonstrate pipeline correctness, feature extraction validity, and relative model behavior; they **must not** be construed as evidence of generalized cross-design predictive capability.

## 2. Stage 4 Scope and Non-Goals

### Stage 4 Substage Scope
- **Stage 4a (Multi-Run Sweep)**: Parameterized sweep runner (`scripts/run_sweep.py`) automating full 7-step RTL-to-fused-tensor flow per configuration point. Achieved 6/6 successful flow completions (100% success rate).
- **Stage 4b (Dataset Loader & Introspection)**: Unified dataset abstraction (`PhysicalDesignDataset`), multi-tensor integrity validation, automated SHA-256 duplicate detection, global channel statistics, and the `smoke_config_holdout` split policy.
- **Stage 4c (Heuristic Baselines)**: Implementation of 7 non-learning spatial heuristic predictors and spatial continuous/thresholded metrics (MAE, RMSE, Pearson $r$, Spearman $\rho$, Recall@5%, IoU@5%) to establish reference performance floors.
- **Stage 4d (Classical ML Baselines)**: Grid-cell tabular formulation ($4,096$ rows per sample, 12 feature channels) evaluating 6 learned classical regressors (Mean baseline, Linear, Ridge, Lasso, Random Forest, HistGradientBoosting) with strict anti-leakage verification.

### Explicit Non-Goals
- **No Deep Learning (CNN / U-Net / GNN)**: No convolutional or graph neural networks were trained in Stage 4. Spatial deep learning is deliberately deferred until dataset scaling creates sufficient unique layouts.
- **No Cross-Design Claims**: No claims of cross-design transferability are made. The benchmark evaluates only parameter perturbations on a single design (`uart_tx`).
- **No EDA Flow Re-runs or Overfitting**: No iterative hyperparameter tuning on the 3 layouts was conducted to artificially inflate scores.

## 3. Dataset Overview

- **Dataset Path**: [`data/processed/datasets/uart_nangate45_smoke_v001`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001)
- **Design**: `uart_tx`
- **PDK / Platform**: `nangate45`
- **Grid Resolution**: `64 x 64` ($4,096$ grid cells per sample)
- **Scan Metadata Source**: `synthetic_proxy` (Synthetic sequential chain proxy)
- **Tensors per Sample**:
  - Physical Input Tensor (`features_spatial.npz`): `(7, 64, 64)`
  - Scan-Aware Feature Tensor (`features_scan.npz`): `(5, 64, 64)`
  - Fused Input Tensor (`fused_input.npz`): `(12, 64, 64)`
  - Routing Target Tensor (`labels_routing.npz`): `(8, 64, 64)`
- **Target Routing Channel**: `total_wirelength_density` (Normalized post-route wirelength density)

### Sample Execution Summary

| Sample ID | Utilization | Clock Period | Scan Chains | Run Status | Sample Validation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `uart_tx_nangate45_u45_clk3p0` | 0.45 | 3.0 ns | 1 | success | PASS |
| `uart_tx_nangate45_u45_clk5p0` | 0.45 | 5.0 ns | 1 | success | PASS |
| `uart_tx_nangate45_u55_clk3p0` | 0.55 | 3.0 ns | 1 | success | PASS |
| `uart_tx_nangate45_u55_clk5p0` | 0.55 | 5.0 ns | 1 | success | PASS |
| `uart_tx_nangate45_u65_clk3p0` | 0.65 | 3.0 ns | 1 | success | PASS |
| `uart_tx_nangate45_u65_clk5p0` | 0.65 | 5.0 ns | 1 | success | PASS |

## 4. Dataset Duplicate Analysis

Automated duplicate detection (`detect_duplicates()`) evaluated both exact SHA-256 tensor content hashes and tolerance-based floating-point differences ($|a - b| < 10^{-6}$) across all fused input and routing label tensors. The analysis revealed that the 6 generated samples form exactly **3 duplicate groups**, containing **3 unique physical layouts**.

| Group | Sample Pair | Match Type | Max Fused Diff | Max Label Diff | Shared / Differing Config |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `uart_tx_nangate45_u45_clk3p0`, `uart_tx_nangate45_u45_clk5p0` | exact_sha256 | 0.0 | 0.0 | Util 0.45 / Clock: 3.0 vs 5.0 ns |
| 1 | `uart_tx_nangate45_u55_clk3p0`, `uart_tx_nangate45_u55_clk5p0` | exact_sha256 | 0.0 | 0.0 | Util 0.55 / Clock: 3.0 vs 5.0 ns |
| 2 | `uart_tx_nangate45_u65_clk3p0`, `uart_tx_nangate45_u65_clk5p0` | exact_sha256 | 0.0 | 0.0 | Util 0.65 / Clock: 3.0 vs 5.0 ns |

### Physical Interpretation & Significance
Because `uart_tx` is a small combinational/sequential block (~155 gates), timing constraints of 3.0 ns (333 MHz) and 5.0 ns (200 MHz) both possess positive worst negative slack (WNS). The OpenROAD timing-driven placer and router faced zero timing pressure at either clock frequency, resulting in deterministic placement locations governed entirely by wirelength and standard cell density minimization. This empirical finding reinforces the critical need for automated duplicate detection in EDA ML pipelines to prevent inflated evaluation scores.

## 5. Split Policy

Dataset partitioning was defined in [`splits_v001.json`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json) using the `smoke_config_holdout` strategy, which reserves utilization 0.65 for evaluation while assigning 0.45 and 0.55 to training.

| Partition | Sample Count | Sample IDs | Unique Layouts |
| :--- | :---: | :--- | :---: |
| **Train** | 4 | `uart_tx_nangate45_u45_clk3p0`, `uart_tx_nangate45_u45_clk5p0`, `uart_tx_nangate45_u55_clk3p0`, `uart_tx_nangate45_u55_clk5p0` | 2 (u45, u55) |
| **Validation** | 1 | `uart_tx_nangate45_u65_clk3p0` | 1 (u65) |
| **Test** | 1 | `uart_tx_nangate45_u65_clk5p0` | 1 (u65 — identical to Val) |

> [!WARNING]
> **Split Duplication Limitation Disclosure**
>
> The validation sample (`uart_tx_nangate45_u65_clk3p0`) and the test sample (`uart_tx_nangate45_u65_clk5p0`) belong to the exact same duplicate group (max difference = 0.0). As a direct consequence, test partition metrics mirror validation partition metrics identically with zero variance. This split policy serves solely to validate data loader mechanics, split filtering, and metric calculation pipelines without leaking training data into evaluation.

## 6. Heuristic Baseline Results

Stage 4c evaluated 7 non-learning spatial heuristic baselines on the validation split. All heuristics use per-sample max normalization to map single-channel and multi-channel proxies to $[0.0, 1.0]$, matching ground-truth routing labels.

### Validation Split Heuristic Performance (`split = val`)

| Heuristic Baseline | MAE | RMSE | Pearson ($r$) | Spearman ($\rho$) | Recall@5% | Precision@5% | IoU@5% | Dice@5% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cell_density` | 0.0155 | 0.0761 | 0.6485 | 0.6477 | 50.24% | 50.24% | 0.3355 | 50.24% |
| **`simple_weighted_sum`** | 0.0155 | 0.0750 | 0.6587 | 0.7609 | 66.83% | 66.83% | 0.5018 | 66.83% |
| `pin_density` | 0.0158 | 0.0772 | 0.6369 | 0.6653 | 53.17% | 53.17% | 0.3621 | 53.17% |
| **`scan_segment_density`** | 0.0170 | 0.0836 | 0.5734 | 0.6891 | 58.05% | 58.05% | 0.4089 | 58.05% |
| `scan_local_wirelength_proxy` | 0.0212 | 0.0997 | 0.2387 | 0.3181 | 23.90% | 23.90% | 0.1357 | 23.90% |
| `scan_cell_density` | 0.0212 | 0.0994 | 0.2574 | 0.3182 | 23.90% | 23.90% | 0.1357 | 23.90% |
| `sequential_cell_density` | 0.0212 | 0.0994 | 0.2574 | 0.3182 | 23.90% | 23.90% | 0.1357 | 23.90% |

### Heuristic Key Findings
- **Best Heuristic Reference**: `simple_weighted_sum` (0.35 pin + 0.25 cell + 0.25 scan_segment + 0.15 scan_local_wirelength) achieved the lowest validation error among non-learning baselines (MAE = 0.0155, Pearson $r$ = 0.6587, Recall@5% = 66.83%).
- **Scan Routing Demand**: The single-channel scan routing proxy `scan_segment_density` outperformed purely physical sequential cell density (Pearson $r$ = 0.5734 vs 0.2574, Recall@5% = 58.05% vs 23.90%), demonstrating that topological scan chain routing demand captures interconnect congestion far more effectively than point flip-flop locations.

## 7. Classical ML Baseline Results

Stage 4d evaluated 6 classical machine learning regressors operating on grid-cell tabular rows ($4,096$ rows per sample, 12 input feature channels from `fused_input.npz`). Training used 16,384 rows (4 train samples), while validation and testing each evaluated 4,096 rows. Post-route ground-truth channels were strictly excluded from input matrices to enforce zero leakage.

### Validation Split Model Performance (`split = val`)

| Model | MAE | RMSE | Pearson ($r$) | Spearman ($\rho$) | Recall@5% | Precision@5% | IoU@5% | Dice@5% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Heuristic Ref (`simple_weighted_sum`)* | *0.0155* | *0.0750* | *0.6587* | *0.7609* | *66.83%* | *66.83%* | *0.5018* | *66.83%* |
| **`random_forest_regressor`** | 0.0113 | 0.0502 | 0.8611 | 0.7285 | 89.76% | 89.76% | 0.8142 | 89.76% |
| **`gradient_boosting_regressor`** | 0.0116 | 0.0495 | 0.8654 | 0.4561 | 81.46% | 81.46% | 0.6872 | 81.46% |
| `linear_regression` | 0.0236 | 0.0705 | 0.7020 | 0.3701 | 65.85% | 65.85% | 0.4909 | 65.85% |
| `ridge_regression` | 0.0237 | 0.0706 | 0.6990 | 0.3700 | 65.85% | 65.85% | 0.4909 | 65.85% |
| `lasso_regression` | 0.0241 | 0.0708 | 0.6964 | 0.3700 | 65.85% | 65.85% | 0.4909 | 65.85% |
| `mean_baseline` | 0.0425 | 0.0987 | N/A | N/A | 0.00% | 0.00% | 0.0000 | 0.00% |

### Classical ML Performance Analysis
- **Non-Linear Tree Supremacy**: Non-linear tree regressors dramatically surpassed linear models and heuristic baselines. `random_forest_regressor` achieved the lowest overall MAE (0.0113) and top hotspot detection (Recall@5% = 89.76%, IoU@5% = 0.8142).
- **Correlation Leadership**: `gradient_boosting_regressor` achieved the highest Pearson correlation ($r = 0.8654$) and lowest RMSE (0.0495).
- **Linear Regularization**: Ridge ($\alpha = 10.0$) and Lasso ($\alpha = 0.0001$) stabilized linear coefficients, achieving Val MAE $\sim 0.0237$ and Pearson $r \sim 0.70$, matching linear regression while zeroing out redundant features.
- **Optional Dependency Notice**: `xgboost_regressor` was gracefully skipped as an uninstalled optional dependency; all standard baselines executed via scikit-learn.

## 8. Feature Importance and DFT-Aware Signal

Analysis of tree-based Gini feature importances (`RandomForestRegressor`) and standardized linear regression coefficients provides direct empirical insight into which features drive wirelength density prediction on this benchmark.

### Random Forest Gini Feature Importances

| Rank | Feature Channel | Category | Importance | Share (%) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `scan/scan_chain_segment_density` | Scan-Aware | 0.351211 | 35.12% |
| 2 | `physical/pin_density` | Physical | 0.339516 | 33.95% |
| 3 | `physical/normalized_x` | Physical | 0.190858 | 19.09% |
| 4 | `physical/normalized_y` | Physical | 0.080985 | 8.10% |
| 5 | `physical/combinational_cell_density` | Physical | 0.017281 | 1.73% |
| 6 | `scan/scan_chain_order_index` | Scan-Aware | 0.009035 | 0.90% |
| 7 | `physical/cell_density` | Physical | 0.005548 | 0.55% |
| 8 | `scan/scan_chain_local_wirelength_proxy` | Scan-Aware | 0.004103 | 0.41% |
| 9 | `scan/scan_cell_density` | Scan-Aware | 0.000695 | 0.07% |
| 10 | `physical/sequential_cell_density` | Physical | 0.000601 | 0.06% |
| 11 | `scan/scan_chain_endpoint_density` | Scan-Aware | 0.000167 | 0.02% |
| 12 | `physical/blockage_or_macro_density` | Physical | 0.000000 | 0.00% |

### Linear Model Standardized Coefficients

| Feature Channel | Linear Regression | Ridge ($\alpha=10.0$) | Lasso ($\alpha=0.0001$) |
| :--- | :---: | :---: | :---: |
| `physical/cell_density` | -0.5708 | -0.2829 | 0.0000 |
| `physical/sequential_cell_density` | 0.0061 | -0.0244 | -0.0008 |
| `physical/combinational_cell_density` | 0.5259 | 0.2733 | 0.2595 |
| `physical/pin_density` | 0.8082 | 0.7643 | 0.3939 |
| `physical/blockage_or_macro_density` | 0.0000 | 0.0000 | 0.0000 |
| `physical/normalized_x` | -0.0001 | -0.0001 | -0.0000 |
| `physical/normalized_y` | -0.0358 | -0.0359 | -0.0358 |
| `scan/scan_cell_density` | 0.0061 | -0.0243 | -0.0658 |
| `scan/scan_chain_endpoint_density` | 0.0035 | -0.0095 | 0.0000 |
| `scan/scan_chain_segment_density` | 0.4766 | 0.4741 | 0.4644 |
| `scan/scan_chain_order_index` | -0.0134 | -0.0258 | -0.0000 |
| `scan/scan_chain_local_wirelength_proxy` | -0.1047 | -0.1322 | -0.0102 |
| `[intercept]` | 0.0285 | 0.0286 | 0.0288 |

### Empirical Assessment of DFT-Aware Signal
On this smoke dataset, **`scan/scan_chain_segment_density` emerges as the single most influential feature** (35.12% tree importance), closely followed by `physical/pin_density` (33.95%). Together, these two features account for over **69%** of the predictive weight in Random Forest. In linear models, `scan/scan_chain_segment_density` consistently receives large positive coefficients ($+0.4766$ in Linear, $+0.4741$ in Ridge, $+0.4644$ in Lasso).

> [!NOTE]
> **Careful Empirical Qualification**:
> *On this smoke dataset, scan-chain segment density is a strong predictor of routed wirelength.* This confirms that the synthetic scan routing demand tensor introduces meaningful, orthogonal predictive signal to standard pin density. However, because all samples derive from a single design (`uart_tx`) under synthetic proxy DFT, this correlation must be confirmed across diverse designs with real OpenROAD DFT stitching before drawing broader scientific conclusions.

## 9. Artifacts Produced

### Dataset Artifacts
- Manifest: [`data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001/dataset_manifest.json)
- Statistics: [`data/processed/datasets/uart_nangate45_smoke_v001/dataset_statistics.json`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001/dataset_statistics.json)
- Duplicate Report: [`data/processed/datasets/uart_nangate45_smoke_v001/duplicate_report.json`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001/duplicate_report.json)
- Split Policy: [`data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001/splits_v001.json)
- Sample Directory: [`data/processed/datasets/uart_nangate45_smoke_v001/samples/`](file:///mnt/e/projects/DFT/data/processed/datasets/uart_nangate45_smoke_v001/samples) (6 fully validated samples)

### Baseline Experiment Artifacts
- Stage 4c Heuristic Metrics: [`experiments/heuristics/uart_nangate45_smoke_v001/metrics_by_split.csv`](file:///mnt/e/projects/DFT/experiments/heuristics/uart_nangate45_smoke_v001/metrics_by_split.csv)
- Stage 4c Heuristic Summary: [`experiments/heuristics/uart_nangate45_smoke_v001/metrics_summary.json`](file:///mnt/e/projects/DFT/experiments/heuristics/uart_nangate45_smoke_v001/metrics_summary.json)
- Stage 4d Classical Metrics by Split: [`experiments/classical/uart_nangate45_smoke_v001/metrics_by_split.csv`](file:///mnt/e/projects/DFT/experiments/classical/uart_nangate45_smoke_v001/metrics_by_split.csv)
- Stage 4d Model Summary: [`experiments/classical/uart_nangate45_smoke_v001/model_summary.json`](file:///mnt/e/projects/DFT/experiments/classical/uart_nangate45_smoke_v001/model_summary.json)
- Stage 4d Feature Importance: [`experiments/classical/uart_nangate45_smoke_v001/feature_importance.csv`](file:///mnt/e/projects/DFT/experiments/classical/uart_nangate45_smoke_v001/feature_importance.csv)
- Stage 4d Linear Coefficients: [`experiments/classical/uart_nangate45_smoke_v001/coefficients.csv`](file:///mnt/e/projects/DFT/experiments/classical/uart_nangate45_smoke_v001/coefficients.csv)

### Report Artifacts
- Benchmark Report: `reports/stage4_smoke_benchmark.md`
- Machine-Readable Summary: `reports/stage4_smoke_benchmark_summary.json`

## 10. Limitations

The Stage 4 benchmark is bounded by the following explicit limitations:
1. **Single Design**: All 6 runs originate from the single small design `uart_tx` (~155 standard cells).
2. **Tiny Dataset**: 6 total runs is insufficient for standard statistical learning or validation splits.
3. **Duplicate Layouts**: Clock period variation (3.0 ns vs 5.0 ns) produced 0.0 max absolute difference in placement/routing due to ample timing slack, yielding only 3 unique physical layouts.
4. **Synthetic Scan Metadata**: Because standard Nangate45 Liberty lacks `test_cell()` definitions, scan chains were synthesized via sequential-cell proxy rather than true OpenROAD DFT stitching.
5. **Tabular Grid-Cell Independence**: Tabular models treat each $64 \times 64$ grid cell independently, completely ignoring spatial convolutions, receptive fields, and multi-scale contextual features.
6. **Split Duplication**: Validation and test partitions contain identical layouts, meaning test metrics reproduce validation metrics with zero variance.
7. **No Cross-Design Split**: Model transferability to unseen circuit topologies or netlists cannot be tested.
8. **No Deep Learning (CNN/GNN) Yet**: U-Net and GNN architectures have not been trained or evaluated on this dataset.

## 11. Readiness Review

A systematic audit of all project components confirms the following readiness checklist:

- [x] **End-to-End Flow Execution**: RTL-to-routed-layout flow works reliably via OpenROAD and Yosys.
- [x] **Automated Sweep Pipeline**: Multi-run parameter sweeps execute without manual intervention (6/6 runs succeeded).
- [x] **Multi-Tensor Validation**: 12-channel fused input tensors and 8-channel routing labels validate strictly against schema contracts.
- [x] **Evaluation Metrics**: Continuous (MAE, RMSE, Pearson, Spearman) and thresholded hotspot metrics (Recall@5%, IoU@5%) implemented and tested.
- [x] **Heuristic Floor Established**: 7 spatial heuristics evaluated, defining reference performance baselines.
- [x] **Classical ML Suite Operational**: 6 tabular models trained, evaluated, and verified with anti-leakage guarantees.
- [x] **Introspection & Duplicate Detection**: Automated SHA-256 and tolerance-based duplicate detection operational.
- [x] **Honest Limitations Documented**: All constraints and caveats documented in code, logs, and summaries.
- [x] **Ready for Larger Dataset Generation**: **YES** — The data generation, validation, and introspection pipeline is hardened and ready to scale.
- [ ] **Ready for Meaningful U-Net Benchmark**: **NOT YET** — Training a CNN/U-Net on 3 unique layouts would lead to trivial spatial memorization rather than generalized feature learning.
- [x] **Ready for U-Net Architecture Smoke Test**: **YES** — A standalone forward/backward pass code-path validation is feasible, provided no scientific claims are attached.

## 12. Recommended Next Stage

### Primary Recommendation: Stage 5a: Expanded Dataset v002
Before conducting serious deep learning (U-Net) benchmarking, the project must scale the dataset to achieve true physical and topological diversity. Specifically, **Stage 5a** should incorporate:
1. **Additional Designs**: Integrate 2 to 3 larger, structurally diverse open-source designs (e.g., `ibex` RISC-V core, `aes` cipher, `sparc_core`, or arithmetic accelerators).
2. **Real OpenROAD DFT Platform**: Leverage SkyWater 130hd (validated in Stage 3b) to generate real OpenROAD DFT scan chains alongside Nangate45 synthetic proxies.
3. **Broader Physical Parameter Variations**: Sweep core utilization across 0.35 to 0.75, aspect ratios (0.8, 1.0, 1.25), and aggressive clock periods that induce genuine timing pressure and varied routing congestion.
4. **Target Dataset Volume**: Generate at least **30 to 50 unique physical layouts**, providing sufficient sample variance for meaningful cross-design validation splits.

### Secondary / Parallel Option: Stage 5b: U-Net Architecture Prototype & Smoke Test
In parallel or immediately following Stage 5a setup, implement the PyTorch U-Net / FCN architecture, verifying tensor feeding, receptive field configuration, multi-scale skip connections, and loss convergence as a **pure software smoke test** (clearly labeled as a code-path verification, not a benchmark evaluation).

---

*Report generated by `scripts/generate_stage4_report.py` on 2026-10-14.*
