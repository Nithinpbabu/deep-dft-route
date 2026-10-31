# Stage 4 Status: Multi-Run Dataset Generation, Dataset Introspection & Baseline Formulations

**Current Status**: Complete (Stages 4a, 4b, 4c, 4d, and 4e Complete)  
**Date**: 2026-10-14  
**Next Stage Target**: Stage 5a (Expanded Multi-Design Dataset Generation `v002` before U-Net Benchmark)

---

## 1. Stage 4 Overview & Substage Roadmap

- [x] **Stage 4a: Small Multi-Run Smoke Dataset Generation (COMPLETE)**
  - Declarative sweep configuration architecture (`configs/sweeps/uart_nangate45_smoke_v001.yaml`).
  - Automated sweep runner orchestrating full 7-step pipeline per run (`src/dft_hotspot/sweep/`, `scripts/run_sweep.py`).
  - End-to-end execution of 6-run parameter sweep on UART/Nangate45: 6 attempted, 6 succeeded (100% success rate, 0 failures).
  - Canonical dataset layout generated at `data/processed/datasets/uart_nangate45_smoke_v001/` with manifest (`dataset_manifest_v001`), summary markdown, sample list, per-sample artifacts, and resolved run configs.
  - Comprehensive unit test suite extended with 7 sweep test cases (`tests/unit/test_sweep.py`); 85/85 tests passing in WSL2.
  - Architectural decisions documented in Decision 018 (`docs/decision_log.md`).
- [x] **Stage 4b: Dataset Loader, Statistics, and Split Policy (COMPLETE)**
  - Dataset collection and loader architecture (`src/dft_hotspot/data/dataset_collection.py`) with `PhysicalDesignDataset` and `SampleRecord`.
  - Multi-tensor validation across physical, scan, fused input, and routing ground-truth labels.
  - Automated deterministic duplicate detection combining SHA-256 tensor hashing and tolerance-based `np.allclose` comparisons (`detect_duplicates()`).
  - Empirical detection of 3 duplicate groups across 6 runs (clock period 3.0 ns and 5.0 ns produced identical layouts with 0.0 max absolute difference).
  - Global channel statistics computation across all 12 fused channels and 8 routing label channels.
  - Split policy generation (`splits_v001.json`, `smoke_config_holdout`): 4 train samples, 1 val sample, 1 test sample, with explicit limitations disclosure regarding duplicate val/test layouts.
  - CLI analysis tool (`scripts/analyze_dataset.py`) producing complete introspection reports (`dataset_statistics.json`, `duplicate_report.json`, `splits_v001.json`, `dataset_analysis_summary.md`).
  - Comprehensive unit test suite extended with 9 test cases (`tests/unit/test_dataset_collection.py`); 94/94 tests passing in WSL2.
  - Architectural decisions documented in Decision 019 (`docs/decision_log.md`).
- [x] **Stage 4c: Non-Learning Heuristic Baselines on Smoke Dataset (COMPLETE)**
  - Spatial metric evaluation framework (`src/dft_hotspot/evaluation/metrics.py`): continuous (MAE, RMSE, Pearson $r$, Spearman $\rho$) and thresholded hotspot metrics (Recall@k, Precision@k, IoU@k, Dice@k for $k \in \{5\%, 10\%\}$).
  - Suite of 7 non-learning heuristic baselines (`src/dft_hotspot/evaluation/heuristics.py`).
  - Established reference floor: `simple_weighted_sum` (VAL MAE = $0.0155$, Pearson $r = 0.6587$, Recall@5% = $66.83\%$).
  - Comprehensive unit test suite extended with 16 test cases (`tests/unit/test_evaluation_metrics.py`, `tests/unit/test_heuristics.py`); 110/110 tests passing in WSL2.
  - Architectural decisions documented in Decision 020 (`docs/decision_log.md`).
- [x] **Stage 4d: Classical ML Baselines on Smoke Dataset (COMPLETE)**
  - Tabular spatial feature matrix extraction (`src/dft_hotspot/ml/tabular_dataset.py`): $4,096$ grid cells per sample flattened into tabular rows with strict anti-leakage verification.
  - Classical machine learning baseline suite (`src/dft_hotspot/ml/classical_baselines.py`):
    1. `mean_baseline` (MeanBaselineRegressor)
    2. `linear_regression` (LinearRegression)
    3. `ridge_regression` (Ridge with validation alpha sweep)
    4. `lasso_regression` (Lasso with validation alpha sweep)
    5. `random_forest_regressor` (RandomForestRegressor, max_depth=8)
    6. `gradient_boosting_regressor` (HistGradientBoostingRegressor, max_depth=6)
    7. `xgboost_regressor` (gracefully skipped as optional uninstalled dependency)
  - Evaluated on `smoke_config_holdout` train ($16,384$ rows), val ($4,096$ rows), and test ($4,096$ rows) partitions.
  - Automated CLI training and evaluation driver (`scripts/train_classical_baselines.py`).
  - Comprehensive unit test suite extended with 9 test cases (`tests/unit/test_tabular_dataset.py`, `tests/unit/test_classical_baselines.py`); 119/119 tests passing in WSL2.
  - Architectural decisions documented in Decision 021 (`docs/decision_log.md`).
- [x] **Stage 4e: Final Stage 4 Smoke Benchmark Report and Readiness Review (COMPLETE)**
  - Comprehensive synthesis module (`src/dft_hotspot/reporting/stage4_benchmark.py`).
  - Dedicated CLI report generator (`scripts/generate_stage4_report.py`).
  - Published comprehensive benchmark report (`reports/stage4_smoke_benchmark.md`) containing 12 structured sections with Markdown tables, relative paths, and prominent smoke-only disclaimers.
  - Published machine-readable summary (`reports/stage4_smoke_benchmark_summary.json`) conforming to schema `stage4_smoke_benchmark_v001`.
  - Comprehensive unit test suite extended with 5 test cases (`tests/unit/test_stage4_report.py`); **124/124 tests passing in WSL2**.
  - Architectural decisions documented in Decision 022 (`docs/decision_log.md`).

---

## 2. Stage 4 Baseline Performance Comparison & Empirical Findings

### 2.1 Tabular Formulation & Anti-Leakage
- **Target Channel**: `total_wirelength_density` (normalized ground-truth routing label from `labels_routing.npz`).
- **Feature Set**: 12 pre-route and scan-aware channels from `fused_input.npz` (`physical/cell_density`, `physical/sequential_cell_density`, `physical/combinational_cell_density`, `physical/pin_density`, `physical/blockage_or_macro_density`, `physical/normalized_x`, `physical/normalized_y`, `scan/scan_cell_density`, `scan/scan_chain_endpoint_density`, `scan/scan_chain_segment_density`, `scan/scan_chain_order_index`, `scan/scan_chain_local_wirelength_proxy`).
- **Strict Anti-Leakage Rule**: Automated verification ensures zero post-route routing label channels appear in feature matrices.
- **Dataset Matrix Shapes**:
  - Training matrix ($N_{\text{train}} = 4$ samples): $X_{\text{train}} \in \mathbb{R}^{16384 \times 12}$, $y_{\text{train}} \in \mathbb{R}^{16384}$
  - Validation matrix ($N_{\text{val}} = 1$ sample): $X_{\text{val}} \in \mathbb{R}^{4096 \times 12}$, $y_{\text{val}} \in \mathbb{R}^{4096}$
  - Test matrix ($N_{\text{test}} = 1$ sample): $X_{\text{test}} \in \mathbb{R}^{4096 \times 12}$, $y_{\text{test}} \in \mathbb{R}^{4096}$

### 2.2 Model Performance Comparison (VAL Partition)

| Model | MAE | RMSE | Pearson ($r$) | Spearman ($\rho$) | Recall@5% | Precision@5% | IoU@5% | Dice@5% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Heuristic Ref (`simple_weighted_sum`)* | *0.0155* | *0.0750* | *0.6587* | *0.7609* | *66.83%* | *66.83%* | *0.5018* | *0.6683* |
| `mean_baseline` | 0.0425 | 0.0987 | N/A | N/A | 0.00% | 0.00% | 0.0000 | 0.0000 |
| `linear_regression` | 0.0236 | 0.0705 | 0.7020 | 0.3701 | 65.85% | 65.85% | 0.4909 | 0.6585 |
| `ridge_regression` ($\alpha = 10.0$) | 0.0237 | 0.0706 | 0.6990 | 0.3700 | 65.85% | 65.85% | 0.4909 | 0.6585 |
| `lasso_regression` ($\alpha = 0.0001$) | 0.0241 | 0.0708 | 0.6964 | 0.3700 | 65.85% | 65.85% | 0.4909 | 0.6585 |
| `gradient_boosting_regressor` | 0.0116 | **0.0495** | **0.8654** | 0.4561 | 81.46% | 81.46% | 0.6872 | 0.8146 |
| `random_forest_regressor` | **0.0113** | 0.0502 | 0.8611 | **0.7285** | **89.76%** | **89.76%** | **0.8142** | **0.8976** |

### 2.3 Feature Importance & Coefficient Findings

#### Linear Model Coefficients
- `physical/pin_density`: $+0.8082$ (Linear), $+0.7643$ (Ridge), $+0.3939$ (Lasso)
- `scan/scan_chain_segment_density`: $+0.4766$ (Linear), $+0.4741$ (Ridge), $+0.4644$ (Lasso)
- `physical/combinational_cell_density`: $+0.5259$ (Linear), $+0.2733$ (Ridge), $+0.2595$ (Lasso)

#### Tree Model Feature Importances (`random_forest_regressor`)
1. **`scan/scan_chain_segment_density`**: **35.12%** (Highest overall importance)
2. **`physical/pin_density`**: **33.95%** (Second highest importance)
3. `physical/normalized_x`: $19.09\%$
4. `physical/normalized_y`: $8.10\%$
5. `physical/combinational_cell_density`: $1.73\%$
6. All other features combined: $< 2.0\%$

**Core Empirical Takeaway**: The two features driving learned routing wirelength hotspot predictions are **scan segment routing demand** ($35.12\%$) and **standard cell pin density** ($33.95\%$), empirically substantiating the DFT-aware routing prediction thesis.

---

## 3. Stage 4 Readiness Review & Audit

Systematic review of the entire Stage 4 platform yields the following capability audit:

- [x] **RTL-to-Route Automation**: Fully operational via OpenROAD and Yosys.
- [x] **Parameter Sweep Runner**: Multi-run parameter sweeps execute without manual intervention (6/6 runs succeeded, 100% success rate).
- [x] **Multi-Tensor Validation**: 12-channel fused input tensors and 8-channel routing labels validate strictly against schema contracts.
- [x] **Spatial Metrics Framework**: Continuous (MAE, RMSE, Pearson, Spearman) and thresholded hotspot metrics (Recall@5%, IoU@5%) operational.
- [x] **Heuristic Reference Floor**: 7 spatial heuristics evaluated, establishing quantitative benchmark floors.
- [x] **Classical ML Infrastructure**: 6 tabular regressors trained, evaluated, and verified with anti-leakage guarantees.
- [x] **Duplicate Detection**: Automated SHA-256 and tolerance-based duplicate detection operational.
- [x] **Honest Limitations Documented**: All constraints and caveats documented in code, logs, and summaries.
- [x] **Ready for Larger Dataset Generation**: **YES** — The data generation, validation, and introspection pipeline is hardened and ready to scale.
- [ ] **Ready for Meaningful U-Net Benchmark**: **NOT YET** — Training a CNN/U-Net on 3 unique layouts would lead to trivial spatial memorization rather than generalized feature learning.
- [x] **Ready for U-Net Architecture Smoke Test**: **YES** — A standalone forward/backward pass code-path validation is feasible, provided no scientific claims are attached.

---

## 4. Transition Roadmap to Stage 5

### Primary Target: Stage 5a — Expanded Dataset Generation (`v002`)
To establish valid statistical footing for deep learning, the dataset must scale beyond single-design parameter invariance:
1. **Multi-Design Diversity**: Integrate 2 to 3 larger, structurally diverse open-source designs (e.g., `ibex` RISC-V core, `aes` cipher, `sparc_core`, or arithmetic accelerators).
2. **Real OpenROAD DFT Platform**: Leverage SkyWater 130hd (validated in Stage 3b) to generate real OpenROAD DFT scan chains alongside Nangate45 synthetic proxies.
3. **Broader Physical Parameter Variations**: Sweep core utilization across 0.35 to 0.75, aspect ratios (0.8, 1.0, 1.25), and aggressive clock periods that induce genuine timing pressure and varied routing congestion.
4. **Target Dataset Volume**: Generate at least **30 to 50 unique physical layouts**, providing sufficient sample variance for meaningful cross-design validation splits.

### Secondary / Parallel Target: Stage 5b — U-Net Architecture Prototype & Smoke Test
Implement the PyTorch U-Net / FCN architecture, verifying tensor feeding, receptive field configuration, multi-scale skip connections, and loss convergence as a **pure software smoke test** (clearly labeled as a code-path verification, not a benchmark evaluation).
