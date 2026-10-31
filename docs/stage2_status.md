# Stage 2 Status: Physical Feature Extraction & Spatial Representation

## Overview
Stage 2 of the **DFT-Aware Physical Design Hotspot Prediction** framework bridges the physical EDA toolchain with machine learning representations by converting physical layout artifacts (DEF, floorplan dimensions, cell placements, pin connectivity) into standardized 2D spatial feature tensors and metadata samples.

---

## Stage 2a Milestones & Results

### 1. Scope & Execution
- **Target Flow Run**: `runs/uart_tx_nangate45_20261011_064013`
- **PDK / Technology**: Nangate45 Open Cell Library (`FreePDK45`)
- **Primary Input Artifact**: `uart_tx_final.def` (DEF 5.8, 154 components, 14 IO pins, 196 routed nets)
- **Output Sample Directory**: `data/processed/samples/uart_tx_nangate45_20261011_064013/`
- **Output Artifacts Generated**:
  - `features_spatial.npz` (Compressed spatial feature tensor and raw counts)
  - `metadata.json` (Structured design statistics, layout dimensions, channel metrics, and methodology disclosures)
  - `preview.png` (High-resolution multi-panel heatmap visualization)

---

### 2. Feature Tensor Specification
- **Grid Resolution**: `64 x 64` (height $H=64$, width $W=64$, die area: $100.0 \times 100.0\,\mu\text{m}$)
- **Tensor Shape Convention**: `C x H x W = (7, 64, 64)`
- **Data Type**: `float32`
- **Validation**:
  - All values are strictly finite (`np.isfinite == True`)
  - Normalized density values are in range `[0.0, 1.0]`
  - Spatial coordinates `normalized_x` and `normalized_y` are centered in `[0.0, 1.0]`
  - Cell density is verified non-zero across standard-cell placement core region

---

### 3. Channel Catalog & UART Statistics

| Index | Channel Name | Description | Raw Range | Norm Range | Active Bins |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 0 | `cell_density` | Total placed standard cells per grid cell | 0 - 3 | 0.0 - 1.0 | 113 |
| 1 | `sequential_cell_density` | Sequential flip-flop cells per grid cell | 0 - 2 | 0.0 - 1.0 | 31 |
| 2 | `combinational_cell_density` | Combinational logic gates and buffers per grid cell | 0 - 3 | 0.0 - 1.0 | 91 |
| 3 | `pin_density` | Netlist connectivity pin proxy (sum of connected pins + IO pins) | 0 - 13 | 0.0 - 1.0 | 119 |
| 4 | `blockage_or_macro_density` | Macro/placement blockage density (marked unavailable for UART) | 0 - 0 | 0.0 - 0.0 | 0 |
| 5 | `normalized_x` | Normalized horizontal layout coordinate $(col + 0.5) / W$ | 0.008 - 0.992 | 0.008 - 0.992 | 4096 |
| 6 | `normalized_y` | Normalized vertical layout coordinate $(row + 0.5) / H$ | 0.008 - 0.992 | 0.008 - 0.992 | 4096 |

---

### 4. Physical Layout & Extraction Metrics
- **Die Dimensions**: $100.0 \times 100.0\,\mu\text{m}$ ($200{,}000 \times 200{,}000$ DBU at 2000 DBU/$\mu\text{m}$)
- **Total Standard Cells**: 154
  - **Sequential Flip-Flops**: 33 ($21.4\%$) — 32 `DFFR_X1`, 1 `DFFS_X1`
  - **Combinational Elements**: 121 ($78.6\%$) — Inverters, NAND/NOR/AND/OR/XOR gates, MUXes, clock buffers
- **Top-Level IO Pins**: 14 (placed along bottom boundary `metal6`)
- **Nets**: 196
- **Net Pin Connections**: 577 instance pin attachments across nets

---

### 5. Methodology & Disclosures
1. **DEF-First Text Extraction**:
   Parsed directly from final routed DEF (`uart_tx_final.def`) without requiring OpenDB C++ runtime bindings.
2. **Sequential Cell Classification**:
   Heuristic regex classification matching standard flip-flop and latch master prefixes: `DFF`, `DFFR`, `DFFS`, `SDFF`, `EDFF`, `LATCH`, `DLH`, `DLL`.
3. **Pin Density Netlist Proxy**:
   Since standard cell LEF pin offsets were not directly extracted in Stage 2a, pin count was calculated per placed instance from net connectivity in DEF `NETS` and accumulated at the instance placement coordinate, alongside IO pins from DEF `PINS`.
4. **Channel Normalization**:
   Density channels (0–4) are normalized by maximum scaling: $\text{norm} = \text{raw} / \max(\text{raw}_{\max}, 1.0)$. Both normalized float32 tensor and raw counts are saved in `features_spatial.npz`.

---

### 6. Verification & Test Suite
- Automated unit and integration tests under `tests/unit/test_features.py`:
  - `TestCellClassifier`: Verified sequential vs combinational classification across all cell types.
  - `TestGridMapper`: Verified coordinate binning, boundary clamping, and normalized coordinate arrays.
  - `TestDefParser`: Verified parsing on synthetic DEF fixtures (die area, components, pins, nets).
  - `TestPhysicalFeatureExtractor`: Verified tensor shape `(7, H, W)`, finiteness, normalization, and sample serialization.
  - `TestRealUartRunExtraction`: Verified extraction directly against `runs/uart_tx_nangate45_20261011_064013`.
- Test suite outcome: **19/19 passing** (`Ran 19 tests in 2.328s: OK`).

---

### 7. Preview Heatmap
The generated heatmap visualization (`data/processed/samples/uart_tx_nangate45_20261011_064013/preview.png`) displays:
- Standard cell density showing placement legalization in the lower-central core area.
- Sequential flip-flop spatial distribution across 31 grid bins.
- Pin density highlighting high-connectivity logic clusters and bottom-edge IO pins.

---

---

## Stage 2b Milestones & Results: Post-Route Routing Label Extraction

### 1. Scope & Execution
- **Target Flow Run**: `runs/uart_tx_nangate45_20261011_064013`
- **PDK / Technology**: Nangate45 Open Cell Library (`FreePDK45`)
- **Primary Input Artifact**: `uart_tx_final.def` (DEF 5.8, 196 nets, 165 routed nets, 934 segments, 1083 vias)
- **Output Sample Directory**: `data/processed/samples/uart_tx_nangate45_20261011_064013/`
- **Output Artifacts Generated**:
  - `labels_routing.npz` (8-channel normalized `labels` and `raw_labels` float32 tensors)
  - `routing_label_metadata.json` (Structured routing metrics, layer distributions, channel statistics, and disclosures)
  - `routing_preview.png` (High-resolution 8-panel heatmap preview)

---

### 2. Label Tensor Specification
- **Grid Resolution**: `64 x 64` (height $H=64$, width $W=64$, die area: $100.0 \times 100.0\,\mu\text{m}$)
- **Tensor Shape Convention**: `C x H x W = (8, 64, 64)`
- **Data Type**: `float32`
- **Wirelength Units**: Microns ($\mu\text{m}$), derived from DBU coordinates divided by `UNITS DISTANCE MICRONS 2000`
- **Length Conservation**: Continuous Manhattan wire segments are discretized using exact 1D interval intersections across grid boundaries, guaranteeing $\sum_{\text{cells}} \text{density}_{\text{cell}} = \text{segment\_length}$.
- **Validation**:
  - All values are strictly finite (`np.isfinite == True`)
  - Normalized density values are in range `[0.0, 1.0]`
  - Total wirelength density equals sum of horizontal and vertical channels: $565.3 + 765.6 = 1330.9\,\mu\text{m}$
  - Total wirelength density equals sum of layer channels: $0.0 + 630.9 + 513.2 + 186.8 = 1330.9\,\mu\text{m}$

---

### 3. Channel Catalog & UART Statistics (8 Channels)

| Index | Channel Name | Description | Raw Range | Norm Max | Active Bins | Raw Sum |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 0 | `total_wirelength_density` | Total routed wirelength ($\mu\text{m}$) across all layers | 0.0 - 14.61 | 1.000 | 271 | 1330.9 $\mu\text{m}$ |
| 1 | `horizontal_wirelength_density` | Horizontal routed wirelength ($\mu\text{m}$) | 0.0 - 9.77 | 1.000 | 205 | 565.3 $\mu\text{m}$ |
| 2 | `vertical_wirelength_density` | Vertical routed wirelength ($\mu\text{m}$) | 0.0 - 8.55 | 1.000 | 262 | 765.6 $\mu\text{m}$ |
| 3 | `via_count_density` | Count of layout vias placed in grid cell | 0 - 16 | 1.000 | 211 | 1083.0 vias |
| 4 | `metal1_wirelength_density` | Routed wirelength on `metal1` (std-cell pin connections only) | 0.0 - 0.0 | 0.000 | 0 | 0.0 $\mu\text{m}$ |
| 5 | `metal2_wirelength_density` | Routed wirelength on `metal2` (horizontal routing layer) | 0.0 - 8.74 | 1.000 | 220 | 630.9 $\mu\text{m}$ |
| 6 | `metal3_wirelength_density` | Routed wirelength on `metal3` (vertical routing layer) | 0.0 - 9.16 | 1.000 | 197 | 513.2 $\mu\text{m}$ |
| 7 | `metal4plus_wirelength_density` | Routed wirelength on `metal4` + `metal5` + `metal6` | 0.0 - 7.90 | 1.000 | 79 | 186.8 $\mu\text{m}$ |

---

### 4. Observed Physical Routing Metrics for Reference (UART Run)
*(Note: These figures reflect the observed layout characteristics of `uart_tx_final.def` extracted in Stage 2b and are not hardcoded test constraints.)*

- **Die Dimensions**: $100.0 \times 100.0\,\mu\text{m}$ ($200{,}000 \times 200{,}000$ DBU)
- **Observed Nets**: 196 total nets (165 routed nets, 31 unrouted/special nets)
- **Observed Wire Segments**: 934 total
  - **Horizontal Segments**: 433 ($46.4\%$)
  - **Vertical Segments**: 501 ($53.6\%$)
  - **Diagonal / Non-Manhattan**: 0 ($0.0\%$)
- **Observed Vias**: 1083
- **Observed Routed Wirelength**: $1330.92\,\mu\text{m}$ ($2{,}661{,}840$ DBU)

#### Observed Layer Breakdown:
| Layer | Routing Pitch / Direction | Wirelength ($\mu\text{m}$) | Segments | Vias |
| :--- | :--- | :---: | :---: | :---: |
| `metal1` | Pin access only | 0.00 | 0 | 556 |
| `metal2` | Horizontal | 630.91 | 596 | 481 |
| `metal3` | Vertical | 513.25 | 309 | 18 |
| `metal4` | Horizontal | 101.36 | 10 | 14 |
| `metal5` | Vertical | 9.24 | 5 | 14 |
| `metal6` | IO escape / Pad access | 76.16 | 14 | 0 |

---

### 5. Methodology & Disclosures
1. **Post-Route Label Tensors**:
   Output tensors in `labels_routing.npz` are strictly designated as post-route ground-truth targets (labels), distinct from the model input features generated in Stage 2a (`features_spatial.npz`).
2. **DEF 5.8 Route Parsing**:
   Wire segments and vias are extracted directly from DEF `+ ROUTED <layer>` and `NEW <layer>` clauses, resolving coordinate taps `( x y )`, coordinate extensions `( x y ext )`, and coordinate reuses `( * y )` / `( x * )`.
3. **Exact Length Conservation**:
   Segments traversing multiple grid cells are partitioned along cell boundaries with fractional wirelength assignment, guaranteeing that total wirelength is strictly conserved from layout coordinates to discrete tensor bins ($\sum \text{density}_{\text{cell}} = \text{segment\_length}$).
4. **Absence of Fabricated Congestion Channels**:
   Because the OpenROAD physical flow run completed without generating global routing congestion/overflow reports in `reports/`, no artificial congestion labels were synthesized. Ground truth is represented faithfully by actual post-route wirelength densities and via counts.
5. **Channel Normalization**:
   Each channel is normalized via max scaling: $\text{norm} = \text{raw} / \max(\text{raw}_{\max}, 1.0)$. Both normalized float32 tensor and raw arrays (in microns) are saved in `labels_routing.npz`.

---

### 6. Verification & Test Suite
- Automated unit and integration tests under `tests/unit/test_routing_labels.py`:
  - `test_synthetic_def_horizontal_route_parsing`: Synthetic DEF horizontal route parsing.
  - `test_synthetic_def_vertical_route_parsing`: Synthetic DEF vertical route parsing.
  - `test_synthetic_def_via_parsing`: Synthetic DEF via parsing and layer assignment.
  - `test_coordinate_reuse_syntax`: Coordinate reuse syntax with `( * y )` and `( x * )`.
  - `test_layer_specific_channel_assignment`: Layer channel assignment (`metal1`, `metal2`, `metal3`, `metal4+`).
  - `test_segment_rasterization_length_conservation`: Exact length conservation across grid boundaries.
  - `test_label_tensor_shape_dtype_and_finite_values`: Shape `(8, 8, 8)`, float32 dtype, and finite value bounds.
  - `test_process_and_save_sample`: File serialization and sample artifact generation.
  - `test_uart_final_def_routing_extraction`: Integration test against UART final DEF (skips gracefully if artifact missing; verifies sane nonzero counts without hardcoded values).
- Full project test suite outcome: **29/29 passing** (`Ran 29 tests in 3.693s: OK`).

---

### 7. Preview Heatmap
The generated heatmap preview (`data/processed/samples/uart_tx_nangate45_20261011_064013/routing_preview.png`) renders an 8-panel visual representation:
- **Top Row**: Primary layout density targets — Total Wirelength, Horizontal Wirelength, Vertical Wirelength, and Via Count.
- **Bottom Row**: Metallization layer breakdown — Metal 1, Metal 2, Metal 3, and Metal 4+ (Metal 4-6).
- Visual alignment directly matches the standard cell placement core from Stage 2a (`preview.png`), with IO routing extending to bottom-edge pins.

---

### 8. Stage 2c Milestones & Results: Paired Dataset Representation & Unified Sample Builder

#### 1. Scope & Execution
- **Target Sample**: `data/processed/samples/uart_tx_nangate45_20261011_064013/`
- **Methodological Terminology Refinement**:
  - Tensors produced in Stage 2a are designated as **physical layout feature tensors (intended to be pre-route-compatible)** rather than claimed as true pre-route extractions.
  - *Rationale*: In the Stage 2a prototype, spatial tensors were extracted from `uart_tx_final.def` to validate the coordinate binning, standard-cell categorization, and pin proxy algorithms. In the true ML methodology, input features must strictly originate from early/pre-route-compatible artifacts (post-placement / pre-CTS / pre-global-route DEF), while ground truth labels originate from post-route DEF. Prototype final-DEF-derived inputs are therefore documented with this precise distinction.
- **Paired Artifacts Loaded & Validated**:
  - `features_spatial.npz` (Stage 2a physical layout feature tensors, 7 channels, float32)
  - `metadata.json` (Stage 2a extraction metadata)
  - `labels_routing.npz` (Stage 2b post-route routing targets, 8 channels, float32)
  - `routing_label_metadata.json` (Stage 2b routing metadata)
- **Unified Sample Abstraction**: `PhysicalDesignSample` in `src/dft_hotspot/data/dataset.py`

---

#### 2. Paired Sample Specification

| Dimension | Physical Layout Feature Tensor (Stage 2a) | Ground-Truth Label Tensor (Stage 2b) | Congruence / Verification |
| :--- | :---: | :---: | :---: |
| **Tensor Shape** | `(7, 64, 64)` | `(8, 64, 64)` | Spatial $H \times W$ matching ($64 \times 64$) |
| **Raw Tensor Shape** | `(7, 64, 64)` | `(8, 64, 64)` | Exact shape match to normalized tensors |
| **Data Type** | `float32` | `float32` | Strictly finite (`np.isfinite == True`) |
| **Channel Count** | 7 channels | 8 channels | Exact match to channel names lists |
| **Physical Die Area** | $100.0 \times 100.0\,\mu\text{m}$ | $100.0 \times 100.0\,\mu\text{m}$ | Strictly congruent |
| **Source Flow Run** | `runs/uart_tx_nangate45_20261011_064013` | `runs/uart_tx_nangate45_20261011_064013` | Strictly congruent |

---

#### 3. Complete Channel Catalog

- **Physical Layout Input Features (7 Channels, Pre-Route-Compatible Target)**:
  - `[0] cell_density`: Placed standard cells per grid cell
  - `[1] sequential_cell_density`: Sequential flip-flop cells per grid cell
  - `[2] combinational_cell_density`: Combinational logic gates per grid cell
  - `[3] pin_density`: Netlist connectivity pin proxy + IO pins
  - `[4] blockage_or_macro_density`: Macro/placement blockage density
  - `[5] normalized_x`: Normalized horizontal grid coordinate
  - `[6] normalized_y`: Normalized vertical grid coordinate

- **Post-Route Routing Targets (8 Channels)**:
  - `[0] total_wirelength_density`: Total routed wirelength ($\mu\text{m}$)
  - `[1] horizontal_wirelength_density`: Horizontal routed wirelength ($\mu\text{m}$)
  - `[2] vertical_wirelength_density`: Vertical routed wirelength ($\mu\text{m}$)
  - `[3] via_count_density`: Via count per grid cell
  - `[4] metal1_wirelength_density`: Metal 1 wirelength ($\mu\text{m}$)
  - `[5] metal2_wirelength_density`: Metal 2 wirelength ($\mu\text{m}$)
  - `[6] metal3_wirelength_density`: Metal 3 wirelength ($\mu\text{m}$)
  - `[7] metal4plus_wirelength_density`: Metal 4-6 wirelength ($\mu\text{m}$)

---

#### 4. Validation Architecture & Pragmatic Metadata Contracts
The `PhysicalDesignSample.validate()` method and `validate_design_sample.py` CLI enforce a 7-tier validation contract:
1. **Artifact Completeness**: Guarantees all 4 required NPZ and JSON files exist on disk (`features_spatial.npz`, `metadata.json`, `labels_routing.npz`, `routing_label_metadata.json`).
2. **Tensor Dimensionality**: Verifies 3D shape `(C, H, W)` for both normalized and raw arrays (`features.shape == raw_features.shape`, `labels.shape == raw_labels.shape`).
3. **Spatial Grid Alignment**: Asserts $(H_{\text{feat}}, W_{\text{feat}}) == (H_{\text{lbl}}, W_{\text{lbl}})$.
4. **Finite Values Verification**: Asserts `np.all(np.isfinite(tensor))` across all normalized and raw arrays (strictly no NaN or Inf).
5. **Channel Name Count Matching**: Asserts $\text{len}(\text{channel\_names}) == C$ for both features and labels.
6. **Metadata Grid & Shape Congruence**: Verifies metadata `grid_size` and `tensor_shape` match loaded arrays where provided.
7. **Source Run & Design Consistency**: Pragmatically checks metadata consistency:
   - **Hard Errors**: Raised on true contradictions (conflicting source run directories or conflicting RTL design names).
   - **Soft Warnings**: Issued via `UserWarning` if optional metadata fields (`source_run_dir` or `design`) are missing from one metadata file, avoiding fragile crashes during incremental dataset development.

---

#### 5. CLI Tools Implemented
1. **Sample Validator (`scripts/validate_design_sample.py`)**:
   ```bash
   PYTHONPATH=src python3 scripts/validate_design_sample.py \
     --sample-dir data/processed/samples/uart_tx_nangate45_20261011_064013
   ```
   Outputs formatted sample breakdown and reports `[RESULT] PASS` (exit code 0 on PASS, 1 on FAIL).

2. **Unified Builder (`scripts/extract_design_sample.py`)**:
   ```bash
   PYTHONPATH=src python3 scripts/extract_design_sample.py \
     --run-dir runs/uart_tx_nangate45_20261011_064013 \
     --grid-size 64 \
     --output-dir data/processed/samples/uart_tx_nangate45_20261011_064013
   ```
   Orchestrates Stage 2a feature extraction, Stage 2b routing label extraction, and executes paired sample validation in a single atomic pass without duplicating extraction logic.

---

#### 6. Test Suite & Verification
Automated unit tests under `tests/unit/test_dataset.py`:
- `test_load_real_uart_sample`: Validates real UART paired sample on disk (skipping gracefully if missing).
- `test_missing_file_error`: Validates `FileNotFoundError` when any of the 4 required files is missing.
- `test_mismatched_grid_shape_error`: Validates `ValueError` when feature and label grids have mismatched dimensions ($64 \times 64$ vs $32 \times 32$).
- `test_channel_count_mismatch_error`: Validates `ValueError` on channel count discrepancy.
- `test_finite_value_validation`: Validates `ValueError` on NaN or Inf in feature or label tensors.
- `test_metadata_inconsistency_error`: Validates `ValueError` on conflicting source run directories.
- `test_metadata_design_mismatch_error`: Validates `ValueError` on conflicting RTL design names.
- `test_optional_metadata_missing_fields_allowed_and_warns`: Validates that missing optional metadata fields pass validation while emitting `UserWarning`.
- `test_summary_output_fields`: Validates all required keys in `sample.summary()`.

**Full Test Suite Result**: **38/38 passing** (`Ran 38 tests in 4.241s: OK`).

---

### 9. Recommended Next Stage: Stage 3a (Multi-Run Dataset Generation & RTL Benchmark Automation)
With Stage 2 complete (physical layout feature tensors, post-route routing labels, and unified paired sample abstraction):
1. **RTL Benchmark Flow Sweeps**: Run additional benchmark designs through the physical flow (`fifo`, `spi_master`, `aes`, `picorv32`) with parameter sweeps (utilization, aspect ratios, target clocks).
2. **Automated Batch Sample Extraction**: Use `scripts/extract_design_sample.py` across all completed runs into `data/processed/samples/`.
3. **PyTorch Dataset Scaffolding**: Implement `torch.utils.data.Dataset` / `DataLoader` wrappers around `PhysicalDesignSample` in `src/dft_hotspot/data/` for neural network ingestion.
4. **Data Splitting & Cross-Validation**: Formalize unseen-design dataset splits in accordance with Decision 003.

