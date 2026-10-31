# Stage 3 Status: DFT/Scan Flow Validation & Benchmark Flow Automation

**Current Status**: Stage 3 Complete (Stages 3a through 3f Fully Completed)  
**Date**: 2026-10-14  
**Next Stage Target**: Stage 4a (Multi-Run Parameterized Flow Automation & Dataset Generation)

---

## 1. Stage 3 Overview & Substage Roadmap

- [x] **Stage 3a: Open-Source DFT/Scan Capability Survey (COMPLETE)**
  - Local toolchain inspection of OpenROAD binary, source tree, Yosys, and external packages.
  - Documented native OpenROAD DFT module (`src/dft/`) and available commands.
  - Documented external tool status (Fault not installed; deferred).
  - Recorded detailed report in `docs/tool_validation/dft_validation.md`.
- [x] **Stage 3b: Minimal Sequential DFT Flow Validation (COMPLETE)**
  - Created synthesizable minimal 16-bit sequential counter (`rtl/dft_smoke/dft_counter.v`).
  - Executed two-track validation:
    - **Track A (Sky130hd - Primary Candidate)**: Verified native OpenROAD DFT commands (`set_dft_config`, `scan_replace`, `report_dft_plan`, `execute_dft_plan`), generating real DEF `SCANCHAINS` and netlist scan ports (`scan_enable_0`, `scan_in_0`).
    - **Track B (Nangate45 - Negative Control)**: Proved exact failure boundary when Liberty lacks `test_cell()` annotations (`[WARNING DFT-0002] No scan equivalent lib cell found`), with zero scan chains reported/stitched.
  - Built repeatable validation automation (`scripts/validate_dft.py`, `flow/openroad/dft_validation.tcl`).
  - Implemented structured output schemas and reports (`dft_validation_summary.json`).
  - Verified with 5 new unit tests (`tests/unit/test_dft_validation.py`) with 45 total suite tests passing.
- [x] **Stage 3c: Scan Metadata Extraction From Validated OpenROAD DFT Outputs (COMPLETE)**
  - Implemented core metadata model and extractor in `src/dft_hotspot/dft/scan_metadata.py`.
  - Parsed DEF `SCANCHAINS` preserving exact ordered cell sequence (`_142_` to `_143_`).
  - Enriched scan cells with placement coordinates (`x`, `y` DBU and `x_um`, `y_um` microns), cell masters, and placement flags using DEF `COMPONENTS`.
  - Extracted top-level scan ports (`scan_enable_0`, `scan_in_0`, `q[8]`) and scan configuration parameters.
  - Persisted canonical artifact `data/dft_validation/dft_smoke_sky130hd_20261011_185057/scan_metadata.json` (`scan_metadata_v001` schema).
  - Added CLI tool `scripts/extract_scan_metadata.py` and 8 unit/integration tests (`tests/unit/test_scan_metadata.py`), achieving 53/53 test passes across the repository.
- [x] **Stage 3d: Synthetic Scan Metadata Fallback for Nangate45 (COMPLETE)**
  - Implemented deterministic synthetic scan chain generator in `src/dft_hotspot/dft/synthetic_scan.py`.
  - Extracted placed sequential cell components from layout DEF (`runs/uart_tx_nangate45_20261011_064013/scripts/uart_tx_final.def`) using sequential master heuristics (`DFF`, `DFFR`, `DFFS`).
  - Ordered 33 placed flip-flops deterministically via row-major spatial ordering (`row_major_by_y_then_x`: sorted by $y$, then $x$, then instance name), preserving strict physical layout coordinates.
  - Formatted synthetic proxy scan chain (`synthetic_chain_0`) under canonical schema `scan_metadata_v001` with explicit provenance `source: "synthetic_proxy"`.
  - Persisted canonical processed artifacts:
    - JSON: `data/processed/samples/uart_tx_nangate45_20261011_064013/scan_metadata.json`
    - Summary: `data/processed/samples/uart_tx_nangate45_20261011_064013/scan_metadata_summary.txt`
  - Added CLI generator `scripts/generate_synthetic_scan_metadata.py` and comprehensive unit/integration tests (`tests/unit/test_synthetic_scan.py`), achieving 61/61 test passes across the test suite.
- [x] **Stage 3e: Scan-Aware Physical Feature Tensor Prototype (COMPLETE)**
  - Implemented `src/dft_hotspot/features/scan_features.py` with `ScanFeatureExtractor` and `generate_scan_preview`.
  - Built 5 canonical scan-aware spatial feature channels on the $64 \times 64$ grid matching Stage 2 layout dimensions:
    1. `scan_cell_density`: spatial distribution of scan flip-flops (sum = 33.0, 31 active bins).
    2. `scan_chain_endpoint_density`: start and stop chain entry/exit pins (sum = 2.0, 2 active bins).
    3. `scan_chain_segment_density`: exact continuous-to-discrete length-conserving Manhattan chain routing demand (sum = 303.24 µm, 120 active bins).
    4. `scan_chain_order_index`: normalized scan chain sequence progression in $[0.0, 1.0]$ across bins.
    5. `scan_chain_local_wirelength_proxy`: localized scan wirelength demand around scan cells (sum = 606.48 µm, 31 active bins).
  - Generated decoupled sample artifacts in `data/processed/samples/uart_tx_nangate45_20261011_064013/`:
    - `features_scan.npz`: tensor shape $(5, 64, 64)$ for both normalized (`scan_features`) and raw (`raw_scan_features`).
    - `scan_feature_metadata.json`: schema `scan_features_v001` with detailed channel statistics and provenance.
    - `scan_feature_preview.png`: 5-panel heatmap visualization.
    - `scan_feature_summary.txt`: human-readable summary.
  - Added CLI tool `scripts/extract_scan_features.py` and 8 new unit/integration tests (`tests/unit/test_scan_features.py`), bringing total repository test count to 69/69 passing tests.
- [x] **Stage 3f: Combined Physical + Scan-Aware Sample Validation and Fusion Policy (COMPLETE)**
  - Extended `PhysicalDesignSample` in `src/dft_hotspot/data/dataset.py` to optionally load, validate, summarize, and fuse scan-aware input tensors (`features_scan.npz`) alongside physical layout features (`features_spatial.npz`) and routing labels (`labels_routing.npz`).
  - Supported dual-branch contract:
    - Physical layout input tensor: $(7, 64, 64)$
    - Scan-aware input tensor: $(5, 64, 64)$
    - Routing target tensor: $(8, 64, 64)$
  - Standardized fused baseline contract:
    - Concatenated along channel axis (axis 0) into a single 12-channel input tensor: $(12, 64, 64)$.
    - Prefixed channel names to prevent ambiguity: `physical/<name>` (7 channels) and `scan/<name>` (5 channels).
  - Implemented multi-tensor congruence validation in `validate_scan_features`:
    - Grid resolution congruence ($H_{\text{scan}} = H_{\text{phys}} = H_{\text{lbl}}$, $W_{\text{scan}} = W_{\text{phys}} = W_{\text{lbl}}$).
    - Finiteness check across normalized and raw scan tensors ($\text{np.isfinite}$).
    - Channel count matching between tensor shape and channel name list.
    - Metadata consistency verification.
  - Updated `scripts/validate_design_sample.py` with scan reporting and optional `--require-scan` flag.
  - Added standalone non-destructive exporter `scripts/export_fused_sample.py`, producing `fused_input.npz` and `fused_input_metadata.json` (`fused_input_v001` schema).
  - Added 9 comprehensive unit tests in `tests/unit/test_dataset.py`, bringing total passing test suite to 78/78 tests.

---

## 2. Key Findings & Architecture Decisions from Stage 3f

1. **Dual Contract vs. Fused Contract**:
   - The dataset representation supports both a dual-branch representation (`sample.feature_tensor` and `sample.scan_feature_tensor`) and a fused baseline representation (`sample.get_fused_input_tensor()`).
   - Fused $(12, 64, 64)$ is documented as the default baseline input for initial ML models (linear regression, Random Forest, UNet/CNN), while dual-branch access is preserved for future multi-branch or CNN+GNN models.
2. **Channel Namespace Prefixing**:
   - Fused channel names are strictly namespaced (`physical/cell_density`, `physical/pin_density`, ..., `scan/scan_cell_density`, `scan/scan_chain_segment_density`, ...).
3. **PDK-Agnostic Interface**:
   - Downstream models consume identical tensor shapes and channel definitions regardless of whether scan features originate from real OpenROAD DFT (`source: "openroad_dft"`, Sky130hd) or synthetic proxy sequencing (`source: "synthetic_proxy"`, Nangate45).
4. **Backward Compatibility**:
   - Physical design samples remain valid without scan features (falling back to physical baseline).
   - Validation fails cleanly if `--require-scan` is specified on an unstitched/scan-less sample, or if scan tensors exhibit spatial or numerical inconsistencies.

---

## 3. Stage 3 Completion Summary

With Stages 3a through 3f complete, Stage 3 is 100% finished:
- Tool capabilities and limitations empirically mapped (Native OpenROAD DFT verified on Sky130hd; Nangate45 negative control established).
- Automated scan metadata extraction formalized (`scan_metadata_v001`).
- Deterministic synthetic proxy scan sequencing established for Nangate45.
- Scan-aware spatial feature tensor prototype built ($5 \times 64 \times 64$, `scan_features_v001`).
- Combined paired sample validation, dual-branch contract, and 12-channel fused baseline contract established with full regression testing (78/78 tests passing).

**Next Step**: Proceed to **Stage 4: Dataset Generation and Baseline Machine Learning Modeling**.



