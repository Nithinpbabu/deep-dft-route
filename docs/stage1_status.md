# Stage 1 Execution & Acceptance Summary

## Overview
Stage 1 of the **DFT-Aware Physical Design Hotspot Prediction** framework establishes physical design flow orchestration, configuration management, script generation, dry-run infrastructure, **WSL2 Ubuntu real EDA environment validation**, **Nangate45 PDK asset mirror**, **real Yosys 0.52 synthesis execution**, and **real OpenROAD physical design flow (Floorplan, Pin Placement, Global/Detailed Placement, CTS, Global/Detailed Routing, DEF/ODB export, Timing Closure)**.

---

## Canonical Manifest Status Taxonomy

The flow runner produces a `manifest.json` for every run attempt, using the following standardized `status` field values:

| `status` Value | Description |
| :--- | :--- |
| `dry_run` | Flow rendered scripts, copied RTL, and prepared workspace without invoking EDA tools (`dry_run_completed: true`). |
| `success` | Physical design flow completed synthesis, placement, and routing successfully. |
| `failed` | Run failed due to runtime EDA tool errors during synthesis/P&R execution. |
| `blocked_missing_eda_env` | Run blocked because Yosys/OpenROAD binaries or Linux/WSL2 environment are unavailable. (If Yosys completed, `synthesis_completed: true` is set). |
| `blocked_missing_pdk_assets` | Run blocked because required physical PDK files (Liberty, tech LEF, stdcell LEF) are missing on disk. |

---

## Stage 1 Execution Results & Metrics

- **Environment Subsystem**: WSL2 Ubuntu 26.04.1 LTS (Linux Kernel `6.18.40.1-microsoft-standard-WSL2`)
- **Python Virtual Environment**: `~/.venv_wsl` (Python 3.14.4)
- **PDK Assets Mirrored**: `pdk/nangate45/`
  - `NangateOpenCellLibrary_typical.lib` (6.69 MB)
  - `NangateOpenCellLibrary.tech.lef` (22 layers, 27 vias)
  - `NangateOpenCellLibrary.macro.lef` (134 standard cells)
  - `make_tracks.tcl` (routing track pitches and offsets for `metal1`-`metal10`)
- **Yosys Synthesis Engine**: Installed (`Yosys 0.52`) and **EXECUTED REAL SYNTHESIS**.
  - **Cell Count**: 147 mapped standard cells (32 `DFFR_X1`, 1 `DFFS_X1`, 14 `INV_X1`, 15 `NAND2_X1`, 17 `OAI21_X1`, 13 `AOI21_X1`, 10 `MUX2_X1`, 10 `AND2_X1`).
  - **Total Gate Area**: `299.782 um²` (Sequential elements: `175.560 um²` / 58.56%).
  - **Output Netlist**: `synth_uart_tx.v`
  - **Synthesis Log**: `logs/synth.log`
- **OpenROAD Toolchain Resolution & Build**:
  - **Previous Blockers Resolved**:
    1. *Missing System Libraries*: Resolved via `libzstd-dev` and `libicu-dev`.
    2. *Abseil Link-time Mismatch*: The linker originally pulled system Abseil `20260107` instead of bundled OR-Tools Abseil `20250512`. Resolved via CMake configuration prioritizing `/home/nithin/eda/OpenROAD-flow-scripts/dependencies/or-tools` in `CMAKE_PREFIX_PATH` and `absl_DIR`.
  - **Binary Location**: `/home/nithin/eda/OpenROAD-flow-scripts/tools/OpenROAD/build/bin/openroad`
  - **System Symlink**: `/usr/local/bin/openroad`
  - **OpenROAD Version**: `26Q4-110-g91e5f5ff2f` (+GPU +GUI +Python)
- **Physical Design Execution Metrics (`uart_tx` on Nangate45)**:
  - **Floorplan**: Die area `100 x 100 um`, core area `80 x 80 um` (56 rows of 420 sites, Site `FreePDK45_38x28_10R_NP_162NW_34O`).
  - **Pin Placement**: Placed across `metal5` (horizontal) and `metal6` (vertical) tracks.
  - **Global Placement**: Nesterov Global Placement converged at iteration 404 (HPWL: `897.26 um`).
  - **Detailed Placement**: Legalized via negotiation legalizer (0 violations).
  - **Clock Tree Synthesis (CTS)**: TritonCTS created H-Tree topology for `clk` net (33 sinks, 5 clock buffers `CLKBUF_X1`, 5 clock nets, max level 2).
  - **Post-CTS Legalization**: Detailed placement completed with 0 DRC violations.
  - **Global & Detailed Routing**: TritonRoute routed 100% of nets (`metal1`-`metal10`).
  - **Timing Slack Verification**:
    - Hold Slack: `+0.10 ns` (MET)
    - Setup Slack: `+9.50 ns` (MET, target period: `10.0 ns`)
    - Worst Negative Slack (WNS): `0.00 ns`
    - Total Negative Slack (TNS): `0.00 ns`
  - **Exported Physical Artifacts**:
    - `uart_tx_final.def` (117 KB)
    - `uart_tx_final.odb` (948 KB)

---

## Stage 1 Run Telemetry (`manifest.json`)

- **Command**:
  ```bash
  source ~/.venv_wsl/bin/activate
  cd /mnt/e/projects/DFT
  PYTHONPATH=src python3 scripts/run_flow.py --config configs/flows/uart_nangate45.yaml
  ```
- **Execution Run Directory**: `runs/uart_tx_nangate45_20261011_064013`
- **Manifest Results**:
  ```json
  {
    "design": "uart_tx",
    "pdk": "nangate45",
    "flow_backend": "openroad_standalone",
    "status": "success",
    "dry_run": false,
    "dry_run_completed": false,
    "synthesis_completed": true,
    "timestamp": "20261011_064013",
    "run_dir": "/mnt/e/projects/DFT/runs/uart_tx_nangate45_20261011_064013",
    "tool_availability": {
      "wsl_available": true,
      "os_platform": "Linux",
      "execution_environment": "wsl2_linux",
      "yosys": "native:/usr/bin/yosys",
      "openroad": "native:/usr/local/bin/openroad",
      "missing_tools": [],
      "eda_ready": true
    },
    "missing_tools": [],
    "pdk_validation": {
      "valid": true,
      "missing_files": []
    },
    "copied_rtl": [
      "rtl/uart_tx.v"
    ],
    "generated_scripts": [
      "scripts/synth.tcl",
      "scripts/openroad.tcl"
    ],
    "failure_reason": null
  }
  ```

---

## Acceptance Verification

- **Dry-Run Command**:
  ```bash
  PYTHONPATH=src python3 scripts/run_flow.py --config configs/flows/uart_nangate45.yaml --dry-run
  ```
  *Result*: Creates run directory, copies RTL, renders Tcl scripts (`synth.tcl`, `openroad.tcl`), logs folder, and `manifest.json` (`status: "dry_run"`, `dry_run_completed: true`).

- **Unit Test Suite**:
  ```bash
  PYTHONPATH=src python3 -m unittest discover tests/unit
  ```
  *Result*: All 10 unit tests pass cleanly in WSL2 (`Ran 10 tests in 0.454s: OK`).

---

## Current Readiness

> [!TIP]
> **Stage 1 is fully complete and accepted.**
> Both Yosys and OpenROAD are installed, integrated, and verified on real hardware through WSL2. The UART physical flow successfully generates final routed DEF and ODB databases with 100% DRC clean routing and zero timing violations (`WNS = 0.00 ns`, `TNS = 0.00 ns`). No active blockers remain.
> 
> *Stage 2 Progression*: Physical design outputs from Stage 1 (`uart_tx_final.def`) have been successfully validated for spatial feature extraction in **Stage 2a** (see `docs/stage2_status.md`).
