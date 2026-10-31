# WSL2 Ubuntu Real EDA Environment Setup Guide

This guide details the setup, verification, and execution workflow for running real open-source EDA tools (**Yosys** and **OpenROAD / OpenROAD-flow-scripts**) on Windows using **WSL2 Ubuntu**.

---

## 1. Overview & Architecture

- **Host OS**: Windows 10 / 11
- **Execution Subsystem**: WSL2 (Windows Subsystem for Linux) with Ubuntu 26.04 LTS (Codename: `resolute`), Linux Kernel `6.18.40.1-microsoft-standard-WSL2`
- **Toolchain**:
  - **Yosys**: Installed (`/usr/bin/yosys`, Yosys 0.52) — **Verified Running**
  - **OpenROAD / ORFS**: Installed (`/usr/local/bin/openroad`, OpenROAD 26Q4-110-g91e5f5ff2f) — **Verified Running Full Flow**
- **Containerization**: Native Linux execution via WSL2 (No Docker required or used)

---

## 2. Verified WSL2 Ubuntu Status & Telemetry

```
WSL Version           : 2
Default Distro        : Ubuntu (Running)
Ubuntu Version        : Ubuntu 26.04.1 LTS
Default User          : nithin
Project Mount         : /mnt/e/projects/DFT
Python Virtualenv     : ~/.venv_wsl (Python 3.14.4)
Yosys Binary          : /usr/bin/yosys (Yosys 0.52)
OpenROAD Binary       : /usr/local/bin/openroad (OpenROAD 26Q4-110-g91e5f5ff2f)
ORFS Repository Path  : /home/nithin/eda/OpenROAD-flow-scripts
Local PDK Path        : /mnt/e/projects/DFT/pdk/nangate45
```

---

## 3. PDK / Platform Asset Location & Local Mirror

OpenROAD-flow-scripts platform assets have been cloned and copied locally into the project:

- **Source Location**: `/home/nithin/eda/OpenROAD-flow-scripts/flow/platforms/nangate45/`
- **Project PDK Mirror**: `/mnt/e/projects/DFT/pdk/nangate45/`
  - Liberty File: `NangateOpenCellLibrary_typical.lib` (6.69 MB)
  - Tech LEF File: `NangateOpenCellLibrary.tech.lef` (19.4 KB)
  - StdCell LEF File: `NangateOpenCellLibrary.macro.lef` (349.9 KB)
  - Routing Tracks: `make_tracks.tcl` (routing track pitches and offsets for `metal1`-`metal10`)

---

## 4. Installed EDA Toolchain Status

### 1. Yosys Synthesis Engine: INSTALLED & VERIFIED REAL RUN
- Installed via `sudo apt install -y yosys`
- Location: `/usr/bin/yosys`
- Version: `Yosys 0.52 (git sha1 fee39a3284c90249e1d9684cf6944ffbbcbb8f90)`
- **Execution Results**: Successfully synthesizes `uart_tx.v` RTL to 147 Nangate45 standard cells (Area: 299.782 um²) and generates `scripts/synth_uart_tx.v` and `logs/synth.log`.

### 2. OpenROAD Physical Design Engine: INSTALLED & VERIFIED FULL FLOW
- Source Tree: `/home/nithin/eda/OpenROAD-flow-scripts/tools/OpenROAD`
- Binary Location: `/usr/local/bin/openroad`
- Version: `26Q4-110-g91e5f5ff2f` (+GPU +GUI +Python)
- **Resolved Build Items**:
  1. System libraries `libzstd-dev` and `libicu-dev` installed via apt.
  2. Abseil linker resolution: OR-Tools bundled Abseil prioritized in `CMAKE_PREFIX_PATH` and `absl_DIR`.
- **Execution Results**: Complete physical design flow verified:
  - Floorplanning (Die: 100x100 µm, Core: 80x80 µm)
  - Pin placement across `metal5`/`metal6` tracks
  - Global & detailed placement (HPWL 897.26 µm, 0 violations)
  - TritonCTS clock tree synthesis (33 sinks, 5 `CLKBUF_X1`, max level 2)
  - Global & detailed routing (TritonRoute 100% routed, 0 DRC violations)
  - Timing closure: Setup slack +9.50 ns (MET), Hold slack +0.10 ns (MET), WNS 0.00 ns, TNS 0.00 ns
  - Output artifacts: `uart_tx_final.def` (117 KB) and `uart_tx_final.odb` (948 KB)

---

## 5. Execution & Manifest Status Taxonomy (WSL2 Tested)

- **Dry-Run Mode**:
  ```bash
  python3 scripts/run_flow.py --config configs/flows/uart_nangate45.yaml --dry-run
  ```
  *Result*: `"status": "dry_run"`, `"dry_run_completed": true`.

- **Full Real Flow Execution**:
  ```bash
  PYTHONPATH=src python3 scripts/run_flow.py --config configs/flows/uart_nangate45.yaml
  ```
  *Result*: `"status": "success"`, `"synthesis_completed": true`, produces final routed DEF/ODB and logs.

---

## 6. Verification Checklist

- [x] WSL2 enabled and Ubuntu running on Version 2.
- [x] Isolated Python virtual environment `~/.venv_wsl` initialized with project dependencies.
- [x] `python3 -m unittest discover tests/unit` passes inside WSL2 (38/38 OK).
- [x] Yosys binary verified inside WSL2 (`Yosys 0.52`).
- [x] Real Yosys synthesis execution verified on `uart_tx` RTL + Nangate45 PDK (`synth_uart_tx.v` generated).
- [x] Nangate45 PDK files (`.lib`, `.tech.lef`, `.macro.lef`, `make_tracks.tcl`) mirrored to `pdk/nangate45/`.
- [x] OpenROAD binary built from source, installed to `/usr/local/bin/openroad`, and verified (`OpenROAD 26Q4-110-g91e5f5ff2f`).
- [x] Full physical design flow executed with 100% routing completion and timing closure (`status: "success"`).
- [x] Flow dry-run verified inside WSL2 (`status: "dry_run"`).

