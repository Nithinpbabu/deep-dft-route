# Open-Source DFT / Scan Capability Survey (Stage 3a)

**Survey Date/Time**: 2026-10-11 17:42 UTC (23:12 IST)  
**Host Subsystem**: WSL2 Ubuntu 26.04.1 LTS (`resolute`), Linux Kernel `6.18.40.1-microsoft-standard-WSL2`  
**Python Environment**: Isolated virtualenv `~/.venv_wsl` (Python 3.14.4)  
**OpenROAD Binary**: `/usr/local/bin/openroad`  
**OpenROAD Version**: `26Q4-110-g91e5f5ff2f` (+GPU +GUI +Python)  
**OpenROAD Source Tree**: `/home/nithin/eda/OpenROAD-flow-scripts/tools/OpenROAD/`  

---

## 1. Survey Scope & Methodology Notice

> [!IMPORTANT]
> **Methodological Disclaimer**:  
> **No scan insertion or ATPG capability is claimed at Stage 3a; this is a capability survey only.**  
> The purpose of this survey is strictly to inspect, verify, and document locally available DFT commands, source modules, and toolchain capabilities before implementing any DFT workflows or synthetic metadata in Stage 3b/3c.

---

## 2. OpenROAD Native DFT Command Survey

Introspection was performed directly against the active OpenROAD binary (`/usr/local/bin/openroad`) in non-interactive batch mode without user initialization (`-no_init -no_splash`).

### Command Availability Table

| Command / Term | Available? | Evidence Command | Direct Output / Signature | Notes |
| :--- | :---: | :--- | :--- | :--- |
| `set_dft_config` | **YES** | `info commands set_dft_config` | `set_dft_config [-max_length max_length] [-max_chains max_chains] [-clock_mixing clock_mixing] [-scan_enable_name_pattern pat] [-scan_in_name_pattern pat] [-scan_out_name_pattern pat]` | Configures scan chain parameters (length, chain count, clock mixing mode, port naming patterns). |
| `report_dft_config`| **YES** | `info commands report_dft_config` | `report_dft_config` | Dumps current DFT architecture configuration. |
| `scan_replace` | **YES** | `info commands scan_replace` | `scan_replace` | Replaces standard flip-flops with scan-equivalent cells (`SDFF`) using Liberty `test_cell` definitions before placement. |
| `report_dft_plan` | **YES** | `info commands report_dft_plan` | `report_dft_plan [-verbose]` | Generates preview of scan chains to be stitched. Run after `scan_replace` and placement. |
| `execute_dft_plan`| **YES** | `info commands execute_dft_plan`| `execute_dft_plan` | Stitches scan chains to minimize routed wirelength based on placement. |
| `scan_opt` | **YES** (Stub) | `info commands scan_opt` | `scan_opt` | Documented as **NOP** (Not yet implemented in current OpenROAD). |
| `scan` | **NO** (Tcl) | `info commands scan` | `scan string format ?varName ...?` | Standard Tcl built-in string scanner; not an EDA command. |
| `insert_dft` | **NO** | `info commands insert_dft` | Not found (`llength == 0`) | Not part of OpenROAD syntax; OpenROAD uses `scan_replace` + `execute_dft_plan`. |
| `preview_dft` | **NO** | `info commands preview_dft` | Not found (`llength == 0`) | OpenROAD uses `report_dft_plan [-verbose]`. |
| `report_dft` | **NO** | `info commands report_dft` | Not found (`llength == 0`) | OpenROAD uses `report_dft_config` and `report_dft_plan`. |
| `atpg` | **NO** | `info commands *atpg*` | Not found (`llength == 0`) | OpenROAD does not have ATPG engine built-in. |
| `fault` | **NO** | `info commands *fault*` | Only `detailed_route_set_default_via` | No Fault/ATPG integration inside OpenROAD binary. |

---

## 3. OpenROAD Source Tree & Documentation Findings

Inspection of the local OpenROAD source tree at `/home/nithin/eda/OpenROAD-flow-scripts/tools/OpenROAD/` confirmed an active, dedicated C++ DFT engine:

### 1. Dedicated Subsystem Directory
- **Path**: `src/dft/`
- **Module Architecture**:
  - `src/dft/src/Dft.cpp`, `src/dft/src/MakeDft.cpp`: Core DFT tool initialization and DB hooks.
  - `src/dft/src/replace/ScanReplace.cpp`, `ScanReplace.hh`: Flop replacement matching non-scan cells to scan cells via `sta::LibertyCell` and `sta::TestCell`.
  - `src/dft/src/cells/ScanCellFactory.cpp`, `ScanCell.cpp`, `OneBitScanCell.cpp`: Cell abstraction for single-bit scan cells.
  - `src/dft/src/architect/ScanArchitect.cpp`, `ScanArchitectHeuristic.cpp`, `ScanChain.cpp`: Scan chain partitioning, length balancing, and clock-domain mixing (`no_mix` vs `clock_mix`).
  - `src/dft/src/stitch/ScanStitch.cpp`, `ScanStitch.hh`: Scan chain connection and net stitching.
  - `src/dft/src/dft.tcl`, `src/dft/src/dft.i`: SWIG/Tcl command bindings and argument parsing.

### 2. Module Documentation (`src/dft/README.md`)
OpenROAD documents the standard native DFT insertion flow:
```tcl
set_dft_config -max_length 10 -clock_mixing clock_mix
report_dft_config
scan_replace
# Run global placement...
report_dft_plan -verbose
execute_dft_plan
```

### 3. OpenROAD DFT Regression Test Suite
Under `src/dft/test/`, 6 regression test scripts exist:
- `scan_architect_clock_mix_sky130.tcl`
- `one_cell_sky130.tcl`
- `scan_architect_register_bank_no_clock_mix_sky130.tcl`
- `scan_architect_no_mix_sky130.tcl`
- `max_chain_count_sky130.tcl`
- `place_sort_sky130.tcl`

---

## 4. PDK Compatibility Analysis for Native DFT

`ScanCellFactory.cpp` and `ScanReplace.cpp` require the target standard cell library to provide explicit Liberty `test_cell ()` construct annotations with `signal_type : "test_scan_in"` and `signal_type : "test_scan_enable"`:

| PDK | `SDFF` Cells Available? | Liberty `test_cell ()` Syntax? | OpenROAD Native `scan_replace` Support |
| :--- | :---: | :---: | :--- |
| **SkyWater 130 (`sky130hd`)** | **YES** (`sky130_fd_sc_hd__sdfsbp_*`) | **YES** (Over 10 `test_cell ()` blocks defined) | **Full Native Support** (Verified by OpenROAD unit tests) |
| **Nangate45** (`NangateOpenCellLibrary_typical.lib`) | **YES** (`SDFF_X1`, `SDFF_X2`, `SDFFR_X1`, `SDFFRS_X1`) | **NO** (`test_cell ()` absent in standard distribution) | **Requires Liberty annotation** OR **Direct synthesis mapping to `SDFF`** |

---

## 5. Fault & External DFT Tools Availability

A survey of external open-source ATPG/DFT packages was conducted across system packages, Python packages, and local directories:

| Tool / Check | Status | Verification Command | Notes / Findings |
| :--- | :---: | :--- | :--- |
| **Fault Binary** (`fault`) | **NOT FOUND** | `which fault` | Not present in system `$PATH` or local virtualenv. |
| **Fault CLI Help** | **NOT FOUND** | `fault --help` | Command not available. |
| **Fault Pip Package** | **NOT FOUND** | `pip show fault` | Not installed in Python virtual environment (`~/.venv_wsl`). |
| **System ATPG Tool** (`atpg`) | **NOT FOUND** | `which atpg` | No standalone ATPG tool in Ubuntu packages. |
| **Local EDA Repositories** | Inspected | `ls -la /home/nithin/eda` | Contains `OpenROAD-flow-scripts`, `cudd`, `lemon-graph`. No Fault repository present. |
| **Yosys DFT Features** | Limited | `yosys -p 'help' \| grep -i dft` | Only `dft_tag` (data-flow tracking for security verification). No scan chain insertion engine. |

### Feasibility of Fault Installation Later:
- **Architecture**: Fault is an open-source DFT/ATPG tool developed by Cloud-V / efabless.
- **Dependencies**: Historically designed for Ubuntu 18.04 and packaged via Docker. Native compilation on Ubuntu 26.04 without Docker requires Swift compiler runtimes and custom C++ toolchain integration.
- **Assessment**: Attempting a complex Fault installation now would contradict project rules (no Docker, no large external dependencies without prior proof of necessity). Since OpenROAD already provides native scan chain insertion and stitching commands, installing Fault is not currently required for physical layout hotspot prediction.

---

## 6. Known OpenROAD DFT Limitations

Based on OpenROAD documentation and source inspection:
1. **`scan_opt` is a NOP**: Scan chain re-ordering after detailed placement is not yet functional.
2. **One-Bit Cells Only**: The current `ScanCellFactory` only handles 1-bit scan registers (no multi-bit scan cell banking).
3. **No User-Defined Scan Ordering**: Chains are ordered automatically by the `ScanArchitectHeuristic` to minimize estimated wirelength.
4. **Liberty Requirement**: Library must contain `test_cell ()` descriptions for automatic `scan_replace`. Without `test_cell ()`, non-scan flops cannot be replaced automatically, but pre-mapped scan flops (`SDFF`) can still be stitched by `execute_dft_plan`.

---

## 7. Recommended Stage 3 Substage Sequence

Based on empirical findings:

1. **Stage 3b: Minimal Sequential DFT Flow Validation**:
   - Construct a minimal sequential Verilog test block with ~10–20 flip-flops.
   - Test OpenROAD native DFT commands (`set_dft_config`, `report_dft_plan`, `execute_dft_plan`).
   - Validate whether scan stitching produces real scan pins (`scan_in`, `scan_out`, `scan_enable`) and physical scan routes in the resulting DEF.
   - If using Nangate45: Validate direct synthesis of `SDFF` cells or Liberty `test_cell ()` patch; compare against Sky130 where `test_cell ()` is already verified.
2. **Stage 3c: Scan-Enabled Physical Benchmark Flow**:
   - Integrate verified scan flow into `ScriptGenerator` and `FlowRunner`.
   - Run benchmark designs with and without scan to observe physical routing differences.
3. **Fault Status**:
   - Defer external Fault installation indefinitely unless ATPG test vectors or fault coverage metrics are strictly demanded by future experimental goals.

---

## 8. Stage 3b Minimal Sequential DFT Flow Validation (Empirical Findings)

**Validation Date**: 2026-10-11 18:50 UTC (00:20 IST)  
**Sequential Benchmark**: `rtl/dft_smoke/dft_counter.v` (16-bit counter, synchronous load/enable, asynchronous active-low reset `rst_n`, exactly 16 flip-flops)  
**Validation Driver**: `scripts/validate_dft.py` calling `flow/openroad/dft_validation.tcl`  
**Automated Unit Tests**: `tests/unit/test_dft_validation.py` (5 tests passing)  

### 8.1 Summary Comparison Table

| Metric / Dimension | Track A: Sky130hd (Primary Candidate) | Track B: Nangate45 (Negative Control) |
| :--- | :--- | :--- |
| **Rationale** | Primary candidate because local assets appear to include Liberty `test_cell()` blocks. | Retained strictly as negative control; no Liberty patching in Stage 3b. |
| **Liberty `test_cell()` Present?** | **YES** | **NO** |
| **Baseline Non-DFT Flow** | **SUCCESS** (`runs/dft_counter_sky130hd_20261011_184659`) | **SUCCESS** (`runs/dft_counter_nangate45_20261011_184602`) |
| **Commands Attempted** | `report_dft_config`, `set_dft_config`, `scan_replace`, `report_dft_plan`, `execute_dft_plan` | `report_dft_config`, `set_dft_config`, `scan_replace`, `report_dft_plan`, `execute_dft_plan` |
| **Commands Succeeded** | **5 / 5** (All commands succeeded) | **2 / 5** (`report_dft_config`, `set_dft_config`) |
| **`scan_replace` Outcome** | **SUCCESS** (16 FFs substituted with `sky130_fd_sc_hd__sdfrtp_1`) | **FAILED** (`[WARNING DFT-0002] No scan equivalent lib cell found`) |
| **`report_dft_plan` Outcome** | **SUCCESS** (Reported 1 chain with 16 cells) | **FAILED** (`[WARNING DFT-0007] Cell is not a scan cell. Number of chains: 0`) |
| **`execute_dft_plan` Outcome**| **SUCCESS** (Scan chain stitched) | **NOP** (0 chains stitched) |
| **DEF Evidence (`SCANCHAINS`)** | **YES** (`SCANCHAINS 1; - chain_0 ... 16 cells`) | **NO** (`SCANCHAINS` block absent) |
| **Netlist Evidence (Scan Ports)**| **YES** (`scan_enable_0`, `scan_in_0`, `q[8]`) | **NO** (No scan ports generated) |
| **Failure Attribution** | None (Validation passed) | Command: `scan_replace`, Reason: Missing Liberty `test_cell()` metadata |
| **Output Run Directory** | `data/dft_validation/dft_smoke_sky130hd_20261011_185057` | `data/dft_validation/dft_smoke_nangate45_20261011_185058` |

---

### 8.2 Track A: Sky130hd Scan Evidence

#### 1. DEF `SCANCHAINS` Block (`dft_counter_scan.def`)
```def
SCANCHAINS 1 ;

- chain_0
+ START PIN scan_in_0
+ ORDERED
  _142_ ( IN SCD ) ( OUT Q )
  _135_ ( IN SCD ) ( OUT Q )
  _136_ ( IN SCD ) ( OUT Q )
  _137_ ( IN SCD ) ( OUT Q )
  _138_ ( IN SCD ) ( OUT Q )
  _134_ ( IN SCD ) ( OUT Q )
  _141_ ( IN SCD ) ( OUT Q )
  _140_ ( IN SCD ) ( OUT Q )
  _139_ ( IN SCD ) ( OUT Q )
  _149_ ( IN SCD ) ( OUT Q )
  _148_ ( IN SCD ) ( OUT Q )
  _147_ ( IN SCD ) ( OUT Q )
  _144_ ( IN SCD ) ( OUT Q )
  _145_ ( IN SCD ) ( OUT Q )
  _146_ ( IN SCD ) ( OUT Q )
  _143_ ( IN SCD ) ( OUT Q )
+ PARTITION default
+ STOP PIN q[8] ;

END SCANCHAINS
```

#### 2. Generated Scan Ports in Verilog Netlist (`dft_counter_scan.v`)
```verilog
module dft_counter (
    clk,
    load,
    rst_n,
    ...
    scan_enable_0,
    scan_in_0
);
```

#### 3. Structured Summary Output (`dft_validation_summary.json`)
```json
{
  "platform": "sky130hd",
  "design": "dft_counter",
  "openroad_version": "26Q4-110-g91e5f5ff2f",
  "liberty_has_test_cell": true,
  "commands_attempted": [
    "report_dft_config",
    "set_dft_config",
    "scan_replace",
    "report_dft_plan",
    "execute_dft_plan"
  ],
  "commands_succeeded": [
    "report_dft_config",
    "set_dft_config",
    "scan_replace",
    "report_dft_plan",
    "execute_dft_plan"
  ],
  "failure_command": null,
  "failure_reason": null,
  "scan_replace_succeeded": true,
  "report_dft_plan_succeeded": true,
  "execute_dft_plan_succeeded": true,
  "scan_metadata_available": true,
  "generated_scan_ports_or_nets": [
    "scan_enable_0",
    "scan_in_0"
  ],
  "limitations": []
}
```

---

### 8.3 Track B: Nangate45 Negative Control Evidence

#### 1. OpenROAD Diagnostic Messages (`dft_validation.log`)
```
[WARNING DFT-0002] Can't scan replace cell '_016_', that has lib cell 'DFFR_X1'. No scan equivalent lib cell found
[WARNING DFT-0002] Can't scan replace cell '_017_', that has lib cell 'DFFR_X1'. No scan equivalent lib cell found
...
[WARNING DFT-0007] Cell '_016_' is not a scan cell. Can't use it for scan architect
...
***************************
Report DFT Plan
Number of chains: 0
Clock domain: No Mix
***************************
```

#### 2. Structured Summary Output (`dft_validation_summary.json`)
```json
{
  "platform": "nangate45",
  "design": "dft_counter",
  "openroad_version": "26Q4-110-g91e5f5ff2f",
  "liberty_has_test_cell": false,
  "commands_attempted": [
    "report_dft_config",
    "set_dft_config",
    "scan_replace",
    "report_dft_plan",
    "execute_dft_plan"
  ],
  "commands_succeeded": [
    "report_dft_config",
    "set_dft_config"
  ],
  "failure_command": "scan_replace",
  "failure_reason": "No scan equivalent lib cell found (Liberty lacks test_cell annotations with test_scan_in/enable/out attributes).",
  "scan_replace_succeeded": false,
  "report_dft_plan_succeeded": false,
  "execute_dft_plan_succeeded": false,
  "scan_metadata_available": false,
  "generated_scan_ports_or_nets": [],
  "limitations": [
    "Platform 'nangate45' Liberty library lacks test_cell() annotations required by OpenROAD native scan_replace.",
    "Nangate45 native scan insertion blocked; requires Stage 3d synthetic scan metadata or Liberty test_cell patching."
  ]
}
```

---

### 8.4 Stage 3b Conclusions & Path Forward

1. **Native OpenROAD DFT Viability**:
   - OpenROAD native DFT commands (`scan_replace`, `report_dft_plan`, `execute_dft_plan`) are fully functional and capable of inserting, architecting, and stitching real scan chains into DEF and gate-level netlists.
2. **Library Dependency Confirmed**:
   - Automated scan replacement strictly depends on Liberty `test_cell()` blocks. Sky130hd standard cells provide this natively, while Nangate45 lacks it.
3. **Decoupled Architecture**:
   - For physical design flow benchmarking with native DFT, Sky130hd provides the verified reference path.
   - For Nangate45 experiments where PDK change is not desired, Stage 3d synthetic scan metadata generation provides the ideal decoupled bridge without corrupting baseline library models.

---

## 9. Stage 3c Scan Metadata Extraction (Machine-Readable Representation)

**Extraction Date**: 2026-10-12 06:53 UTC (12:23 IST)  
**Input Artifact Directory**: `data/dft_validation/dft_smoke_sky130hd_20261011_185057`  
**Schema Version**: `scan_metadata_v001`  
**Primary Extractor**: `src/dft_hotspot/dft/scan_metadata.py` (`ScanMetadataExtractor`)  
**CLI Tool**: `scripts/extract_scan_metadata.py`  
**Automated Tests**: `tests/unit/test_scan_metadata.py` (8 new tests, 53 total suite tests passing)  

### 9.1 Summary of Extracted Scan Structures

- **Design**: `dft_counter`
- **Platform**: `sky130hd`
- **Source DEF**: `data/dft_validation/dft_smoke_sky130hd_20261011_185057/dft_counter_scan.def`
- **Source Netlist**: `data/dft_validation/dft_smoke_sky130hd_20261011_185057/dft_counter_scan.v`
- **Chains Extracted**: 1 (`chain_0`)
- **Total Scan Cells**: 16 (16 bits)
- **Placed Scan Cells**: 16 / 16 (100% placed and mapped)
- **Cell Ordering**: Exactly preserved from DEF `+ ORDERED` block (`_142_` -> `_135_` -> ... -> `_143_`)
- **Start Endpoint**: `PIN scan_in_0`
- **Stop Endpoint**: `PIN q[8]` (OpenROAD DFT terminates at functional output pin)
- **Scan Ports Detected**:
  - `scan_enable`: `["scan_enable_0"]`
  - `scan_in`: `["scan_in_0"]`
  - `scan_out`: `["q[8]"]`
- **Placement Enrichment**:
  - All 16 cells mapped to master cell `sky130_fd_sc_hd__sdfrbp_1`
  - All 16 cells enriched with DBU coordinates `(x, y)` and physical layout micron coordinates `(x_um, y_um)`
  - Placed status set to `True` based on DEF `COMPONENTS` section

### 9.2 Machine-Readable Output Artifact (`scan_metadata.json`)

The structured metadata is persisted in `data/dft_validation/dft_smoke_sky130hd_20261011_185057/scan_metadata.json` alongside `scan_metadata_summary.txt`:

```json
{
  "schema_version": "scan_metadata_v001",
  "source": "openroad_dft",
  "design": "dft_counter",
  "platform": "sky130hd",
  "source_dir": "data/dft_validation/dft_smoke_sky130hd_20261011_185057",
  "def_path": "data/dft_validation/dft_smoke_sky130hd_20261011_185057/dft_counter_scan.def",
  "netlist_path": "data/dft_validation/dft_smoke_sky130hd_20261011_185057/dft_counter_scan.v",
  "created_at": "2026-10-12T06:53:33.268096+00:00",
  "scan_config": {
    "max_chains": 1,
    "max_length": 16,
    "clock_mixing": "No Mix",
    "scan_enable_name_pattern": "scan_enable_{}",
    "scan_in_name_pattern": "scan_in_{}",
    "scan_out_name_pattern": "scan_out_{}"
  },
  "scan_ports": {
    "scan_enable": ["scan_enable_0"],
    "scan_in": ["scan_in_0"],
    "scan_out": ["q[8]"]
  },
  "chains": [
    {
      "chain_id": "chain_0",
      "start": { "type": "PIN", "name": "scan_in_0" },
      "stop": { "type": "PIN", "name": "q[8]" },
      "partition": "default",
      "cell_count": 16,
      "cells": [
        {
          "index": 0,
          "instance": "_142_",
          "scan_in_pin": "SCD",
          "scan_out_pin": "Q",
          "master": "sky130_fd_sc_hd__sdfrbp_1",
          "x": 43513,
          "y": 23256,
          "x_um": 43.513,
          "y_um": 23.256,
          "placed": true
        }
      ]
    }
  ],
  "statistics": {
    "chain_count": 1,
    "total_scan_cells": 16,
    "min_chain_length": 16,
    "max_chain_length": 16,
    "avg_chain_length": 16.0,
    "placed_scan_cells": 16
  },
  "limitations": [
    "Extracted from OpenROAD DEF SCANCHAINS; no ATPG patterns generated.",
    "Scan-out stop may terminate at functional output if OpenROAD DFT uses existing output pin."
  ]
}
```

---

## 7. Stage 3d: Deterministic Synthetic Scan Metadata Fallback for Nangate45

### 1. Motivation & Boundary Condition
As proven during Stage 3b Track B validation, standard open-source Nangate45 FreePDK45 Liberty files (`NangateOpenCellLibrary_typical.lib`) lack OpenROAD-compatible `test_cell()` definitions. When `scan_replace` executes, OpenROAD issues:
```
[WARNING DFT-0002] No scan equivalent lib cell found for cell DFF_X1.
[WARNING DFT-0002] No scan equivalent lib cell found for cell DFFR_X1.
```
This is a standard library metadata limitation rather than a flow failure. The project plan explicitly provisions a fallback:
> "Use sequential-cell chain construction as a proxy scan chain; label it as synthetic scan metadata; focus first on scan-chain routing demand rather than full ATPG."

### 2. Implementation & Ordering Heuristic
Implemented in `src/dft_hotspot/dft/synthetic_scan.py` and CLI `scripts/generate_synthetic_scan_metadata.py`:
- Extracts placed sequential cell components from layout DEF (`uart_tx_final.def`) matching regex `SEQUENTIAL_PATTERN` (`DFF`, `DFFR`, `DFFS`).
- Applies deterministic row-major spatial ordering (`row_major_by_y_then_x`): cells sorted by $y$ ascending, then $x$ ascending, then instance name as unique tie-break.
- Assembles synthetic scan chains (`synthetic_chain_0`) with endpoints `SYNTHETIC synthetic_scan_in_0` and `SYNTHETIC synthetic_scan_out_0`.
- All metadata is generated under canonical schema `scan_metadata_v001` with `source: "synthetic_proxy"`.

### 3. Empirical Results for Nangate45 UART Run (`runs/uart_tx_nangate45_20261011_064013`)
- **Total Components**: 154
- **Sequential Cells Extracted**: Exactly 33 (`32x DFFR_X1`, `1x DFFS_X1`)
- **Placed Sequential Cells**: 33 (100% placed)
- **Chain Count**: 1 (`synthetic_chain_0`, 33 cells)
- **Output Artifacts**:
  - JSON: `data/processed/samples/uart_tx_nangate45_20261011_064013/scan_metadata.json`
  - Summary: `data/processed/samples/uart_tx_nangate45_20261011_064013/scan_metadata_summary.txt`
- **Limitations Enforced**:
  - Synthetic proxy scan metadata only; no scan replacement or ATPG was performed.
  - Generated from placed sequential cells in Nangate45 DEF.
  - Scan pins are unknown because cells were not replaced with real scan flops.
  - Use only for scan-chain routing demand proxy experiments.


