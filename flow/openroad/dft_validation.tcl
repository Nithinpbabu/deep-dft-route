# ==============================================================================
# OpenROAD Native DFT Validation Script
# Evaluates set_dft_config, scan_replace, report_dft_plan, and execute_dft_plan
# Stage 3b minimal sequential flow validation
# ==============================================================================

if {![info exists env(TECH_LEF)] || ![info exists env(STDCELL_LEF)] || ![info exists env(LIB_FILE)]} {
    puts "ERROR: Required environment variables TECH_LEF, STDCELL_LEF, or LIB_FILE not set."
    exit 1
}

set tech_lef $env(TECH_LEF)
set stdcell_lef $env(STDCELL_LEF)
set lib_file $env(LIB_FILE)
set verilog_netlist $env(VERILOG_NETLIST)
set top_module [expr {[info exists env(TOP_MODULE)] ? $env(TOP_MODULE) : "dft_counter"}]
set clock_port [expr {[info exists env(CLOCK_PORT)] ? $env(CLOCK_PORT) : "clk"}]
set clock_period [expr {[info exists env(CLOCK_PERIOD)] ? $env(CLOCK_PERIOD) : "10.0"}]
set die_area [expr {[info exists env(DIE_AREA)] ? $env(DIE_AREA) : "0 0 100 100"}]
set core_area [expr {[info exists env(CORE_AREA)] ? $env(CORE_AREA) : "10 10 90 90"}]
set site [expr {[info exists env(SITE)] ? $env(SITE) : "FreePDK45_38x28_10R_NP_162NW_34O"}]
set tracks_file [expr {[info exists env(TRACKS_FILE)] ? $env(TRACKS_FILE) : ""}]
set pin_hor_layer [expr {[info exists env(PIN_HOR_LAYER)] ? $env(PIN_HOR_LAYER) : "metal5"}]
set pin_ver_layer [expr {[info exists env(PIN_VER_LAYER)] ? $env(PIN_VER_LAYER) : "metal6"}]
set max_chains [expr {[info exists env(MAX_CHAINS)] ? $env(MAX_CHAINS) : "1"}]
set max_length [expr {[info exists env(MAX_LENGTH)] ? $env(MAX_LENGTH) : "16"}]
set out_verilog [expr {[info exists env(OUT_VERILOG)] ? $env(OUT_VERILOG) : "dft_output.v"}]
set out_def [expr {[info exists env(OUT_DEF)] ? $env(OUT_DEF) : "dft_output.def"}]
set out_odb [expr {[info exists env(OUT_ODB)] ? $env(OUT_ODB) : "dft_output.odb"}]

puts "============================================================"
puts " Starting OpenROAD DFT Validation Flow"
puts " Design: $top_module"
puts " Tech LEF: $tech_lef"
puts " Stdcell LEF: $stdcell_lef"
puts " Liberty: $lib_file"
puts " Synthesized Netlist: $verilog_netlist"
puts "============================================================"

# Read LEF & Liberty
read_lef $tech_lef
read_lef $stdcell_lef
read_liberty $lib_file

# Read Verilog Netlist & Link Design
read_verilog $verilog_netlist
link_design $top_module

# Clock Constraint
create_clock -name $clock_port -period $clock_period [get_ports $clock_port]

# Floorplan Initialization
initialize_floorplan -die_area $die_area -core_area $core_area -site $site
if {$tracks_file ne "" && [file exists $tracks_file]} {
    puts "Sourcing tracks file: $tracks_file"
    source $tracks_file
}

# --- DFT STEP 1: Configuration ---
puts "=== DFT_STAGE_BEGIN: REPORT_DEFAULT_CONFIG ==="
set rpt_cfg_default_rc [catch {report_dft_config} rpt_cfg_default_out]
puts $rpt_cfg_default_out
puts "=== DFT_STAGE_END: REPORT_DEFAULT_CONFIG ==="

puts "=== DFT_STAGE_BEGIN: SET_CONFIG ==="
set set_cfg_rc [catch {set_dft_config -max_chains $max_chains -max_length $max_length} set_cfg_out]
puts $set_cfg_out
puts "=== DFT_STAGE_END: SET_CONFIG ==="

puts "=== DFT_STAGE_BEGIN: REPORT_CONFIG ==="
set rpt_cfg_rc [catch {report_dft_config} rpt_cfg_out]
puts $rpt_cfg_out
puts "=== DFT_STAGE_END: REPORT_CONFIG ==="

# --- DFT STEP 2: Pre-placement Scan Replacement ---
puts "=== DFT_STAGE_BEGIN: SCAN_REPLACE ==="
set scan_replace_rc [catch {scan_replace} scan_replace_out]
puts $scan_replace_out
puts "=== DFT_STAGE_END: SCAN_REPLACE ==="

# --- Placement ---
puts "=== DFT_STAGE_BEGIN: PLACEMENT ==="
set place_pins_rc [catch {place_pins -hor_layers $pin_hor_layer -ver_layers $pin_ver_layer} place_pins_out]
puts $place_pins_out
set gplace_rc [catch {global_placement -density 0.60} gplace_out]
puts $gplace_out
puts "=== DFT_STAGE_END: PLACEMENT ==="

# --- DFT STEP 3: Post-placement Scan Plan Reporting ---
puts "=== DFT_STAGE_BEGIN: REPORT_DFT_PLAN ==="
set rpt_plan_rc [catch {report_dft_plan -verbose} rpt_plan_out]
puts $rpt_plan_out
puts "=== DFT_STAGE_END: REPORT_DFT_PLAN ==="

# --- DFT STEP 4: Scan Stitching Execution ---
puts "=== DFT_STAGE_BEGIN: EXECUTE_DFT_PLAN ==="
set exec_plan_rc [catch {execute_dft_plan} exec_plan_out]
puts $exec_plan_out
puts "=== DFT_STAGE_END: EXECUTE_DFT_PLAN ==="

# --- Write Result Artifacts ---
puts "Writing post-DFT artifacts..."
if {$out_verilog ne ""} {
    write_verilog $out_verilog
    puts "Wrote Verilog to: $out_verilog"
}
if {$out_def ne ""} {
    write_def $out_def
    puts "Wrote DEF to: $out_def"
}
if {$out_odb ne ""} {
    write_db $out_odb
    puts "Wrote ODB to: $out_odb"
}

puts "=== DFT_VALIDATION_COMPLETE ==="