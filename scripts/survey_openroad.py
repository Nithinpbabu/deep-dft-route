"""
Survey script for introspecting OpenROAD commands and querying DFT/scan capabilities.
"""

import subprocess
import sys


def main():
    tcl_script = """
puts "=== ALL COMMANDS DUMP ==="
set all_cmds [info commands]
puts "TOTAL_COMMANDS: [llength $all_cmds]"

proc find_matching {pat} {
    set matches [info commands $pat]
    puts "MATCH ($pat): [llength $matches] -> $matches"
}

find_matching *dft*
find_matching *scan*
find_matching *test*
find_matching *fault*
find_matching *atpg*
find_matching insert_*
find_matching replace_*
find_matching report_*

# Inspect detailed help for each DFT command
puts "=== COMMAND DETAILED HELP ==="
set target_cmds {
    set_dft_config
    report_dft_config
    report_dft_plan
    execute_dft_plan
    scan_replace
    scan
    scan_opt
}

foreach cmd $target_cmds {
    puts "--------------------------------------------------"
    puts "COMMAND: $cmd"
    if {[catch {eval help $cmd} err]} {
        puts "help $cmd failed: $err"
    }
    # Also test running cmd with invalid/help flag or -help
    if {[catch {eval $cmd -help} err2]} {
        puts "$cmd -help output/error: $err2"
    }
}
exit
"""
    res = subprocess.run(
        ["openroad", "-no_init", "-no_splash"],
        input=tcl_script,
        text=True,
        capture_output=True,
    )
    print("STDOUT:")
    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)

    # 2. Source tree inspection
    print("\n" + "=" * 60)
    print("=== OPENROAD SOURCE TREE INSPECTION ===")
    print("=" * 60)
    or_dir = "/home/nithin/eda/OpenROAD-flow-scripts/tools/OpenROAD"
    
    # Search for files with dft, scan, atpg, fault in filename
    find_cmd = ["find", or_dir, "-type", "f", "(", "-iname", "*dft*", "-o", "-iname", "*scan*", "-o", "-iname", "*atpg*", ")"]
    res_find = subprocess.run(find_cmd, capture_output=True, text=True)
    matching_files = [f for f in res_find.stdout.strip().splitlines() if f]
    print(f"Total matching files by name: {len(matching_files)}")
    for f in matching_files:
        # Show relative path from or_dir
        rel = f.replace(or_dir, "")
        print(f"  FILE: {rel}")

    # Search for documentation or markdown files mentioning DFT
    print("\n--- Documentation & Markdown Files Mentioning DFT ---")
    grep_cmd = ["grep", "-rnI", "-i", "dft", f"{or_dir}/docs"]
    res_grep = subprocess.run(grep_cmd, capture_output=True, text=True)
    doc_lines = [l for l in res_grep.stdout.strip().splitlines() if l]
    print(f"Matches in docs/: {len(doc_lines)}")
    for l in doc_lines[:30]:
        print(f"  {l}")

    # Search in src/ for dft command declarations
    print("\n--- Command Declarations in src/ ---")
    grep_src = ["grep", "-rnI", "set_dft_config", f"{or_dir}/src"]
    res_grep_src = subprocess.run(grep_src, capture_output=True, text=True)
    src_lines = [l for l in res_grep_src.stdout.strip().splitlines() if l]
    print(f"Matches for set_dft_config in src/: {len(src_lines)}")
    for l in src_lines[:30]:
        print(f"  {l}")

    # Search OpenROAD-flow-scripts for DFT references
    print("\n--- OpenROAD-flow-scripts Flow References to DFT ---")
    orfs_dir = "/home/nithin/eda/OpenROAD-flow-scripts/flow"
    grep_orfs = ["grep", "-rnI", "-i", "dft", orfs_dir]
    res_orfs = subprocess.run(grep_orfs, capture_output=True, text=True)
    orfs_lines = [l for l in res_orfs.stdout.strip().splitlines() if l]
    print(f"Matches in ORFS flow/: {len(orfs_lines)}")
    for l in orfs_lines[:30]:
        print(f"  {l}")

    # 3. Fault & other tools check
    print("\n" + "=" * 60)
    print("=== FAULT & EXTERNAL DFT TOOLS CHECK ===")
    print("=" * 60)

    for cmd_name in ["fault", "atpg", "dft", "opensta", "yosys"]:
        which_res = subprocess.run(["which", cmd_name], capture_output=True, text=True)
        path = which_res.stdout.strip()
        print(f"Tool '{cmd_name}': {'FOUND at ' + path if path else 'NOT FOUND'}")
        if path:
            help_res = subprocess.run([cmd_name, "--version"], capture_output=True, text=True)
            print(f"  Version: {help_res.stdout.strip() or help_res.stderr.strip()[:100]}")

    # Check pip for fault
    pip_res = subprocess.run([sys.executable, "-m", "pip", "show", "fault"], capture_output=True, text=True)
    print(f"\nPip package 'fault':\n{pip_res.stdout.strip() or 'Not installed via pip'}")

    # Check /home/nithin/eda for any other tools
    eda_dir = "/home/nithin/eda"
    ls_eda = subprocess.run(["ls", "-la", eda_dir], capture_output=True, text=True)
    print(f"\nDirectory listing of {eda_dir}:\n{ls_eda.stdout.strip()}")


if __name__ == "__main__":
    main()

