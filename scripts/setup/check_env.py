#!/usr/bin/env python3
"""
Environment & EDA Tooling Check Script for DFT-Aware Hotspot Prediction
"""

import sys
import shutil
import subprocess
import platform
from pathlib import Path

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

def check_python_environment():
    print_header("Python Environment & Key Dependencies")
    print(f"Python Version: {platform.python_version()}")
    print(f"OS Platform: {platform.platform()}")

    core_packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "yaml",
        "tqdm"
    ]

    optional_ml_packages = [
        "xgboost",
        "torch",
        "torch_geometric"
    ]

    print("\n--- Stage 1 Core Dependencies ---")
    for pkg in core_packages:
        try:
            if pkg == "sklearn":
                import sklearn
                ver = sklearn.__version__
            elif pkg == "yaml":
                import yaml
                ver = getattr(yaml, "__version__", "installed")
            else:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg:<18}: OK ({ver})")
        except ImportError:
            print(f"  [X ] {pkg:<18}: MISSING")

    print("\n--- Optional ML Dependencies (Stage 2+) ---")
    for pkg in optional_ml_packages:
        try:
            if pkg == "torch_geometric":
                import torch_geometric
                ver = torch_geometric.__version__
            else:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg:<18}: OK ({ver})")
        except ImportError:
            print(f"  [OPTIONAL] {pkg:<14}: Not installed (Optional for Stage 1)")

    # Check PyTorch CUDA availability if installed
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"\n  PyTorch CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"  CUDA Device Count    : {torch.cuda.device_count()}")
            print(f"  Device Name          : {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

def check_command(cmd_name: str, args: list[str] = ["--version"]) -> str:
    """Checks if a command is executable natively or via wsl."""
    native_path = shutil.which(cmd_name)
    if native_path:
        try:
            res = subprocess.run([native_path] + args, capture_output=True, text=True, timeout=5)
            stdout = res.stdout.strip().replace("\x00", "")
            stderr = res.stderr.strip().replace("\x00", "")
            out = stdout or stderr
            if "not installed" in out.lower():
                return "NOT FOUND (WSL subsystem not installed)"
            first_line = out.split("\n")[0] if out else "Executable found"
            return f"Native: {native_path} ({first_line})"
        except Exception as e:
            return f"Native path found ({native_path}), but error running: {e}"

    # Check via WSL if on Windows
    if platform.system() == "Windows" and shutil.which("wsl"):
        try:
            res = subprocess.run(["wsl", cmd_name] + args, capture_output=True, text=True, timeout=5)
            stdout = res.stdout.strip().replace("\x00", "")
            stderr = res.stderr.strip().replace("\x00", "")
            combined = stdout + stderr
            if res.returncode == 0 and "not installed" not in combined.lower():
                out = combined.split("\n")[0]
                return f"WSL: wsl {cmd_name} ({out})"
        except Exception:
            pass

    return "NOT FOUND"

def check_eda_toolchain():
    print_header("EDA Toolchain Check (Native & WSL2)")

    tools = {
        "yosys": ["-V"],
        "openroad": ["-version"],
        "klayout": ["-v"],
        "fault": ["-h"],
        "wsl": ["--status"],
    }

    for tool, args in tools.items():
        status = check_command(tool, args)
        symbol = "OK" if "NOT FOUND" not in status else "X "
        print(f"  [{symbol}] {tool:<15}: {status}")

def main():
    print("Running DFT-Aware Physical Design Hotspot Environment Check...\n")
    check_python_environment()
    check_eda_toolchain()
    print_header("Check Completed")

if __name__ == "__main__":
    main()
