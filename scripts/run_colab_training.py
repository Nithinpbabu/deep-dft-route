#!/usr/bin/env python3
"""
DeepDFT-Route: Google Colab Remote Execution Runner.
Automates end-to-end training on Google Colab Pro GPU using google-colab-cli:
1. Checks authentication (ADC credentials).
2. Provisions a cloud VM session with GPU (T4/A100).
3. Uploads self-contained package (dataset_multi_pdk.tar.gz).
4. Runs 30-epoch Multi-PDK training in the cloud.
5. Downloads best_model.pt and evaluation report.
6. Automatically stops session to preserve compute units.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
COLAB_BIN = Path.home() / ".venv_ml" / "bin" / "colab"
ADC_PATH = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def run_cmd(
    cmd: list[str],
    check: bool = True,
    stdin_text: Optional[str] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Runs a shell command and streams or captures output."""
    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / '.venv_ml' / 'bin'}:{Path.home() / 'google-cloud-sdk' / 'bin'}:{env.get('PATH', '')}"
    if capture:
        return subprocess.run(
            cmd,
            env=env,
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )
    return subprocess.run(cmd, env=env, input=stdin_text, text=True if stdin_text else None, check=check)


def check_auth() -> bool:
    """Verifies that Google Cloud ADC credentials exist."""
    return ADC_PATH.is_file()


def main():
    parser = argparse.ArgumentParser(description="DeepDFT-Route Colab Pro Cloud Training Runner")
    parser.add_argument("--gpu", type=str, default="T4", choices=["T4", "A100", "L4"], help="Colab GPU accelerator")
    parser.add_argument("--session", type=str, default="dft-multi-pdk", help="Colab session name")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--output-dir", type=str, default="experiments/fusion/multi_pdk_conditioned_v001", help="Local output directory")
    args = parser.parse_args()

    local_archive = ROOT_DIR / "data" / "dataset_multi_pdk.tar.gz"
    if not local_archive.is_file():
        print(f"Error: Dataset archive not found at {local_archive}")
        return 1

    local_out = ROOT_DIR / args.output_dir
    local_out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" DEEPDFT-ROUTE: GOOGLE COLAB PRO CLOUD TRAINING RUNNER")
    print(f" Target Accelerator : {args.gpu} GPU (Google Colab Pro)")
    print(f" Session Name       : {args.session}")
    print(f" Local Archive      : {local_archive} ({local_archive.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 80)

    # 1. Check Authentication
    if not check_auth():
        print("\n[!] Google Cloud ADC credentials not found!")
        print("Please authenticate your Google account (with your Jio Colab Pro plan) by running:")
        print("\n  wsl bash -c 'gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory'\n")
        return 1

    colab = str(COLAB_BIN) if COLAB_BIN.is_file() else "colab"

    try:
        # 2. Provision VM
        print(f"\n[1/5] Provisioning remote Colab VM with {args.gpu} GPU (session: {args.session})...")
        run_cmd([colab, "--auth=adc", "new", "-s", args.session, "--gpu", args.gpu])

        # 3. Upload dataset bundle
        if local_archive.stat().st_size > 45 * 1024 * 1024:
            print(f"\n[2/5] Splitting and uploading large bundle ({local_archive.stat().st_size / 1024 / 1024:.2f} MB in ~38MB chunks)...")
            split_prefix = str(local_archive.parent / "part_")
            for old_p in local_archive.parent.glob("part_*"):
                old_p.unlink()
            subprocess.run(["split", "-b", "38M", str(local_archive), split_prefix], check=True)
            parts = sorted(list(local_archive.parent.glob("part_*")))
            for p in parts:
                print(f"  * Uploading chunk {p.name} ({p.stat().st_size / 1024 / 1024:.2f} MB)...")
                run_cmd([colab, "--auth=adc", "upload", "-s", args.session, str(p), f"/content/{p.name}"])
            # Reassemble on remote Colab VM
            reassemble_code = (
                "import subprocess\n"
                "print('[colab] Reassembling dataset archive from chunks...')\n"
                "subprocess.run('cat /content/part_* > /content/dataset_multi_pdk.tar.gz', shell=True, check=True)\n"
                "subprocess.run('rm -f /content/part_*', shell=True, check=True)\n"
                "print('[colab] Reassembled archive successfully!')\n"
            )
            reassemble_file = Path("/tmp/colab_reassemble.py")
            reassemble_file.write_text(reassemble_code, encoding="utf-8")
            run_cmd([colab, "--auth=adc", "exec", "-s", args.session, "--timeout", "180", "-f", str(reassemble_file)])
        else:
            print(f"\n[2/5] Uploading self-contained bundle ({local_archive.stat().st_size / 1024 / 1024:.2f} MB)...")
            run_cmd([colab, "--auth=adc", "upload", "-s", args.session, str(local_archive), "/content/dataset_multi_pdk.tar.gz"])

        # 4. Extract bundle and install dependencies on remote VM
        print("\n[3/5] Setting up environment on remote Colab VM...")
        setup_code = (
            "import subprocess\n"
            "print('[colab] Extracting dataset and code bundle...')\n"
            "subprocess.run(['tar', '-xzf', '/content/dataset_multi_pdk.tar.gz', '-C', '/content'], check=True)\n"
            "print('[colab] Installing torch-geometric and dependencies...')\n"
            "subprocess.run(['pip', 'install', '-q', 'torch-geometric', 'scipy', 'matplotlib'], check=True)\n"
            "print('[colab] Remote environment setup completed successfully!')\n"
        )
        setup_file = Path("/tmp/colab_setup.py")
        setup_file.write_text(setup_code, encoding="utf-8")
        run_cmd([colab, "--auth=adc", "exec", "-s", args.session, "--timeout", "300", "-f", str(setup_file)])

        # 5. Run Training
        print(f"\n[4/5] Executing {args.epochs}-epoch Multi-PDK training on Colab {args.gpu} GPU...")
        train_code = (
            "import subprocess, sys, os\n"
            "env = os.environ.copy()\n"
            "env['PYTHONPATH'] = '/content/src:' + env.get('PYTHONPATH', '')\n"
            "cmd = [\n"
            "    sys.executable, '-u', '/content/scripts/train_multi_pdk.py',\n"
            "    '--dataset-dir', 'multi_pdk_v001',\n"
            f"    '--output-dir', '{args.output_dir}',\n"
            f"    '--epochs', '{args.epochs}',\n"
            "    '--batch-size', '2',\n"
            "    '--lr', '0.001',\n"
            "    '--loss-type', 'correlation_aware',\n"
            "    '--lambda-corr', '0.15',\n"
            "    '--early-stopping-patience', '25',\n"
            "]\n"
            "print('[colab] Starting training run across 45nm + 14nm FinFET layouts...')\n"
            "proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in proc.stdout:\n"
            "    print(line, end='', flush=True)\n"
            "proc.wait()\n"
            "if proc.returncode != 0:\n"
            "    raise RuntimeError(f'Training failed with exit code {proc.returncode}')\n"
            "print('[colab] Training finished successfully!')\n"
        )
        train_file = Path("/tmp/colab_train.py")
        train_file.write_text(train_code, encoding="utf-8")
        t0 = time.time()
        run_cmd([colab, "--auth=adc", "exec", "-s", args.session, "--timeout", "3600", "-f", str(train_file)])
        print(f"Remote training completed in {time.time() - t0:.1f}s.")

        # 6. Download model and results
        print("\n[5/5] Downloading trained checkpoint and evaluation report from Colab...")
        remote_ckpt = f"/content/{args.output_dir}/best_model.pt"
        remote_eval = f"/content/{args.output_dir}/cross_technology_evaluation.json"
        local_ckpt = local_out / "best_model.pt"
        local_eval = local_out / "cross_technology_evaluation.json"

        run_cmd([colab, "--auth=adc", "download", "-s", args.session, remote_ckpt, str(local_ckpt)])
        run_cmd([colab, "--auth=adc", "download", "-s", args.session, remote_eval, str(local_eval)])

        print(f"\n[SUCCESS] Downloaded best model to: {local_ckpt}")
        print(f"[SUCCESS] Downloaded cross-tech evaluation to: {local_eval}")

    finally:
        # 7. Clean up local chunks
        for old_p in local_archive.parent.glob("part_*"):
            try:
                old_p.unlink()
            except Exception:
                pass

        # 8. Stop session to avoid burning compute units
        print(f"\nReleasing Colab session '{args.session}' to preserve compute units...")
        try:
            run_cmd([colab, "--auth=adc", "stop", "-s", args.session], check=False)
            print("Session stopped successfully.")
        except Exception as e:
            print(f"Warning: could not stop session: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
