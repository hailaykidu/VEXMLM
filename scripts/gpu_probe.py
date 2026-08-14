#!/usr/bin/env python3
"""Verify the GPU environment on a compute node and emit reports/GPU_ENVIRONMENT.md.

Mandatory gate before Stage 1. Exits non-zero if CUDA is unavailable or if
neither BF16 nor FP16 is supported -- training must not start in that state,
and must never silently fall back to CPU.

Run on a GPU node:
    srun -p ampere --gres=gpu:a100_80gb:1 python scripts/gpu_probe.py
    # or via sbatch scripts/slurm_gpu_probe.sh
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True).stdout
        return [l.strip() for l in out.strip().splitlines() if l.strip()]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/GPU_ENVIRONMENT.md")
    ap.add_argument("--json-out", default="results/gpu_environment.json")
    ap.add_argument("--allow-fp32", action="store_true",
                    help="do not fail when neither BF16 nor FP16 is available")
    args = ap.parse_args()

    import torch

    available = torch.cuda.is_available()
    n = torch.cuda.device_count() if available else 0

    env = {
        "checked": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_available": available,
        "device_count": n,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if available else None,
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "driver_version": (nvidia_smi("driver_version") or ["unavailable"])[0],
        "devices": [],
    }

    try:
        import accelerate
        env["accelerate_version"] = accelerate.__version__
    except ImportError:
        env["accelerate_version"] = None

    bf16 = fp16 = False
    if available:
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        fp16 = True
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            env["devices"].append({
                "index": i, "name": p.name,
                "vram_gb": round(p.total_memory / 1024**3, 2),
                "compute_capability": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
            })

    env["bf16_supported"] = bf16
    env["fp16_supported"] = fp16
    env["precision_selected"] = "bf16" if bf16 else ("fp16" if fp16 else "NONE")

    # Effective batch size at the paper's per-device batch of 32.
    per_device = 32
    env["expected_effective_batch_size"] = per_device * max(n, 1)
    env["per_device_batch_size"] = per_device
    env["distributed_recommended"] = n > 1

    # Live functional check: a real allocation and matmul, not just a flag.
    if available:
        try:
            dev = torch.device("cuda:0")
            a = torch.randn(2048, 2048, device=dev)
            b = torch.randn(2048, 2048, device=dev)
            torch.cuda.synchronize()
            c = (a @ b).sum().item()
            env["smoke_test"] = {"status": "OK", "matmul_2048_sum_finite": bool(c == c)}
            if bf16:
                ab = a.to(torch.bfloat16) @ b.to(torch.bfloat16)
                env["smoke_test"]["bf16_matmul"] = "OK" if ab.isfinite().all().item() else "NON-FINITE"
            env["smoke_test"]["allocated_mb"] = round(
                torch.cuda.max_memory_allocated(dev) / 1024**2, 1)
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:                                  # noqa: BLE001
            env["smoke_test"] = {"status": "FAILED", "error": str(exc)}

    jout = REPO / args.json_out
    jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(env, indent=2))

    # --- report ---------------------------------------------------------
    ok = available and (bf16 or fp16 or args.allow_fp32)
    verdict = "✅ **GPU VERIFIED**" if ok else "❌ **BLOCKED — training must not start**"

    L = ["# GPU Environment", "",
         f"Checked {env['checked']} on `{env['hostname']}`.", "",
         verdict, "", "## Verification gate", "",
         "| Requirement | Result |", "|---|---|",
         f"| `torch.cuda.is_available()` | **{available}** |",
         f"| Visible devices | {n} |",
         f"| BF16 supported | {bf16} |",
         f"| FP16 supported | {fp16} |",
         f"| Precision selected | **{env['precision_selected']}** |", ""]

    if env["slurm_job_id"]:
        L += ["## SLURM", "", "| Field | Value |", "|---|---|",
              f"| Job ID | {env['slurm_job_id']} |",
              f"| Node | {env['slurm_node']} |",
              f"| Partition | {env['slurm_partition']} |", ""]

    L += ["## Software", "", "| Component | Version |", "|---|---|",
          f"| PyTorch | {env['torch_version']} |",
          f"| CUDA (torch build) | {env['torch_cuda_build']} |",
          f"| cuDNN | {env['cudnn_version']} |",
          f"| NVIDIA driver | {env['driver_version']} |",
          f"| Accelerate | {env['accelerate_version']} |",
          f"| Python | {env['python']} |", ""]

    if env["devices"]:
        L += ["## Devices", "",
              "| # | GPU | VRAM | Compute capability | SMs |", "|---|---|---|---|---|"]
        for d in env["devices"]:
            L.append(f"| {d['index']} | {d['name']} | {d['vram_gb']} GB | "
                     f"{d['compute_capability']} | {d['multi_processor_count']} |")
        L.append("")

    L += ["## Batch configuration", "", "| Field | Value |", "|---|---|",
          f"| Per-device batch size (paper) | {per_device} |",
          f"| GPUs | {max(n, 1)} |",
          f"| **Expected effective batch size** | **{env['expected_effective_batch_size']}** |",
          f"| Distributed recommended | {env['distributed_recommended']} |", ""]
    if n > 1:
        L += [f"> With {n} GPUs, DDP multiplies the effective batch to "
              f"{per_device * n}. To hold the paper's effective 32, set "
              f"`pretraining.per_device_train_batch_size={max(32 // n, 1)}`.", ""]

    if "smoke_test" in env:
        st = env["smoke_test"]
        L += ["## Functional check", "",
              f"- status: **{st.get('status')}**"]
        if st.get("status") == "OK":
            L += [f"- fp32 matmul 2048×2048: OK",
                  f"- bf16 matmul: {st.get('bf16_matmul', 'n/a')}",
                  f"- peak allocation: {st.get('allocated_mb')} MB"]
        else:
            L.append(f"- error: `{st.get('error')}`")
        L.append("")

    if not ok:
        L += ["## Why training is blocked", "",
              "`torch.cuda.is_available()` is False, so no official training run may",
              "start. CPU fallback is not permitted for training. Submit this probe",
              "as a SLURM job on a GPU partition:", "",
              "```bash", "sbatch scripts/slurm_gpu_probe.sh", "```", ""]

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))

    print("\n".join(L[:24]))
    print(f"\nwrote {out} and {jout}")

    if not ok:
        print("\nBLOCKED: CUDA unavailable or no mixed-precision support.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
