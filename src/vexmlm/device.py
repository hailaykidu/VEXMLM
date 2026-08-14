"""GPU-first device selection, precision policy, and hardware monitoring.

Priority: CUDA -> MPS -> CPU. Mixed precision is enabled by default on capable
hardware: BF16 on Ampere (SM 8.0) and newer, FP16 otherwise. CPU is a debug
fallback only.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, asdict

import torch

log = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device_type: str            # cuda | mps | cpu
    device_count: int
    device_names: list[str]
    total_vram_gb: list[float]
    cuda_version: str | None
    torch_version: str
    bf16_supported: bool
    fp16_supported: bool
    capability: list[str]
    distributed: bool
    world_size: int
    local_rank: int

    def as_dict(self) -> dict:
        return asdict(self)


def _cuda_capability() -> list[str]:
    return [f"{torch.cuda.get_device_capability(i)[0]}."
            f"{torch.cuda.get_device_capability(i)[1]}"
            for i in range(torch.cuda.device_count())]


def detect() -> DeviceInfo:
    """Detect the best available accelerator and its precision capabilities."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        vram = [round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                for i in range(n)]
        caps = _cuda_capability()
        # BF16 needs SM 8.0+ (Ampere). A100/H100 qualify; V100/T4 do not.
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        return DeviceInfo("cuda", n, names, vram, torch.version.cuda,
                          torch.__version__, bf16, True, caps,
                          world_size > 1, world_size, local_rank)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", 1, ["Apple MPS"], [0.0], None, torch.__version__,
                          False, False, [], False, 1, 0)

    log.warning("No GPU detected -- falling back to CPU. This is supported for "
                "debugging and tests only; training will be impractically slow.")
    return DeviceInfo("cpu", 0, [], [], None, torch.__version__,
                      False, False, [], False, 1, 0)


def get_device(info: DeviceInfo | None = None) -> torch.device:
    info = info or detect()
    if info.device_type == "cuda":
        return torch.device(f"cuda:{info.local_rank}")
    return torch.device(info.device_type)


def precision_flags(info: DeviceInfo | None = None,
                    prefer: str = "auto") -> dict[str, bool]:
    """Return {'fp16':bool,'bf16':bool} for transformers.TrainingArguments.

    `prefer` may be auto | bf16 | fp16 | none. BF16 is chosen when supported --
    its wider exponent range avoids the loss-scaling instability FP16 can hit
    during MLM pretraining.
    """
    info = info or detect()
    if prefer == "none" or info.device_type != "cuda":
        return {"fp16": False, "bf16": False}
    if prefer == "bf16":
        if not info.bf16_supported:
            log.warning("bf16 requested but unsupported; using fp16")
            return {"fp16": True, "bf16": False}
        return {"fp16": False, "bf16": True}
    if prefer == "fp16":
        return {"fp16": True, "bf16": False}
    return ({"fp16": False, "bf16": True} if info.bf16_supported
            else {"fp16": True, "bf16": False})


def gpu_utilization() -> list[dict]:
    """Live GPU utilization via nvidia-smi. Empty list if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            rows.append({"index": int(parts[0]), "name": parts[1],
                         "util_pct": float(parts[2]), "mem_used_mb": float(parts[3]),
                         "mem_total_mb": float(parts[4]), "temp_c": float(parts[5])})
    return rows


def effective_batch_size(per_device: int, grad_accum: int,
                         info: DeviceInfo | None = None) -> int:
    info = info or detect()
    return per_device * grad_accum * max(info.device_count, 1)


def log_environment(info: DeviceInfo | None = None) -> dict:
    """Log and return the hardware record attached to every experiment."""
    info = info or detect()
    record = info.as_dict()
    record["utilization"] = gpu_utilization()
    log.info("device=%s count=%d names=%s cuda=%s bf16=%s",
             info.device_type, info.device_count, info.device_names,
             info.cuda_version, info.bf16_supported)
    return record
