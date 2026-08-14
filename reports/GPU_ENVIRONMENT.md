# GPU Environment

Checked 2026-08-14T21:16:35 on `gpunode05`.

✅ **GPU VERIFIED**

## Verification gate

| Requirement | Result |
|---|---|
| `torch.cuda.is_available()` | **True** |
| Visible devices | 1 |
| BF16 supported | True |
| FP16 supported | True |
| Precision selected | **bf16** |

## SLURM

| Field | Value |
|---|---|
| Job ID | 60584 |
| Node | gpunode05 |
| Partition | ampere |

## Software

| Component | Version |
|---|---|
| PyTorch | 2.5.1+cu118 |
| CUDA (torch build) | 11.8 |
| cuDNN | 90100 |
| NVIDIA driver | 610.57.04 |
| Accelerate | 1.14.0 |
| Python | 3.13.11 |

## Devices

| # | GPU | VRAM | Compute capability | SMs |
|---|---|---|---|---|
| 0 | NVIDIA A100 80GB PCIe | 79.27 GB | 8.0 | 108 |

## Batch configuration

| Field | Value |
|---|---|
| Per-device batch size (paper) | 32 |
| GPUs | 1 |
| **Expected effective batch size** | **32** |
| Distributed recommended | False |

## Functional check

- status: **OK**
- fp32 matmul 2048×2048: OK
- bf16 matmul: OK
- peak allocation: 68.1 MB
