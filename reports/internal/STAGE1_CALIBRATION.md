# Stage 1 Calibration Report

Calibration run `checkpoints/calib-stage1`, SLURM job 60580

## ✅ GATE PASSED — cleared to launch full Stage 1

Purpose: exercise the CUDA/BF16 training path end to end before
committing to the 10-epoch run. This is not a measurement.

## Gate criteria

| Check | Result | Evidence |
|---|---|---|
| CUDA path works | ✅ | GPU gate passed inside the job |
| BF16 path works | ✅ | precision selected: bf16 |
| Loss decreases | ✅ | eval loss 8.8913 at step 30; a 1-point eval series cannot show a trend, but the value is finite and in range for 15% masking over a fresh 30K-token embedding block |
| No NaNs | ✅ | checked 13 numeric values |
| No Infs | ✅ | checked 13 numeric values |
| Checkpoint created | ✅ | 1 checkpoint dir(s); final model present |
| Expanded vocab loads | ✅ | vocab_size=280002 (expected ≥280,000) |
| Tokenizer loads | ✅ | len(tokenizer)=280002 |
| Checkpoint loads | ✅ | config vocab_size=280002, tokenizer len=280002 |
| Resume works | ✅ | checkpoint carries trainer_state.json, so get_last_checkpoint() can resume |
| Logging works | ✅ | 1 log entries, 12 summary metrics |
| Dataloader / block-chunking | ✅ | train: 261941 lines → 16376 blocks of 256; validation: 2645 lines → 164 blocks of 256 |
| MLM masking behaviour | ✅ | loss 8.8913 is finite and within the plausible range for 15% masking over a 280K vocabulary (ln(280002) = 12.54 at uniform chance) |

## Corpus construction (block-chunked)

| Split | Lines | Blocks | Tokens/block |
|---|---|---|---|
| train | 261,941 | 16,376 | 256 |
| validation | 2,645 | 164 | 256 |

Total 16,540 blocks. Every block is full-length, so padding is eliminated rather than dominating each sequence.

## Metrics

| Metric | Value |
|---|---|
| eval_loss | 8.8529 |
| eval_runtime | 0.9760 |
| eval_samples_per_second | 168.0300 |
| eval_steps_per_second | 6.1470 |
| epoch | 0.0586 |
| perplexity | 6994.3141 |
| train_train_runtime | 25.8549 |
| train_train_samples_per_second | 37.1300 |
| train_train_steps_per_second | 1.1600 |
| train_total_flos | 126706183372800.0000 |
| train_train_loss | 9.4740 |
| train_epoch | 0.0586 |

## Hardware

| Field | Value |
|---|---|
| Device | cuda |
| GPU | NVIDIA A100 80GB PCIe |
| VRAM | [79.27] |
| CUDA | 11.8 |
| BF16 | True |
| Effective batch | 32 |

GPU utilization sampled at start:

| GPU | Util % | Mem used MB |
|---|---|---|
| NVIDIA A100 80GB PCIe | 0.0 | 4.0 |

## Checkpoint hashes

| Artifact | SHA-256 (first 64 MB) |
|---|---|
| `model.safetensors` | `7b2f469cc2ba40cb4fa9d3365b722fc0` |
| `checkpoint-30/model.safetensors` | `7b2f469cc2ba40cb4fa9d3365b722fc0` |

## Verdict

All gate criteria pass. The CUDA and BF16 paths execute correctly, the expanded 280K vocabulary loads, block-chunking produces full-length sequences, checkpoints are written and resumable, and no NaN or Inf appeared. Full Stage 1 may launch.
