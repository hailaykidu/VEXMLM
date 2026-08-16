# Official Training Plan

Execution plan for the official VEXMLM runs. **No training has started.** This
document is the plan to be approved before any job is submitted.

Hyperparameters are the paper's (Table 1) and are not varied here. Estimates use
this repository's measured artifacts, not rules of thumb; the arithmetic is in
`scripts/analyze_mlm_corpora.py` output and the derivation below.

## Model size

Expanded model at 280K vocabulary, `xlm-roberta-base` geometry
(12 layers, hidden 768, 12 heads):

| Component | Parameters |
|---|---|
| Token embeddings (280,000 × 768) | 215.0 M |
| Encoder (12 layers) | 84.9 M |
| **Total** | **≈ 300 M** |

Embeddings are **72% of all parameters** after expansion — up from ~63% at the
base 250K vocabulary. Two consequences: the embedding matrix dominates optimizer
memory, and Stage 1 gradient signal is spread thinly across 30K new rows that
all begin identical under mean-based initialization.

| Memory item | Size (fp32) |
|---|---|
| Weights | 1.20 GB |
| AdamW state (m + v) | 2.40 GB |
| Checkpoint with optimizer | ≈ 3.6 GB |
| Checkpoint, weights only | ≈ 1.2 GB |

---

## Stage 1 — Continued MLM pretraining

### Data

Post-deduplication (dedup is mandatory — raw duplicate rates are 35% and 29%,
see `results/mlm_corpus_analysis.json`):

| Corpus | Unique lines | Approx. tokens |
|---|---|---|
| Amharic | 129,864 | 1.24 M |
| Tigrinya | 141,015 | 1.08 M |
| **Combined** | **270,879** | **≈ 2.3 M** |

### Step budget — decision required

The paper specifies 10 epochs at batch 32, length 256. How many steps that means
depends on how examples are formed, and the two options differ by 13×:

| Example construction | Sequences | Steps/epoch | 10 epochs |
|---|---|---|---|
| One line = one example (current default) | 270,879 | 8,465 | **84,650** |
| Block-chunked to 256 tokens | ≈ 20,800 | 650 | **6,500** |

Lines are short — median 40 characters (Amharic) and 30 (Tigrinya), roughly 6–9
whitespace tokens. Under the current default, a 256-token sequence is
overwhelmingly padding: **roughly 90% of every batch is wasted compute**.
Block-chunking concatenates lines into full 256-token blocks, which is standard
MLM practice.

**Recommendation:** run Stage 1 with block-chunking. It is ~13× cheaper for
strictly more real training signal.

**This is not yet implemented.** It needs an explicit decision:

- **Option A** — run as-is (one line per example). No code change; ~84,650
  steps; ~90% padding.
- **Option B** — add block-chunking to `pretraining/run_mlm.py`, a scoped
  ~30-line change to `build_corpus()`, then run. Requires lifting the freeze for
  that function.

A second concern applies either way: **2.3 M tokens is small for continued
pretraining.** Ten epochs over a corpus this size risks memorization, and
held-out perplexity will be optimistic because dedup cannot remove near-duplicates.
Consider reporting perplexity on a genuinely held-out corpus, and treating epoch
count as an ablation rather than a fixed constant.

### Hyperparameters (paper Table 1)

```
max_seq_length            256
per_device_train_batch    32
epochs                    10
learning_rate             5e-5
optimizer                 AdamW
weight_decay              0.01
mlm_probability           0.15
max_grad_norm             1.0
warmup_ratio              0.06
lr_scheduler              linear
precision                 bf16 (A100) / fp16 (older)
trainable                 all parameters
```

### GPU requirements

| Configuration | VRAM (est.) | Wall clock (est.) |
|---|---|---|
| 1× A100-80GB, bs 32, len 256, bf16 | 18–24 GB | Option A ≈ 6–10 h · Option B ≈ 0.5–1 h |
| 4× A100 DDP, per-device bs 8 | 10–14 GB/GPU | Option A ≈ 2–3 h |

Estimates assume ~4–6 steps/s at this model size and sequence length on A100
bf16. They are unverified — **this machine has no GPU**, so the CUDA path has
never executed. Treat the first job as a calibration run.

With 80 GB available there is generous headroom; gradient checkpointing is
unnecessary and stays off.

**Multi-GPU caveat.** DDP multiplies effective batch size by device count.
`slurm_pretrain_multigpu.sh` lowers per-device batch to 8 so 4 GPUs reproduce
the paper's effective 32. Every run logs its effective batch size.

### Checkpoint schedule

`save_strategy: epoch`, `save_total_limit: 3`, `load_best_model_at_end: true`
on `eval_loss`.

| Item | Size | Count | Total |
|---|---|---|---|
| Checkpoint (weights + optimizer) | 3.6 GB | 3 retained | 10.8 GB |
| Final model | 1.2 GB | 1 | 1.2 GB |
| **Stage 1 storage** | | | **≈ 12 GB** |

Resume is automatic from the last checkpoint; `slurm_pretrain.sh` requeues on
preemption.

### Evaluation schedule

Per epoch on a 1% held-out split: `eval_loss` and perplexity. On completion, run
`vocabulary_expansion/verify_vocab.py` against the **pre-Stage-1** checkpoint to
confirm the initialization, and `evaluation/run_intrinsic.py` for Tables 2–3.

### Outputs

```
checkpoints/vexmlm-stage1/
├── model.safetensors, config.json, tokenizer files
├── stage1_summary.json          metrics + hardware record
└── checkpoint-*/                3 retained
experiment_tracking/runs/stage1-mlm-seed42_*.json
```

---

## Stage 2 — Task-specific fine-tuning

### Hyperparameters (paper Table 1)

```
max_seq_length   256      epochs        4
batch_size       32       learning_rate 2e-5
optimizer        AdamW     weight_decay  0.01
max_grad_norm    1.0      scheduler     linear + warmup
precision        bf16/fp16
```

### Per-task budget

Steps = ⌈train_examples / 32⌉ × 4 epochs.

| Task | Dataset | Train | Steps | VRAM | Time/seed |
|---|---|---|---|---|---|
| QA | TIGQA | 644 q | ≈ 84 | 12–16 GB | ~5 min |
| QA | AmQA | 1,723 q | ≈ 216 | 12–16 GB | ~10 min |
| NER | MasakhaNER (amh) | 1,750 s | ≈ 220 | 10–14 GB | ~8 min |
| NER | Tigrinya NER | 4,562 s | ≈ 572 | 10–14 GB | ~15 min |
| Sentiment | AfriSenti (amh) | per config | varies | 10–14 GB | ~20 min |

QA VRAM is higher: the sliding window (`doc_stride` 128) produces more features
than examples on long contexts.

These datasets are small. Seed variance will be substantial — especially TIGQA
at 644 training questions — which is exactly why the five-seed protocol is
mandatory and single-seed numbers are not reportable.

### Multi-seed protocol

Seeds 42–46 for every task × dataset. `slurm_finetune.sh` is an array job, one
seed per task, so five seeds run concurrently given five free GPUs.

| Run set | Jobs | Sequential GPU-hours (est.) |
|---|---|---|
| 5 datasets × 5 seeds | 25 | ≈ 5 h |
| Ablation: 4 arms × 5 datasets × 5 seeds | 100 | ≈ 20 h + 4 × Stage 1 |

Ablation arms 1–3 skip Stage 1 (arm 1 uses stock XLM-R; arms 2–3 use the
expanded model without continued pretraining), so only arm 4 needs the full
Stage 1 cost.

### Evaluation frequency

Per epoch on validation, best checkpoint by the task's primary metric
(QA: F1 · NER: macro-F1 · Sentiment: accuracy). Test is scored once, at the end,
from the best checkpoint.

### Storage

| Item | Size | Count | Total |
|---|---|---|---|
| Fine-tuned model | 1.2 GB | 25 runs | 30 GB |
| Retained checkpoints (limit 2) | 3.6 GB | 25 × 2 | 180 GB |
| **Stage 2, main runs** | | | **≈ 210 GB** |
| Full ablation (4 arms) | | | **≈ 840 GB** |

**Recommendation:** set `save_total_limit: 1` for multi-seed runs, or delete
optimizer state after each run. Only the final weights and `results.json` are
needed downstream; that cuts Stage 2 to ~30 GB and the ablation to ~120 GB.

### Outputs per run

```
checkpoints/<task>-<dataset>-<seed>/
├── model.safetensors + tokenizer
├── results.json          metrics, seed, config hash
└── predictions.json      QA only
```

`evaluation/generate_tables.py` aggregates all `results.json` into Tables 2–5.

---

## Total budget

| Phase | GPU-hours (est.) | Storage |
|---|---|---|
| Stage 1 (Option B, recommended) | 1–3 | 12 GB |
| Stage 1 (Option A) | 6–10 | 12 GB |
| Stage 2, 5 datasets × 5 seeds | ≈ 5 | 30–210 GB |
| Full ablation (optional) | ≈ 25 | 120–840 GB |
| **Main results** | **≈ 8–15** | **≈ 45 GB** |

Modest — the datasets are small. Cost is dominated by the number of runs, not by
any single run.

## Execution order

1. `python datasets/prepare.py --config configs/base.yaml` — dedup and split.
2. Train tokenizers (32K amh, 50K tir).
3. Extract, merge to 30K, expand vocabulary with `--init global_mean`.
4. `verify_vocab.py` → expect verdict `global_mean`, distance 0.0.
5. **Calibration run:** Stage 1 with `--max-steps 100` on one GPU, to validate
   the CUDA/bf16 path that has never executed here.
6. Stage 1 full.
7. Stage 2, 5 seeds per task (`--mode official`).
8. `run_intrinsic.py`, then `generate_tables.py`.
9. Write model cards from the produced artifacts.

Steps 1–4 need no GPU and can run now. `scripts/reproduce_paper.sh` orchestrates
the whole sequence; `--dry-run` prints the plan.

## Decisions needed before launch

| # | Decision | Default if unanswered |
|---|---|---|
| 1 | Stage 1 example construction: Option A or B | A (no code change, 13× cost) |
| 2 | Keep 10 epochs on a 2.3 M-token corpus? | Yes, per the paper |
| 3 | `save_total_limit` for multi-seed runs | 2 (210 GB) — 1 is advised |
| 4 | Run the full 4-arm ablation, or Table 4 only? | Table 4 only |
| 5 | MLM corpus provenance (blocks publication, not training) | unresolved |

## Preconditions

- [x] Implementation frozen and tested (25 tests passing)
- [x] Datasets integrated, checksummed, licence-reviewed
- [x] Data quality assessed; deduplication path in place
- [x] SLURM scripts written and parse-checked
- [ ] GPU node allocated
- [ ] Calibration run completed (step 5)
- [ ] Decisions 1–4 above
