# Model Card — VEXMLM-{VARIANT}

Copy to `checkpoints/<model>/MODEL_CARD.md` and fill every field. Values marked
`{...}` come from the run's `results.json`, `expansion_manifest.json`, and
`stage1_summary.json` — take them from the artifacts, never from memory.

## Overview

| Field | Value |
|---|---|
| Model | VEXMLM-{VARIANT} |
| Base | `xlm-roberta-base` |
| Architecture | XLM-RoBERTa, 12 layers, hidden 768, 12 heads |
| Vocabulary | {FINAL_VOCAB} ({BASE_VOCAB} base + {NEW_TOKENS} Ge'ez tokens) |
| Initialization | mean-based, `e_t = (1/|V_s|) Σ e_s` (paper Sec. 3.3) |
| Languages | Amharic (amh), Tigrinya (tir) |
| Stage 1 | continued MLM, {EPOCHS} epochs, all parameters trainable |
| Stage 2 | {TASK} fine-tuning |
| Licence | Apache-2.0 |

## Training data

### Stage 1 — continued MLM

| Corpus | Language | Sentences | SHA-256 | Licence | Source |
|---|---|---|---|---|---|
| {NAME} | amh | {N} | `{HASH}` | {LICENCE} | {URL} |
| {NAME} | tir | {N} | `{HASH}` | {LICENCE} | {URL} |

Upsampling: α = {ALPHA} (1.0 = natural proportions, 0.0 = equal).

### Stage 2 — fine-tuning

| Dataset | Task | Train / Dev / Test | Licence |
|---|---|---|---|
| {NAME} | {TASK} | {N}/{N}/{N} | {LICENCE} |

## Hyperparameters

From `configs/base.yaml` (paper Table 1). Record any override.

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| max sequence length | 256 | 256 |
| batch size (per device) | 32 | 32 |
| effective batch size | {EFF_BS} | {EFF_BS} |
| epochs | 10 | 4 |
| learning rate | 5e-5 | 2e-5 |
| optimizer | AdamW | AdamW |
| weight decay | 0.01 | 0.01 |
| gradient clipping | 1.0 | 1.0 |
| MLM probability | 0.15 | — |
| precision | {fp16/bf16} | {fp16/bf16} |

## Results

Mean ± std over seeds {SEEDS}. State the seed count; never report one seed as if
it were a point estimate.

| Metric | Value | Seeds |
|---|---|---|
| {METRIC} | {MEAN} ± {STD} | {N} |

Baseline for comparison: {XLM-R baseline value}.

## Intrinsic metrics

| Metric | XLM-R | VEXMLM |
|---|---|---|
| Fertility (amh) | {V} | {V} |
| Fertility (tir) | {V} | {V} |
| Compression (amh) | {V} | {V} |
| Compression (tir) | {V} | {V} |
| Parity (amh/tir) | {V} | {V} |
| New-token firing rate | — | {V} |

Parity is meaningful only on a parallel corpus; state which was used.

## Reproducibility

| Field | Value |
|---|---|
| Git commit | `{COMMIT}` |
| Config hash | `{CONFIG_HASH}` |
| Tokenizer SHA-256 | `{TOK_HASH}` |
| Seeds | {SEEDS} |
| Hardware | {GPU} × {N} |
| Framework | torch {V}, transformers {V} |

## Limitations

- Amharic and Tigrinya only; other Ge'ez-script languages (Tigre, Ge'ez proper)
  are untested.
- Domain follows the pretraining corpora; performance on out-of-domain text
  (legal, medical, heavy code-switching) is unmeasured.
- Under mean-based initialization all new embeddings start identical and are
  separated during Stage 1. Skipping Stage 1 leaves them undifferentiated.
- {TASK}-specific limits: {...}

## Ethical considerations

- Training data may carry the biases of its sources; social-media corpora such
  as AfriSenti frequently include offensive language.
- Sentiment and NER outputs should not be treated as authoritative for content
  moderation or any consequential decision about individuals.
- Report per-language results separately. A single averaged score can hide the
  fact that the lower-resource language is being served worse — which is the
  problem this work exists to address.

## Citation

See [CITATION.cff](../CITATION.cff).
