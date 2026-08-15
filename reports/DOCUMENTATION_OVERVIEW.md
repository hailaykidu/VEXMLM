# Documentation Overview

**Date**: 2026-08-15
**Purpose**: component-by-component reference for the VEXMLM pipeline.

Each component lists its purpose, entry script, expected inputs, and expected
outputs. The full pipeline runs via `scripts/reproduce_paper.sh`; individual
stages run with `--stage N`.

---

## Pipeline at a Glance

| Stage | Component | Entry point |
|---|---|---|
| 1 | Data preparation | `datasets/prepare.py` |
| 2 | Tokenizer construction | `tokenizer/train_{amharic,tigrinya}_tokenizer.py` |
| 3 | Vocabulary expansion | `tokenizer/extract_new_tokens.py` → `vocabulary_expansion/merge_vocabularies.py` → `expand_xlmr_vocab.py` |
| 3 | Embedding initialization | `vocabulary_expansion/initialization.py` (invoked by the above) |
| 4 | MLM training | `pretraining/run_mlm.py` |
| 5 | Fine-tuning | `finetuning/{qa,ner,sentiment}/run_*.py` |
| 6–7 | Evaluation | `evaluation/run_intrinsic.py`, `evaluation/generate_tables.py` |

---

## 1. Data Preparation

**Purpose** — normalize, deduplicate, filter, and split raw monolingual corpora;
record provenance.

**Entry script** — `datasets/prepare.py`

```bash
python3 datasets/prepare.py --config configs/base.yaml
```

**Inputs**
- `datasets/raw/amharic/*.txt`, `datasets/raw/tigrinya/*.txt`
- `configs/base.yaml` → `tokenizer:` block (NFC normalization,
  `min_sentence_length: 10`, `min_ethiopic_ratio: 0.5`, `deduplicate: true`)

**Outputs**
- `datasets/processed/{amharic,tigrinya}.{train,dev,oov}.txt`
- `datasets/processed/preparation_report.json` — line counts, characters, hashes

**Notes** — fails with a clear message if corpora are absent rather than
proceeding. Dataset sources, licences, and citations are declared in
`datasets/registry.py`; loading goes through `datasets/manager.py`, which records
row counts and content hashes to `datasets/metadata/`.

---

## 2. Tokenizer Construction

**Purpose** — train per-language SentencePiece Unigram models over the prepared
corpora.

**Entry scripts**
- `tokenizer/train_amharic_tokenizer.py`
- `tokenizer/train_tigrinya_tokenizer.py`
- `tokenizer/spm_trainer.py` (shared implementation)

```bash
python3 tokenizer/train_amharic_tokenizer.py \
  --input datasets/raw/amharic/*.txt --output-dir tokenizer/artifacts/amharic
```

**Inputs** — raw corpora; config: `amharic_vocab_size: 32000`,
`tigrinya_vocab_size: 50000`, `model_type: unigram`,
`character_coverage: 0.9995`.

**Outputs**
- `tokenizer/artifacts/{amharic,tigrinya}/{lang}_sp.{model,vocab}`
- `corpus.normalized.txt`
- `manifest.json` — spec, corpus SHA-256, piece counts, Ethiopic fraction

**Supporting** — `tokenizer/evaluate_tokenizer.py` measures fertility and
compression for any tokenizer/corpus pair.

---

## 3. Vocabulary Expansion

**Purpose** — select language-specific tokens absent from XLM-R and append them,
preserving every original entry.

**Entry scripts** (three steps)

```bash
# a. extract candidates per language
python3 tokenizer/extract_new_tokens.py \
  --spm tokenizer/artifacts/tigrinya/tigrinya_sp.model \
  --corpus datasets/processed/tigrinya.train.txt \
  --max-new-tokens 15000 \
  --out vocabulary_expansion/artifacts/new_tokens_tir.json

# b. merge to the 30K target
python3 vocabulary_expansion/merge_vocabularies.py \
  --inputs vocabulary_expansion/artifacts/new_tokens_amh.json \
           vocabulary_expansion/artifacts/new_tokens_tir.json \
  --target-total 30000 \
  --out vocabulary_expansion/artifacts/new_tokens.json

# c. expand the model
python3 vocabulary_expansion/expand_xlmr_vocab.py \
  --new-tokens vocabulary_expansion/artifacts/new_tokens.json \
  --init global_mean --seed 42 \
  --validation-corpus datasets/processed/tigrinya.dev.txt \
  --output checkpoints/vexmlm-expanded
```

**Inputs** — per-language SP models, processed corpora, `xlm-roberta-base`.

**Outputs**
- `vocabulary_expansion/artifacts/new_tokens{,_amh,_tir}.json`
- `checkpoints/vexmlm-expanded/` — model, tokenizer, `expansion_manifest.json`
- `results/init_verification.json` (via `verify_vocab.py`)

**Safeguards**
- An **encoding gate** rejects byte-level-BPE tokens, which can never match input
  in a SentencePiece tokenizer.
- A **firing check** confirms the new tokens are reachable on real text.
- Merging deduplicates across languages and rank-interleaves so each contributes
  its highest-value tokens.

---

## 4. Embedding Initialization

**Purpose** — place new embedding rows inside the pretrained representation
space.

**Entry point** — `vocabulary_expansion/initialization.py`, selected by
`expand_xlmr_vocab.py --init`.

**Strategies**

| Strategy | Role |
|---|---|
| `global_mean` | **Official method** — e_t = (1/\|V_s\|) Σ e_s |
| `random` | Ablation arm |
| `constituent_mean` | Research extension — centroid of a token's own subwords |
| `mixed` | Reference only |

**Inputs** — embedding matrix, count of original rows, new-token list, seed.

**Outputs** — modified embedding matrix; `InitStats` (strategy, norms, per-dim
std, distance to global mean, fallback count) recorded in the expansion manifest.

**Verification** — `vocabulary_expansion/verify_vocab.py` identifies which
strategy produced a checkpoint from its weights alone, comparing observed
E‖e_new − ē‖ against each method's closed form.

---

## 5. MLM Training (Stage 1)

**Purpose** — continued masked-language-model pretraining so the expanded model
adapts to Amharic and Tigrinya. All parameters trainable.

**Entry script** — `pretraining/run_mlm.py`

```bash
python3 pretraining/run_mlm.py \
  --config configs/base.yaml \
  --model checkpoints/vexmlm-expanded \
  --amharic datasets/processed/amharic.train.txt \
  --tigrinya datasets/processed/tigrinya.train.txt \
  --block-chunk --mode official \
  --output checkpoints/vexmlm-stage1
```

Cluster: `sbatch scripts/slurm_pretrain.sh` (1× A100) or
`scripts/slurm_pretrain_multigpu.sh` (4× A100, DDP).

**Inputs** — expanded model, processed corpora, `pretraining:` config block
(`max_seq_length: 256`, batch 32, LR 5e-5, `mlm_probability: 0.15`,
`warmup_ratio: 0.06`).

**Outputs** — `checkpoints/vexmlm-stage1/` with `stage1_summary.json`
(eval loss, perplexity, runtime, FLOPs, hardware).

**Notes** — `--block-chunk` concatenates the corpus and slices it into
full-length blocks, avoiding the ~90% padding waste of one-line-per-example.
Training resumes from the last checkpoint automatically.

---

## 6. Fine-Tuning (Stage 2)

**Purpose** — task-specific adaptation and evaluation across seeds 42–46.

**Entry scripts**

| Task | Script | Datasets | Metrics |
|---|---|---|---|
| QA | `finetuning/qa/run_qa.py` | tigqa, amqa | Exact Match, F1 |
| NER | `finetuning/ner/run_ner.py` | masakhaner_amh, tigrinya_ner | entity F1, macro F1, accuracy |
| Sentiment | `finetuning/sentiment/run_sentiment.py` | afrisenti (`--dataset-config amh`) | accuracy, macro/weighted F1 |

```bash
python3 finetuning/qa/run_qa.py \
  --config configs/base.yaml \
  --model checkpoints/vexmlm-stage1 \
  --dataset tigqa --local-path datasets/raw/tigqa \
  --seed 42 --mode official \
  --output checkpoints/qa-tigqa-42
```

Cluster: `sbatch scripts/slurm_finetune.sh <task> <dataset> [config]`,
or `scripts/slurm_stage2_all.sh` for everything.

**Inputs** — Stage 1 checkpoint, task datasets, `finetuning:` config block
(4 epochs, LR 2e-5, batch 32, `warmup_ratio: 0.1`).

**Outputs** — `checkpoints/<task>-<dataset>-<seed>/results.json`, plus
`predictions.json` for QA; run records in `experiment_tracking/runs/`.

---

## 7. Evaluation

### 7a. Intrinsic

**Purpose** — tokenizer quality independent of any downstream task.

**Entry script** — `evaluation/run_intrinsic.py`
(metrics in `evaluation/metrics/intrinsic.py`)

**Inputs** — one or more tokenizers, `lang=path` corpora, optional `lang=path`
OOV word lists, optional `--base-vocab-size` for new-token share.

**Outputs** — `results/tokenizer_metrics.json`: per language, fertility,
compression, continuation rate, UNK rate, new-token share; per OOV list, OOV
accuracy and no-UNK rate; parity when exactly two languages are supplied.

**Note** — parity is marked `valid: false` unless `--parallel` is passed with
sentence-aligned corpora.

### 7b. Aggregation and Tables

**Purpose** — collect per-seed runs into the paper's tables.

**Entry scripts** — `evaluation/aggregate.py`, `evaluation/generate_tables.py`

```bash
python3 evaluation/generate_tables.py \
  --runs checkpoints --intrinsic results/tokenizer_metrics.json --out results
```

**Outputs** — `results/table{2,3,4,5}_*.csv` and `results/RESULTS.md`.

**Contract** — only measured values are written. Any metric without a run behind
it is emitted as `NOT YET MEASURED` rather than estimated or silently omitted.

---

## Shared Library — `src/vexmlm/`

| Module | Purpose |
|---|---|
| `geez.py` | Ge'ez-script detection, encoding checks |
| `device.py` | device selection (CUDA → MPS → CPU), precision policy, environment logging |
| `config.py` | YAML loading, dotted-path access, config hashing |
| `modes.py` | debug / research / official run profiles |
| `tracking.py` | run records — git state, seeds, hashes, hardware, metrics |

---

## Configuration

| File | Purpose |
|---|---|
| `configs/base.yaml` | all hyperparameters (paper Table 1) |
| `configs/modes.yaml` | run profiles; `official` is the only reportable one |
| `configs/seeds.yaml` | seed policy; three-seed minimum for reporting |
| `configs/ablations/` | ablation variants |

Override on the command line with `--override key=value` so deviations are
recorded, rather than editing `base.yaml`.
