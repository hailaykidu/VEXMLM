# Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya

**Accepted at the LM4UC Workshop, IJCAI 2026.**

Official implementation and companion repository for the VEXMLM paper.

---

## Overview

Amharic and Tigrinya are Ge'ez-script languages with tens of millions of
speakers and little NLP infrastructure. Multilingual models such as XLM-R cover
them nominally but tokenize them poorly: only ~1.2% of XLM-R's vocabulary is
Ge'ez-script, and Tigrinya text fragments into roughly 3.1 subwords per word.
Excessive fragmentation inflates sequence length, consumes context, and splits
the entity and answer spans that downstream tasks depend on.

**VEXMLM** adapts `xlm-roberta-base` to these languages by extending its subword
vocabulary with language-specific tokens, initializing the new embeddings from
the centroid of the pretrained embedding space, and training in two stages:
continued masked-language-model pretraining, then task-specific fine-tuning.

```
XLM-R (250K vocab)
      │
      ├─ Step 1  Language-specific SentencePiece    amh 32K · tir 50K
      ├─ Step 2  Vocabulary expansion               → 280K, originals preserved
      ├─ Step 3  Mean-based initialization          e_t = (1/|V_s|) Σ e_s
      ├─ Step 4  Stage 1 · continued MLM            all parameters trainable
      └─ Step 5  Stage 2 · fine-tuning              QA · NER · Sentiment
```

---

## Contributions

**Vocabulary extension.** XLM-R's 250K vocabulary is extended to 280K with
~30K Ge'ez-derived subwords. All original entries are preserved — tokens are
appended, never replaced or reordered — so pretrained embeddings stay valid.

**Embedding initialization.** New rows are initialized from the mean of the
pretrained embedding space,

$$e_t = \frac{1}{|V_s|}\sum_{s \in V_s} e_s$$

placing them inside the existing representation manifold rather than at random.
Implemented as `global_mean` in `vocabulary_expansion/initialization.py`.

**Language-specific tokenization.** Separate SentencePiece Unigram models are
trained per language (Amharic 32K, Tigrinya 50K) with NFC normalization and an
Ethiopic-ratio filter, then merged by rank-interleaving so each language
contributes its highest-value tokens. The selected tokens are merged natively
into the SentencePiece model (**SP-Merge**), so encoding and decoding are exact.

**Two-stage training.** Stage 1 continues MLM pretraining on Amharic and
Tigrinya with all parameters trainable; Stage 2 fine-tunes per task across
seeds 42–46.

**Intrinsic and downstream evaluation.** Tokenizer quality (fertility,
compression, parity, OOV accuracy) alongside five downstream tasks spanning
question answering, named entity recognition, and sentiment analysis.

---

## Repository Structure

```
configs/          base.yaml (paper Table 1), seeds.yaml, ablations/
datasets/         registry.py, manager.py, prepare.py — sources, licences, hashes
tokenizer/        SentencePiece training, token selection, evaluation
vocabulary_expansion/  expansion, initialization strategies, verification
pretraining/      Stage 1 continued MLM
finetuning/       Stage 2: qa/, ner/, sentiment/
evaluation/       intrinsic metrics, aggregation, table generation
scripts/          reproduce_paper.sh, SLURM job scripts
src/vexmlm/       shared library: geez, device, config, tracking
results/          generated tables (CSV + RESULTS.md)
tests/            pytest suite
```

**`datasets/`** declares every dataset with its source, licence, and citation.
Raw and processed data are not distributed — see [Data Preparation](#data-preparation).

**`checkpoints/`** is not tracked in git. The Stage 1 model is released
separately (see [Main Results](#main-results)).

---

## Installation

```bash
git clone <repo-url> && cd VEXMLM
pip install -e ".[all]"       # omit [all] to skip MLflow and dev tools
pytest -q                     # verify the installation
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0.

GPU strongly recommended. Device selection (CUDA → MPS → CPU) and precision
policy (BF16 on SM 8.0+, else FP16) are automatic — see `src/vexmlm/device.py`.
CPU works for debugging and tests only.

---

## Data Preparation

Corpora and task datasets are **not redistributed**; several carry restrictive or
unresolved licences. Place raw data under `datasets/raw/` and let the pipeline
prepare it:

```bash
# 1. Place raw corpora
datasets/raw/amharic/*.txt
datasets/raw/tigrinya/*.txt

# 2. Normalize, deduplicate, and split
python3 datasets/prepare.py --config configs/base.yaml
```

This writes `datasets/processed/{amharic,tigrinya}.{train,dev,oov}.txt` and
records row counts, content hashes, and label distributions to
`datasets/metadata/`.

See `docs/DATA_SETUP.md` for per-dataset instructions and
`docs/DATASET_PROVENANCE.md` for sources and licences.

> ⚠️ **Licensing.** MasakhaNER-Amharic is **CC BY-NC 4.0** — non-commercial.
> Any model fine-tuned on it inherits that restriction. Check
> `datasets/registry.py` for the licence of every dataset before use.

---

## Training

Run the full pipeline:

```bash
bash scripts/reproduce_paper.sh --dry-run     # inspect the plan
bash scripts/reproduce_paper.sh               # run everything
```

Or stage by stage (`--stage N`):

| Stage | Content |
|---|---|
| 1 | Dataset preparation |
| 2 | SentencePiece tokenizers (amh 32K / tir 50K) |
| 3 | Vocabulary expansion + mean initialization |
| 4 | Stage 1 continued MLM pretraining |
| 5 | Stage 2 fine-tuning (5 tasks × 5 seeds) |
| 6 | Intrinsic evaluation |
| 7 | Table generation |

On a cluster:

```bash
sbatch scripts/slurm_pretrain.sh                            # Stage 1, 1× A100
sbatch scripts/slurm_pretrain_multigpu.sh                   # Stage 1, 4× A100 (DDP)
sbatch scripts/slurm_finetune.sh sentiment afrisenti amh    # array job, 5 seeds
```

Training resumes from the last checkpoint automatically. Every run records git
commit, seed, config hash, dataset hashes, hardware, and metrics to
`experiment_tracking/runs/`.

---

## Evaluation

**Intrinsic** — fertility, compression, parity, OOV accuracy:

```bash
python3 evaluation/run_intrinsic.py \
  --tokenizer xlm-roberta-base checkpoints/vexmlm-stage1-spm \
  --corpus amh=datasets/processed/amharic.dev.txt \
           tir=datasets/processed/tigrinya.dev.txt \
  --oov-words amh=datasets/processed/amharic.oov.txt \
              tir=datasets/processed/tigrinya.oov.txt \
  --base-vocab-size 250002 \
  --out results/spmerge_tokenizer_metrics.json
```

Parity is reported as valid **only** on a sentence-aligned parallel corpus
(`--parallel`). On non-parallel text the ratio reflects content differences
rather than tokenizer fairness, and the output says so.

**Downstream** — task metrics aggregated over seeds:

```bash
python3 evaluation/generate_tables.py --runs checkpoints --out results
```

Metrics with no run behind them are emitted as `NOT YET MEASURED` — never
estimated, never silently omitted.

---

## Main Results

See [RESULTS_SUMMARY.md](reports/RESULTS_SUMMARY.md) for the full tables.

| Table | Content |
|---|---|
| Table 1 | Hyperparameters (`configs/base.yaml`) |
| Table 2 | Tokenizer fertility, compression, parity |
| Table 3 | OOV word accuracy |
| Table 4 | Downstream performance (QA, NER, Sentiment) |
| Table 5 | Ablation over initialization strategies |

Implementation artifacts produced by this repository live in `results/` — see
[results/README.md](results/README.md).

**Model**: `vexmlm-stage1-spm` — the Stage 1 model with the SP-Merge tokenizer
(280,002 subwords). Release location to be added.

---

## Tests

```bash
pytest -q                      # full suite
pytest tests/test_geez.py -v   # script handling and encoding checks
```

---

## Citation

```bibtex
@inproceedings{teklehaymanot2026vexmlm,
  title     = {Expanding the Lexicon of Ge'ez Based African Languages:
               A Comparative Study of Amharic and Tigrinya},
  author    = {Teklehaymanot, Hailay},
  booktitle = {Proceedings of the Workshop on Language Models for
               Underserved Communities (LM4UC), IJCAI},
  year      = {2026}
}
```

---

## License

Apache-2.0 for the code. Dataset licences are recorded per dataset in
`datasets/registry.py` and `datasets/cards/`; several are unresolved and are
tracked in `docs/RELEASE_CHECKLIST.md`.
