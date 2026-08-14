# VEXMLM — Vocabulary-Extended XLM-R for Amharic and Tigrinya

Official implementation of *Vocabulary Expansion for Low-Resource African
Languages: A Case Study in Amharic and Tigrinya*.

VEXMLM adapts `xlm-roberta-base` to the Ge'ez-script languages Amharic (amh) and
Tigrinya (tir) by extending its subword vocabulary with language-specific tokens,
initializing the new embeddings from the centroid of the pretrained embedding
space, and then training in two stages: continued MLM pretraining followed by
task-specific fine-tuning.

```
XLM-R (250K vocab)
      │
      ├─ Step 1  language-specific SentencePiece   amh 32K · tir 50K
      ├─ Step 2  vocabulary expansion              → 280K, originals preserved
      ├─ Step 3  mean-based initialization         e_t = (1/|V_s|) Σ e_s
      ├─ Step 4  Stage 1 · continued MLM           all parameters trainable
      └─ Step 5  Stage 2 · fine-tuning             QA · NER · Sentiment
```

## Install

```bash
git clone <repo-url> && cd VEXMLM
pip install -e ".[all]"       # omit [all] to skip MLflow and dev tools
pytest -q                     # verify the installation
```

Python ≥3.9, PyTorch ≥2.0. GPU strongly recommended — see [Hardware](#hardware).

## Quick start

```bash
# 0. place raw corpora under datasets/raw/{amharic,tigrinya}/  (docs/DATA_SETUP.md)
bash scripts/reproduce_paper.sh --dry-run     # inspect the plan
bash scripts/reproduce_paper.sh               # run everything
```

Or step by step:

```bash
# 1. normalize, deduplicate, split
python datasets/prepare.py --config configs/base.yaml

# 2. train language-specific tokenizers
python tokenizer/train_amharic_tokenizer.py --input datasets/raw/amharic/*.txt \
    --output-dir tokenizer/artifacts/amharic          # 32,000
python tokenizer/train_tigrinya_tokenizer.py --input datasets/raw/tigrinya/*.txt \
    --output-dir tokenizer/artifacts/tigrinya         # 50,000

# 3. select new tokens and expand the vocabulary
python tokenizer/extract_new_tokens.py \
    --spm tokenizer/artifacts/tigrinya/tigrinya_sp.model \
    --corpus datasets/processed/tigrinya.train.txt \
    --max-new-tokens 15000 --out vocabulary_expansion/artifacts/new_tokens_tir.json
python vocabulary_expansion/merge_vocabularies.py \
    --inputs vocabulary_expansion/artifacts/new_tokens_{amh,tir}.json \
    --target-total 30000 --out vocabulary_expansion/artifacts/new_tokens.json
python vocabulary_expansion/expand_xlmr_vocab.py \
    --new-tokens vocabulary_expansion/artifacts/new_tokens.json \
    --init global_mean --validation-corpus datasets/processed/tigrinya.dev.txt \
    --output checkpoints/vexmlm-expanded

# 4. Stage 1 — continued MLM pretraining
python pretraining/run_mlm.py --config configs/base.yaml \
    --model checkpoints/vexmlm-expanded \
    --amharic datasets/raw/amharic/*.txt --tigrinya datasets/raw/tigrinya/*.txt \
    --output checkpoints/vexmlm-stage1

# 5. Stage 2 — fine-tuning
python finetuning/qa/run_qa.py        --model checkpoints/vexmlm-stage1 --dataset tigqa  --seed 42 --output checkpoints/qa-42
python finetuning/ner/run_ner.py      --model checkpoints/vexmlm-stage1 --dataset masakhaner_amh --seed 42 --output checkpoints/ner-42
python finetuning/sentiment/run_sentiment.py --model checkpoints/vexmlm-stage1 --dataset afrisenti --dataset-config amh --seed 42 --output checkpoints/sa-42

# 6. evaluate and build the tables
python evaluation/run_intrinsic.py --tokenizer xlm-roberta-base checkpoints/vexmlm-expanded \
    --corpus amh=datasets/processed/amharic.dev.txt tir=datasets/processed/tigrinya.dev.txt
python evaluation/generate_tables.py --runs checkpoints --out results
```

## Method

### Vocabulary expansion

New tokens come from SentencePiece models trained on each language, and only
tokens **absent** from XLM-R are added — every original entry is preserved, and
IDs 0–250,001 keep their meaning, so the pretrained weights stay valid.

Token selection enforces a hard constraint: candidates must be **raw Unicode**
matching XLM-R's SentencePiece convention. `extract_new_tokens.py` and
`expand_xlmr_vocab.py` both abort on byte-level-BPE-encoded input, because such
tokens can never match Ge'ez text and would silently produce a vocabulary that
looks expanded but never fires. After expansion, a firing check confirms the new
tokens are actually reached on held-out text.

### Mean-based initialization

Each new embedding is set to the centroid of the source embedding space:

$$e_t = \frac{1}{|V_s|}\sum_{s \in V_s} e_s$$

Implemented as `global_mean` in
[`vocabulary_expansion/initialization.py`](vocabulary_expansion/initialization.py)
and the default everywhere. Three further strategies exist:

| Strategy | Role |
|---|---|
| `global_mean` | **Official method** (paper Sec. 3.3) |
| `random` | Table 5 ablation arm |
| `constituent_mean` | Research extension: centroid of a token's own subwords |
| `mixed` | Reference only; see [docs/REFERENCE_ARTIFACTS.md](docs/REFERENCE_ARTIFACTS.md) |

`vocabulary_expansion/verify_vocab.py` identifies which strategy produced any
checkpoint, from its weights alone, by comparing the observed
E‖e_new − ē‖ against each method's closed form.

## Tasks and data

| Task | Datasets | Metrics |
|---|---|---|
| Question Answering | TIGQA (tir), AmQA (amh) | Exact Match, F1 |
| Named Entity Recognition | MasakhaNER (amh), Tigrinya NER | Accuracy, Macro-F1 |
| Sentiment Analysis | AfriSenti | Accuracy |

Datasets are declared in [`datasets/registry.py`](datasets/registry.py) with
source, licence, and citation. MasakhaNER and AfriSenti load from the Hub;
TIGQA, AmQA, and Tigrinya NER have no verified public loader and must be
supplied with `--local-path` (see [docs/DATA_SETUP.md](docs/DATA_SETUP.md)).
Every load records row counts, content hashes, and label distributions to
`datasets/metadata/`, and `DatasetManager.write_card()` renders a dataset card
from them.

## Evaluation

**Intrinsic** (Tables 2–3): fertility (tokens/word), compression (chars/token),
parity, OOV accuracy.

Parity is only reported as valid when computed on a **sentence-aligned parallel
corpus** — pass `--parallel`. On non-parallel text the ratio reflects content
differences rather than tokenizer fairness, and the output says so rather than
presenting a misleading number.

**Extrinsic** (Tables 4–5): the task metrics above, aggregated over seeds
42–46 as mean ± std. Single-seed values are labelled as such;
`configs/seeds.yaml` sets a three-seed minimum for reporting.

`evaluation/generate_tables.py` writes `table2..table5` as CSV plus a rendered
`results/RESULTS.md`. Metrics with no run behind them are emitted as
`NOT YET MEASURED` — never estimated, never silently omitted.

## Hardware

GPU-first, with automatic device selection (CUDA → MPS → CPU) and precision
policy (BF16 on SM 8.0+, else FP16) in
[`src/vexmlm/device.py`](src/vexmlm/device.py). CPU works but is for debugging
and tests only.

```bash
sbatch scripts/slurm_pretrain.sh                      # Stage 1, 1× A100
sbatch scripts/slurm_pretrain_multigpu.sh             # Stage 1, 4× A100 (DDP)
sbatch scripts/slurm_finetune.sh sentiment afrisenti amh   # array job, 5 seeds
```

Multi-GPU runs via `torchrun`. Note that DDP multiplies the effective batch
size by the device count; the multi-GPU script lowers per-device batch size to
compensate, and every run logs its effective batch size.

Training resumes from the last checkpoint automatically.

## Experiment tracking

MLflow when installed, JSON records under `experiment_tracking/runs/` otherwise —
a missing logging library never costs a run. Each run records git commit and
dirty state, seed, config hash, dataset hashes, tokenizer hash, hardware, and
metrics.

## Repository layout

```
configs/          base.yaml (paper Table 1), seeds.yaml, ablations/
datasets/         registry.py, manager.py, prepare.py, cards/, metadata/
tokenizer/        SentencePiece training, token selection, evaluation
vocabulary_expansion/  expansion, the four init strategies, verification
pretraining/      Stage 1 continued MLM
finetuning/       Stage 2: qa/, ner/, sentiment/
evaluation/       intrinsic metrics, aggregation, table generation
scripts/          reproduce_paper.sh, SLURM jobs, artifact tooling
src/vexmlm/       shared library: geez, device, config, tracking
tests/            pytest suite
legacy/           pre-existing artifacts, imported with provenance
```

## Tests

```bash
pytest -q                      # full suite
pytest tests/test_geez.py -v   # script handling and encoding checks
```

## Citation

```bibtex
@article{teklehaymanot2026vexmlm,
  title   = {Vocabulary Expansion for Low-Resource African Languages:
             A Case Study in Amharic and Tigrinya},
  author  = {Teklehaymanot, Hailay},
  year    = {2026}
}
```

## License

Apache-2.0. Dataset licences are recorded per dataset in `datasets/cards/`;
several are unresolved and are marked as release blockers in
[docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).
