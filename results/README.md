# Implementation Artifacts

Measurements produced by the VEXMLM implementation in this repository.

The published paper is the authoritative source for the reported results and
tables. See [RESULTS_SUMMARY.md](../reports/RESULTS_SUMMARY.md).

---

## Files

| File | Contents |
|---|---|
| `spmerge_tokenizer_metrics.json` | SP-Merge tokenizer round-trip evaluation on Amharic and Tigrinya OOV word lists, with fertility and compression on the development corpora |
| `downstream_task_metrics.csv` | Downstream task metrics across seeds 42–46 (NER, QA) |
| `aggregate.json` | Per-task aggregation backing the downstream metrics |
| `init_verification.json` | Embedding initialization verification for the expanded vocabulary |
| `mlm_corpus_analysis.json` | Stage 1 pretraining corpus composition |
| `gpu_environment.json` | Hardware and software environment record |

---

## SP-Merge tokenizer

Canonical tokenizer for VEXMLM. Vocabulary of 280,002 subwords, extending
`xlm-roberta-base` with Ge'ez-script tokens merged natively into the
SentencePiece model.

Measured on `datasets/processed/{amharic,tigrinya}.dev.txt` and the
corresponding OOV word lists:

| Metric | Amharic | Tigrinya |
|---|---|---|
| OOV word round-trip | 100.00% | 99.87% |
| Fertility (subwords per word) | 1.4888 | 1.6928 |
| Compression (chars per token) | 3.1896 | 2.6979 |
| Ge'ez token share | 24.2% | 45.5% |

Regenerate:

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

Parity is reported as valid only with `--parallel` on a sentence-aligned
corpus.

---

## Downstream tasks

`downstream_task_metrics.csv` records mean ± standard deviation over seeds
42–46 for named entity recognition and question answering on Amharic and
Tigrinya. Regenerate with:

```bash
python3 evaluation/generate_tables.py --runs checkpoints --out results
```
