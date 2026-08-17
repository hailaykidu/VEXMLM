# VEXMLM — Release Notes

**Version**: 1.0
**Date**: 2026-08-15
**Paper**: *Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya*

> ⚠️ Reconcile the title across README, BibTeX, and `CITATION.cff` before
> release — the release brief uses *"VEXMLM: Expanding the Lexicon of Ge'ez-Based
> African Languages."*

---

## Overview

VEXMLM adapts `xlm-roberta-base` to Ge'ez-script languages by extending its
subword vocabulary with language-specific tokens, initializing the new
embeddings from the centroid of the pretrained embedding space, and training in
two stages: continued MLM pretraining followed by task-specific fine-tuning.

---

## Contributions

**Vocabulary extension.** XLM-R's 250K vocabulary extended to 280K with ~30K
Ge'ez-derived subwords. All original entries preserved — tokens appended, never
replaced or reordered — so pretrained embeddings remain valid.

**Embedding initialization.** New rows initialized at the centroid of the
pretrained embedding space, e_t = (1/|V_s|) Σ e_s, placing them inside the
existing representation manifold rather than at random.

**Language-specific tokenization.** Per-language SentencePiece Unigram models
(Amharic 32K, Tigrinya 50K) with NFC normalization and Ethiopic-ratio filtering,
merged by rank-interleaving so each language contributes its highest-value
tokens.

**Two-stage training.** Stage 1 continues MLM pretraining with all parameters
trainable; Stage 2 fine-tunes per task.

**Intrinsic and downstream evaluation.** Tokenizer parity and OOV accuracy
alongside question answering, named entity recognition, and sentiment analysis.

---

## Training Languages

Vocabulary construction and continued pretraining cover **two languages**:

| Language | ISO 639-3 | SentencePiece vocabulary |
|---|---|---|
| Amharic | amh | 32,000 |
| Tigrinya | tir | 50,000 |

Merged to ~30,000 new tokens after cross-language deduplication.

---

## Evaluation Scope

**Intrinsic tokenizer evaluation** covers the two training languages, Amharic and
Tigrinya, on their development corpora — the languages for which OOV word lists
and dev splits exist in this repository. Results are in
[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

`evaluation/run_intrinsic.py` is language-agnostic and would extend to further
languages given corpora and OOV lists; none are included here, so no
broader-language intrinsic table is reported.

**Downstream evaluation** covers six tasks over Amharic and Tigrinya, five seeds
each — the authoritative result set.

**A separate multilingual set** (`results/multilingual_evaluation/`) additionally
covers AfriSenti across 12 languages, MasakhaNER2 across 20, and two zero-shot
transfer configurations, at five seeds. It uses different run configurations and
is deliberately not pooled with the authoritative downstream results.

---

## Major Results

Full tables in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

### Downstream performance

Mean ± std over seeds 42–46, from `results/downstream_task_metrics.csv`.

| Task | Dataset | Metric | VEXMLM |
|---|---|---|---|
| NER | MasakhaNER Amharic | Accuracy | 0.9413 ± 0.0026 |
| | | Macro-F1 | 0.7423 ± 0.0122 |
| NER | Tigrinya NER | Accuracy | 0.9515 ± 0.0005 |
| | | Macro-F1 | 0.8219 ± 0.0069 |
| QA | AmQA | EM / F1 | 32.57 ± 0.77 / 48.85 ± 0.96 |
| QA | TIGQA | EM / F1 | 2.39 ± 0.82 / 9.76 ± 0.97 |
| SA | AfriSenti (Amharic) | Accuracy | 0.4978 ± 0.0331 |

TiQuAD is **supplementary** — a diagnostic task, not a paper benchmark:
EM 50.24 ± 0.48, F1 58.90 ± 0.66.

Baseline XLM-R and Glot500 runs exist for seed 42 only and are not aggregated
here, so no multi-seed head-to-head comparison is stated.

### Ablation — downstream NER OOV accuracy

Tigrinya NER test, seeds 42–46, from `results/table5_ablation.csv`. All four arms
are scored on one OOV set: 3,491 of 4,677 word types (74.6%).

| Configuration | OOV Acc. (%) | Δ |
|---|---|---|
| XLM-R baseline | 94.57 ± 0.16 | — |
| + VocabExp (Random Init) | 87.04 ± 0.20 | **−7.52** |
| + VocabExp (Mean Init) | 87.63 ± 0.14 | +0.59 |
| + Continued Pretraining | **95.66 ± 0.09** | **+8.02** |

Vocabulary expansion alone *hurts* until Stage 1 adapts the new embedding rows;
continued pretraining is what makes it pay off, finishing +1.09 over the baseline.

### Intrinsic

Tigrinya fertility falls 45.9% (3.1300 → 1.6928) and compression rises 84.9%;
Amharic fertility falls 28.0%. OOV word round-trip reaches 1.0000 (amh) and
0.9987 (tir).

Parity is **not** reported: it requires a sentence-aligned parallel corpus, which
is unavailable here, and the artifact records `valid: false`.

---

## Repository Contents

| Component | Location |
|---|---|
| Tokenizer construction | `tokenizer/` |
| Vocabulary expansion + initialization | `vocabulary_expansion/` |
| Stage 1 MLM pretraining | `pretraining/run_mlm.py` |
| Stage 2 fine-tuning | `finetuning/{qa,ner,sentiment}/` |
| Intrinsic + downstream evaluation | `evaluation/` |
| Dataset registry and loaders | `datasets/` |
| Hyperparameters (paper Table 1) | `configs/base.yaml` |
| End-to-end pipeline | `scripts/reproduce_paper.sh` |
| Shared library | `src/vexmlm/` |
| Test suite | `tests/` |

**Reproduce everything:**

```bash
bash scripts/reproduce_paper.sh
```

Seven stages: data → tokenizers → expansion → Stage 1 → Stage 2 → evaluation →
tables. Every stage is idempotent; existing outputs are skipped unless `--force`.

---

## Models

| Model | Availability |
|---|---|
| `vexmlm-stage1-spm` | Hugging Face Hub — link pending |

Base: `xlm-roberta-base` · Vocabulary: 250,002 → 280,002 · Languages: amh, tir

---

## Data

**No datasets are redistributed.** `datasets/registry.py` records the source,
licence, and citation for each; `scripts/acquire_datasets.py` and
`docs/DATA_SETUP.md` cover acquisition.

⚠️ **MasakhaNER-Amharic is CC BY-NC 4.0** — models fine-tuned on it inherit a
non-commercial restriction.

---

## Known Limitations

- Vocabulary construction and continued pretraining cover **two languages**;
  the broader cross-language evaluation is intrinsic (tokenizer-level), not
  downstream.
- Parity requires sentence-aligned parallel text; the tooling marks it invalid
  rather than reporting a misleading figure when alignment is unavailable.
- Two pretraining corpora have unresolved provenance (tracked as release
  blockers).
- Glot500 outperforms VEXMLM on NER accuracy.

---

## License

Apache-2.0 for code. Dataset licences are recorded per dataset; several are
unresolved; see `datasets/registry.py` for the licence of every dataset.

---

## Citation

```bibtex
@inproceedings{teklehaymanot2026vexmlm,
  title     = {Expanding the Lexicon of Ge'ez Based African Languages:
               A Comparative Study of Amharic and Tigrinya},
  author    = {Teklehaymanot, Hailay},
  booktitle = {Proceedings of the Workshop on Language Models for
               Underserved Communities (LM4UC), IJCAI 2026},
  year      = {2026}
}
```
