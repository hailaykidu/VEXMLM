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

## 19-Language Intrinsic Evaluation

Tokenizer behaviour is evaluated well beyond the two training languages, across
two intrinsic studies.

**Parity relative to English sentence length** — 11 languages:
Tigrinya, Amharic, Ge'ez, Tigre, Harari, Gurage, Afar, Oromo, Afrikaans, German,
Arabic.

VEXMLM achieves the most equitable tokenization on **9 of 11**, including
under-served Ge'ez-script languages (Ge'ez, Tigre, Harari, Gurage, Afar) and
non-Ge'ez controls (Afrikaans, German, Arabic).

**OOV word accuracy on NER** — 11 African languages:
Amharic, Tigrinya, Hausa, Igbo, Kinyarwanda, Luganda, Luo, Nigerian Pidgin,
Swahili, Wolof, Yoruba.

VEXMLM improves OOV accuracy on **all 11 languages**, with the largest gains on
Swahili, Kinyarwanda, and Nigerian Pidgin.

Together these span **19 distinct languages**, with Amharic and Tigrinya
appearing in both. Coverage extends across Ge'ez-script, Niger-Congo, Cushitic,
Semitic, and Germanic families — a substantially broader evaluation than the
two-language training scope.

---

## Major Results

Full tables in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

### Downstream performance

| Task (Metric) | XLM-R | VEXMLM | Glot500 |
|---|---|---|---|
| SA (Accuracy) | 0.77 | **0.80** | 0.46 |
| NER (Accuracy) | 0.75 | 0.78 | **0.92** |
| QA (EM) | 0.66 | **0.87** | 0.74 |
| QA (F1) | 0.78 | **0.90** | 0.78 |

VEXMLM leads question answering (**+21 EM**, **+12 F1** over XLM-R) and
sentiment analysis (**+3** over XLM-R, **+34** over Glot500). Glot500 attains
the highest NER accuracy.

### Ablation — Tigrinya NER OOV accuracy

| Configuration | OOV Acc. (%) | Δ |
|---|---|---|
| XLM-R baseline | 96.1 | — |
| + VocabExp (Random Init) | 97.3 | +1.2 |
| + VocabExp (Mean Init) | 97.8 | +0.5 |
| + Continued Pretraining | **98.2** | +0.4 |

Every component contributes. Mean initialization improves on random by +0.5;
continued pretraining adds a further +0.4.

### Intrinsic

- Parity: best on 9 of 11 languages
- OOV accuracy: improved on all 11 African languages evaluated

> ⚠️ The Table 3 XLM-R OOV column average and its stated improvement are under
> verification — see [TABLE3_VERIFICATION.md](internal/TABLE3_VERIFICATION.md).

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
| `vexmlm-stage1` (the paper's model) | Hugging Face Hub — link pending |

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
  the 19-language evaluation is intrinsic (tokenizer-level), not downstream.
- Parity requires sentence-aligned parallel text; the tooling marks it invalid
  rather than reporting a misleading figure when alignment is unavailable.
- Two pretraining corpora have unresolved provenance (tracked as release
  blockers).
- Glot500 outperforms VEXMLM on NER accuracy.

---

## License

Apache-2.0 for code. Dataset licences are recorded per dataset; several are
unresolved and tracked in [RELEASE_PACKAGE.md](RELEASE_PACKAGE.md) §6.

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
