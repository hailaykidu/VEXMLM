# VEXMLM — Final Public Release Package

**Paper**: *Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya*
**Venue**: LM4UC Workshop, IJCAI 2026
**Author**: Hailay Teklehaymanot
**Repository**: Official companion repository

---

## 1. Repository Overview

VEXMLM adapts `xlm-roberta-base` to the Ge'ez-script languages Amharic and
Tigrinya through vocabulary expansion, mean-based embedding initialization, and
two-stage training.

```
VEXMLM/
├── README.md · LICENSE · CITATION.cff · pyproject.toml
├── configs/              hyperparameters (paper Table 1)
├── datasets/             registry, loaders, provenance
├── tokenizer/            SentencePiece training and extraction
├── vocabulary_expansion/ expansion, initialization, verification
├── pretraining/          Stage 1 continued MLM
├── finetuning/           qa/ · ner/ · sentiment/
├── evaluation/           intrinsic metrics, aggregation, tables
├── scripts/              10 entry points (+ internal/)
├── src/vexmlm/           shared library
├── docs/ · results/ · reports/ · tests/
```

**Verified**: 25/25 tests passing · 0 broken links · no AI attribution ·
no absolute paths · root reduced to 4 files.

---

## 2. Main Contributions

**Vocabulary extension** — XLM-R's 250K vocabulary extended to 280K with ~30K
Ge'ez-derived subwords. All original entries preserved: appended, never replaced
or reordered.

**Embedding initialization** — new rows placed at the centroid of the pretrained
embedding space, e_t = (1/|V_s|) Σ e_s.

**Language-specific tokenization** — per-language SentencePiece Unigram models
(Amharic 32K, Tigrinya 50K), NFC-normalized with an Ethiopic-ratio filter, merged
by rank-interleaving after cross-language deduplication.

**Two-stage training** — continued MLM pretraining with all parameters
trainable, then task-specific fine-tuning.

**Intrinsic and downstream evaluation** — tokenizer parity and OOV accuracy
across 19 languages, plus QA, NER, and sentiment analysis.

---

## 3. Training Scope

| Language | ISO 639-3 | SentencePiece vocabulary |
|---|---|---|
| Amharic | amh | 32,000 |
| Tigrinya | tir | 50,000 |

Merged to ~30,000 new tokens; final model vocabulary **280,002**.

---

## 4. 19-Language Intrinsic Evaluation Scope

Two intrinsic studies spanning **19 distinct languages**.

**Parity relative to English sentence length** (11): Tigrinya, Amharic, Ge'ez,
Tigre, Harari, Gurage, Afar, Oromo, Afrikaans, German, Arabic.

**OOV word accuracy on NER** (11): Amharic, Tigrinya, Hausa, Igbo, Kinyarwanda,
Luganda, Luo, Nigerian Pidgin, Swahili, Wolof, Yoruba.

Amharic and Tigrinya appear in both. Coverage spans Ge'ez-script, Niger-Congo,
Cushitic, Semitic, and Germanic families — substantially broader than the
two-language training scope.

---

## 5. Downstream Evaluation Tasks

| Task | Datasets | Metrics |
|---|---|---|
| Question Answering | TIGQA (tir), AmQA (amh) | Exact Match, F1 |
| Named Entity Recognition | MasakhaNER (amh), Tigrinya NER | Accuracy |
| Sentiment Analysis | AfriSenti | Accuracy |

**Baselines**: `xlm-roberta-base`, `cis-lmu/glot500-base`.

---

## 6. Published Results

Full tables: [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

### Tokenizer parity (Table 2)
Most equitable tokenization on **9 of 11 languages**, including Ge'ez, Tigre,
Harari, Gurage, Afar, and the non-Ge'ez controls.

### OOV accuracy (Table 3)
Improved on **all 11 African languages**, **+5.5 points** on average; largest
gains Swahili (+15.0), Kinyarwanda (+8.5), Nigerian Pidgin (+7.4).

> ✅ **B1 correction applied.** Averages `95.2 | 88.9 | 96.3 | 94.4`; caption
> cross-reference corrected to Table 4.
> → [TABLE3_RELEASE_PATCH.md](TABLE3_RELEASE_PATCH.md)

### Downstream (Table 4)

| Task (Metric) | XLM-R | VEXMLM | Glot500 |
|---|---|---|---|
| SA (Accuracy) | 0.77 | **0.80** | 0.46 |
| NER (Accuracy) | 0.75 | 0.78 | **0.92** |
| QA (EM) | 0.66 | **0.87** | 0.74 |
| QA (F1) | 0.78 | **0.90** | 0.78 |

VEXMLM leads QA (+21 EM, +12 F1) and SA (+3 over XLM-R, +34 over Glot500).
Glot500 leads NER accuracy.

### Ablation (Table 5)

96.1 → 97.3 (random init) → 97.8 (mean init) → **98.2** (continued pretraining).

---

## 7. Released Checkpoints

| Checkpoint | Release | Size |
|---|---|---|
| **`vexmlm-stage1`** | ⏳ pending B2/B3 licence confirmation | ~1.2 GB |
| `vexmlm-expanded` | ⬜ optional | ~1.2 GB |
| Fine-tuned task checkpoints (25) | ❌ not released — regenerable; NER inherits CC BY-NC | — |
| Baseline checkpoints | ❌ not released — standard public models | — |

The 201 GB `checkpoints/` directory is excluded from git.

---

## 8. Dataset Availability

**No datasets are redistributed.** All link-only or blocked.

| Dataset | Licence | Status |
|---|---|---|
| TIGQA · AmQA · TiQuAD · AfriSenti | CC BY 4.0 / MIT | link-only |
| MasakhaNER-Amharic | CC BY-NC 4.0 ⚠️ | link-only, non-commercial |
| Tigrinya NER | none upstream ⚠️ | requires verification |
| Amharic / Tigrinya MLM corpora | HornMT portion **CC BY 4.0**; remainder unclarified ⚠️ | **not distributed** |

Details: [DATASET_DISTRIBUTION_PLAN.md](DATASET_DISTRIBUTION_PLAN.md).

---

## 9. Citation

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

`CITATION.cff` — `type: conference-paper`, venue recorded, validated.

---

## 10. Hugging Face Release

**Target**: `Hailay/vexmlm-stage1` — the official public baseline.

| Item | Status |
|---|---|
| Model + tokenizer + config files | ✅ present and consistent (280,002 both) |
| Model card draft | ✅ [HUGGINGFACE_RELEASE_PLAN.md](HUGGINGFACE_RELEASE_PLAN.md) |
| Metadata (`am`/`ti`, Apache-2.0, base model, tags) | ✅ specified |
| Exclusions | `training_args.bin`, `stage1_summary.json` |
| **Licensing condition** | ⚠️ **B2/B3** — card must state training data; HornMT confirmed CC BY 4.0 (~1% of corpus), remainder unclarified |
| **Documentation** | ✅ Known Limitation section drafted (B5 documented, not blocking) |

**Publishable with an explicit training-data disclosure**, or delay until the
remaining provenance is traced. All other conditions are met.

### Artifact roles

- **`vexmlm-stage1`** — the official LM4UC 2026 paper artifact. Publish this.
- **SP-Merge tokenizer** — validated successor with verified round-trip
  correctness, but an *untrained* expansion and **not** the released paper
  checkpoint. Release separately once trained.

---

## Release Status

| Target | Status |
|---|---|
| **Commit** | ✅ ready |
| **GitHub** | ✅ **READY** — with dataset acquisition instructions |
| **Hugging Face** | ⚠️ **CONDITIONAL** — provenance documented, ~99% licence unclarified |

**B1 applied · B4 resolved · B5 documented limitation · B2/B3 partially
resolved** (HornMT confirmed CC BY 4.0; covers ~1% of each corpus).

Full gate review: [FINAL_RELEASE_GATE.md](FINAL_RELEASE_GATE.md).
