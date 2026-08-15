# GitHub Release Package

**Date**: 2026-08-15
**Release tag**: `v1.0-paper`

---

## 1. Repository Title

**VEXMLM** — Vocabulary-Extended XLM-R for Amharic and Tigrinya

---

## 2. Repository Description

> Official repository for *"Expanding the Lexicon of Ge'ez Based African
> Languages: A Comparative Study of Amharic and Tigrinya"* (LM4UC Workshop,
> IJCAI 2026). Vocabulary expansion of XLM-R for Amharic and Tigrinya, with
> intrinsic tokenizer evaluation across 19 languages.

**Topics**: `nlp` · `low-resource-languages` · `amharic` · `tigrinya` · `geez` ·
`tokenization` · `vocabulary-expansion` · `xlm-roberta` · `multilingual` ·
`ijcai2026`

---

## 3. Release Notes Summary

### VEXMLM v1.0-paper

Official companion repository for the LM4UC 2026 paper.

**Method** — XLM-R's 250K vocabulary extended to 280,002 with ~30K Ge'ez-derived
subwords (all original entries preserved), new embeddings initialized at the
centroid of the pretrained embedding space, then two-stage training: continued
MLM pretraining followed by task-specific fine-tuning.

**Results** — most equitable tokenization on 9 of 11 languages (Table 2);
improved OOV accuracy on all 11 African languages, +5.5 points on average
(Table 3); leads QA (+21 EM, +12 F1) and sentiment (+3) over XLM-R (Table 4);
every component contributes in ablation (Table 5).

**Included** — full pipeline (`scripts/reproduce_paper.sh`, 7 stages), dataset
registry with sources and licences, intrinsic and downstream evaluation, 25
passing tests.

**Not included** — datasets (link-only; see `docs/DATA_SETUP.md`) and
checkpoints (`vexmlm-stage1` released separately on Hugging Face).

**Known limitation** — the released tokenizer does not exactly round-trip
Ge'ez text through `decode()`. Training, evaluation, offsets, and all published
results are unaffected. Documented in the model card.

---

## 4. Final Directory Structure

```
VEXMLM/
├── README.md                     public release README
├── LICENSE                       Apache-2.0
├── CITATION.cff                  conference-paper, venue recorded
├── pyproject.toml
│
├── configs/                      base.yaml (Table 1) · modes.yaml · seeds.yaml · ablations/
├── datasets/                     registry.py · manager.py · prepare.py · cards/
├── tokenizer/                    SentencePiece training · extraction · evaluation
├── vocabulary_expansion/         expansion · initialization · verification
├── pretraining/                  run_mlm.py
├── finetuning/                   qa/ · ner/ · sentiment/
├── evaluation/                   run_intrinsic.py · aggregate.py · generate_tables.py · metrics/
│
├── scripts/                      10 public entry points
│   └── internal/                 18 archived
│
├── src/vexmlm/                   geez · device · config · modes · tracking
├── docs/                         setup · provenance · redistribution
├── results/                      table2–5 CSV · RESULTS.md · tokenizer_metrics.json
│
├── reports/                      public documentation
│   └── internal/                 62 archived (audit and development history)
│
└── tests/                        25 tests, all passing
```

**Excluded** (`.gitignore`): `checkpoints/` (201 GB) ·
`datasets/{raw,processed,cache}/` · `logs/` · `legacy/` ·
`experiment_tracking/mlruns/` · `slurm-*.out`

---

## 5. Public Documentation

| Document | Purpose |
|---|---|
| `README.md` | entry point — overview, install, data, training, evaluation, results |
| `reports/RESULTS_SUMMARY.md` | published Tables 2–5 |
| `reports/DOCUMENTATION_OVERVIEW.md` | component-by-component reference |
| `reports/MODEL_CARD_VEXMLM_BASE.md` | model card |
| `reports/MLM_FINAL_REPORT.md` | Stage 1 training record |
| `reports/MULTISEED_SUMMARY.md` | seed variance |
| `reports/PRESENTATION_NOTES.md` | talk support |
| `reports/RELEASE_NOTES.md` | release summary |
| `reports/LICENSE_FINAL_STATUS.md` | dataset licence classification |
| `docs/DATA_SETUP.md` | dataset acquisition |
| `docs/DATASET_PROVENANCE.md` | sources, splits, upstream cleaning |
| `docs/DATASET_REDISTRIBUTION.md` | redistribution constraints |

---

## 6. Citation

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

`CITATION.cff` — `type: conference-paper`, `conference.name: LM4UC Workshop,
IJCAI 2026`, validated YAML.

---

## 7. Venue

**LM4UC Workshop, IJCAI 2026** — Workshop on Language Models for Underserved
Communities, International Joint Conference on Artificial Intelligence.

**Paper**: *Expanding the Lexicon of Ge'ez Based African Languages: A
Comparative Study of Amharic and Tigrinya*
**Author**: Hailay Teklehaymanot, L3S Research Center, Leibniz University Hannover

Add DOI and page numbers to `CITATION.cff` when proceedings publish.

---

## 8. Release Tag

### `v1.0-paper` ✅ recommended

Signals the as-published state: the repository and checkpoint exactly as
evaluated for LM4UC 2026. The `-paper` suffix distinguishes archival
reproducibility from later development, leaving `v1.1` / `v2.0` free for the
SP-Merge successor.

```bash
git tag -a v1.0-paper -m "VEXMLM v1.0-paper — LM4UC Workshop, IJCAI 2026"
git push origin v1.0-paper
```

**Release title**: `v1.0-paper — LM4UC Workshop, IJCAI 2026`
**Body**: §3 above.

---

## Pre-Push Checklist

- ✅ Tests passing (25/25)
- ✅ Cross-links resolve (0 broken)
- ✅ No AI attribution anywhere
- ✅ No absolute paths or TODO markers in public code
- ✅ Title and venue consistent
- ✅ `CITATION.cff` valid
- ✅ Table 3 correction applied
- ⬜ Confirm `repository-code` URL matches the public URL
- ⬜ Confirm LICENSE copyright holder and year
- ⬜ Update registry notes for the two MLM corpora (HornMT confirmed; remainder pending)
