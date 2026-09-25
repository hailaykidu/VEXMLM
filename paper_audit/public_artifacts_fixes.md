# Public copies of withdrawn claims — proposed corrections

Status: **proposals only.** Nothing listed here has been edited, committed or pushed.
Survey date 2026-09-24. GitHub was read with `git fetch` + `git grep` on `origin/*`;
Hugging Face cards were downloaded read-only from `huggingface.co/Hailay/<model>/raw/main/README.md`.

Evidence for every "correct value" below is in `evidence_ledger.csv` and `correction_notice.md`.

---

## A. GitHub — `hailaykidu/VEXMLM`

Remote branches: `main` and `fix/deterministic-evaluation` (both `3d96f3b`),
`release/v1.0-paper` (`c66c1a5`). `main` already carries the corrected results
(commit `3d96f3b`); `release/v1.0-paper` — the branch the paper points readers to — does not.

### A1. `release/v1.0-paper:reports/RESULTS_SUMMARY.md` — highest priority
| Line(s) | Published | Problem | Proposed fix |
|---|---|---|---|
| 7, 16–30 | Table 2 parity, 11 languages, incl. Glot500 (Tigrinya 1.36/1.20/0.27 …) | No parallel corpus; 9 of 11 languages never evaluated; Glot500 values untraceable | Remove table. Replace with fertility/compression/round-trip for XLM-R, Glot500, VEXMLM from `results/spmerge_tokenizer_metrics.json` (Glot500: `results/provenance/tokenizer_metrics_2026-08-15.json`) |
| 40–55 | Table 3, 11-language NER OOV accuracy | No generating artifact | Remove |
| 65–76 | Table 4 with Glot500 column; VEXMLM SA 0.80, QA EM 0.87, F1 0.90; "significantly outperforms … +21 EM, +12 F1" | Glot500 column untraceable; VEXMLM values contradicted by 5-seed results; no significance test exists | Replace with `results/downstream_task_metrics.csv` + XLM-R (1 seed) from `results/baselines/xlmr_seed42/`; drop Glot500 and "significantly" |
| 86–99 | Table 5: 96.1 / 97.3 / 97.8 / 98.2 | Contradicted by `results/table5_ablation.csv` | Replace with 94.57 / 87.04 / 87.63 / 95.66 (± std, 5 seeds) |
| 113–120 | Prose summary repeating the above | — | Rewrite to the corrected claims |

Simplest route: replace the file with `main:reports/RESULTS_SUMMARY.md` plus the Glot500
tokenizer row, and add a banner linking the correction notice.

### A2. `release/v1.0-paper:reports/RELEASE_NOTES.md`
| Line(s) | Published | Proposed fix |
|---|---|---|
| 78 | "Together these span **19 distinct languages**" | Remove (no generating artifact; the listed union is 20, not 19) |
| 93–98 | Table 4 with Glot500; "+21 EM, +12 F1" | As A1 |
| 106–109 | Table 5 old values | As A1 |
| 185 | "Glot500 outperforms VEXMLM on NER accuracy" | Remove (no traceable Glot500 NER result) |

### A3. `reports/MODEL_CARD_VEXMLM_BASE.md` — both branches
| Line | Published | Correct | Source |
|---|---|---|---|
| 69–76 | Vocabulary composition totalling **576,520** | **280,002** (250,002 + 30,000) | `checkpoints/vexmlm-stage1-spm/config.json` |
| 110–113 | "HornMT … ~1% (2,030 records) … CC BY 4.0 … ✅ confirmed" | HornMT is ruled out as a source; corpus origin and licence unknown | `docs/DATASET_PROVENANCE.md`, `docs/MLM_CORPUS_ANALYSIS.md` |
| 148 | "Best validation loss 4.5764 (step 4,608 / epoch 9)" | Superseded checkpoint. Released model: best eval loss 3.7120 at epoch 56, step 24,808 | `checkpoints/vexmlm-stage1-spm/checkpoint-24808/trainer_state.json` |
| 161, 167 | Pre-repair single-seed numbers (Amharic NER 42.70%, AmQA EM 19.33%) | 5-seed values | `results/downstream_task_metrics.csv` |

Proposed fix: regenerate from the Hugging Face `Hailay/VEXMLM` card (already largely correct, see B1) or delete and point to it.

### A4. HornMT attribution — both branches
| File | Line | Current | Proposed |
|---|---|---|---|
| `datasets/cards/amharic_mlm.md` | 10 | `Source: https://github.com/asmelashteka/HornMT` | `Source: unknown (see docs/DATASET_PROVENANCE.md)` |
| `datasets/cards/tigrinya_mlm.md` | 10 | same | same |
| `datasets/registry.py` | 216, 231 | `url="https://github.com/asmelashteka/HornMT"` | `url=None` (notes already explain) |
| `scripts/acquire_datasets.py` | 84, 93 | `"official_url": ".../HornMT"` | `None`, with a comment pointing to the provenance doc |
| `datasets/metadata/acquisition.json` | 169, 193 | same | regenerate after the script change |

Wording follows the chosen data statement (Option A in `data_statement_draft.md`): source unknown.

### A5. `README.md` — both branches
| Line | Current | Proposed |
|---|---|---|
| 1 | Old title "Expanding the Lexicon of Ge'ez Based African Languages …" | Corrected paper title (once final) |
| 59–60 | Lists "parity" among reported tokenizer metrics | Remove "parity" |

### A6. `CITATION.cff` — both branches
Title fields ("VEXMLM: Vocabulary-Extended XLM-R …" and the inner `title: >-`) must match the
corrected paper title and, once issued, reference the correction.

Not affected: `reports/PRESENTATION_NOTES.md` (its figures, e.g. Tigrinya fertility 3.1 → 1.7,
match the SP-Merge results); `src/`, `scripts/gpu_probe.py`, `docs/OFFICIAL_TRAINING_PLAN.md`
mention fp16 only as a hardware fallback, which is accurate.

---

## B. Hugging Face — `huggingface.co/Hailay`

### B1. `Hailay/VEXMLM` (main model card)
Downstream numbers, Table 5 ablation, fertility/compression and hyperparameters **already match**
the repository (5-seed means, parity not reported, bf16, 56 of 60 epochs). Remaining issues:

| Section | Current | Problem | Proposed |
|---|---|---|---|
| Training | "eval loss 3.7120, perplexity 41.67" | Mixes two evaluations: 3.7120 is the best in-training eval loss (exp = 40.93); 41.67 = exp(3.7299), the final re-evaluation | "best eval loss 3.7120 (perplexity 40.93)" |
| Training | No statement of corpus origin or licence | Origin and licence unknown | Add the Option A data statement from `data_statement_draft.md` |
| Limitations | "The corpora are drawn largely from religious and news domains" | Measured: religious 7–9%, news ~1%, ~88–90% unmatched by any probe | "Samples contain religious translations and general web text; the corpus origin is undocumented" |
| Limitations | "XLM-R and Glot500 comparison runs exist for seed 42 only" | Glot500 downstream runs are no longer reported | Drop "and Glot500" |
| Licence | `apache-2.0` | Weights are trained on text of unknown licence | Author decision; at minimum note the unresolved corpus licence |
| Citation | Old title | — | Corrected title |

### B2. Downstream cards
`Hailay/VEXMLM-Amharic-NER`, `-Tigrinya-NER`, `-AmQA`, `-TIGQA`, `-AfriSenti-Amharic`, `-TiQuAD`:
- the same "religious and news domains" limitation line → same fix as B1;
- citation title → corrected title;
- `-AmQA`, `-TIGQA`, `-TiQuAD`: "on the dataset's **test** split" is wrong. `run_qa.py` evaluates on the
  validation split when one exists (AmQA 600 dev questions, TIGQA 67 dev questions); state "development split".
  The `Hailay/VEXMLM` card's "67 test questions" (TIGQA) needs the same fix.
They contain no Glot500 claim and no withdrawn number.

### B3. `Hailay/glot500-multilingual-sentiment`
- Heading reads "glot500-multilingual-sentiment **(VEXMLM)**", but the model is the 2024 Glot500
  sentiment prototype trained on a custom multilingual TSV (test accuracy 0.6624, matching
  `~/VEXMLM/eval_results/scores.txt`). It is not a VEXMLM model and not a paper result.
  Proposed: remove "(VEXMLM)" and add one line stating it is unrelated to the paper's Table 4.
- `Hailay/glot500fine-tuned-model` (the name used in `~/git-lfs-3.2.0/upload_model.py`) does not
  exist publicly (HTTP 401). No action unless it exists privately.

### B4. Paper resource link
The paper's "Hugging Face models" link `https://huggingface.co/collections/Hailay/vexmlm`
returns **HTTP 404**. Either create the collection (VEXMLM + six downstream models) or change the
link to `https://huggingface.co/Hailay/VEXMLM`.

### Not affected
`Hailay/EXLMR`, `Hailay/FT_EXLMR` (earlier Tigrinya/AfriSenti classifiers), `Hailay/xlmr-amharic-mlm`,
`Hailay/xlmr-tigrinya-mlm` and the unrelated MT/tokenizer models make no claim from this paper.

---

## C. Order of operations (proposal)
1. Data statement (Option A) and title are decided; create the release tag `v1.1-correction` when committing.
2. Update `release/v1.0-paper` (A1, A2) — this is the branch readers of the paper reach.
3. Update `main` (A3–A6).
4. Update Hugging Face cards (B1–B3) and fix the collection link (B4).
5. Publish the correction notice with the corrected paper.
