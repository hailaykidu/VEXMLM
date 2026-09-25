# Pretraining-corpus data statement — DRAFT for author decision

Status: **decided — Option A** (2026-09-24). The paper's Sec. 4.3 and Ethical Considerations use it.

## 1. What the repository establishes

Sources: `docs/DATASET_PROVENANCE.md`, `docs/MLM_CORPUS_ANALYSIS.md`,
`results/mlm_corpus_analysis.json`, `datasets/processed/preparation_report.json`,
`logs/stage1_spm_60674.err`.

| Fact | Amharic | Tigrinya | Source |
|---|---|---|---|
| Imported file | 200,002 lines (200,001 non-empty) | 200,000 lines | MLM_CORPUS_ANALYSIS.md |
| Unique lines | 129,864 (35.07% duplicates) | 141,015 (29.49% duplicates) | MLM_CORPUS_ANALYSIS.md |
| Whitespace tokens | 1,902,258 | 1,531,950 | MLM_CORPUS_ANALYSIS.md |
| Ge'ez share (non-space chars) | 94.3% | 96.0% | MLM_CORPUS_ANALYSIS.md |
| After filtering + dedup | 129,402 kept | 140,583 kept | preparation_report.json |
| Train / dev split (2%, seed 42) | 126,814 / 2,588 | 137,772 / 2,811 | preparation_report.json |
| Sampling during Stage 1 (alpha = 0.5) | 129,552 | 135,034 | logs/stage1_spm_60674.err |
| Origin | UNKNOWN | UNKNOWN | MLM_CORPUS_ANALYSIS.md verdict |
| Licence | UNKNOWN | UNKNOWN | datasets/registry.py |

Filtering (`datasets/prepare.py`): NFC normalisation, removal of zero-width/bidi controls,
whitespace collapsing, lines < 10 characters dropped, lines < 50% Ethiopic dropped,
exact duplicates removed.

Composition probes (coarse, overlapping): Christian religious 7.1% / 8.9%, Islamic
religious 2.6% / 0.1%, news 0.8% / 0.9%, web/commerce 1.5% / 0.4%, ~88% / ~90% unmatched.
Manual sampling: Biblical and Qur'anic translation passages alongside general web prose.

Both files were imported from a prepared local workspace (`~/lgse-repro/data/lapt/`) that
carries no manifest naming their source.

## 2. What is ruled out

The repository's designated source, HornMT (github.com/asmelashteka/HornMT), cannot be the
origin: HornMT holds ~2,030 aligned sentence pairs, the files hold 200,000 monolingual lines
each (~99× larger), and there is zero exact line overlap with any HornMT copy on the machine.

The published paper itself does **not** cite HornMT; it says "curated monolingual corpora"
and, in Ethical Considerations, "All datasets used are publicly available. We do not
introduce new data collection." Both statements are unsupported for the pretraining corpus.
HornMT is named in public repository files — see `public_artifacts_fixes.md`.

## 3. Proposed paper text — choose one option

**Option A — full disclosure (recommended if the source stays unknown).**
> Continued pretraining uses one Amharic and one Tigrinya monolingual text file of
> 200,001 and 200,000 non-empty lines, taken from a prepared local collection whose original source
> is not documented. We could not identify the source: the files match no named corpus
> available to us, and manual samples contain religious translations alongside general web
> prose. Their licence is therefore unknown, and we do not redistribute them. After
> NFC normalisation, removal of lines shorter than 10 characters or below 50% Ge'ez
> characters, and exact deduplication, 129,402 Amharic and 140,583 Tigrinya lines remain;
> 2% of each is held out as a development set.

**Option B — the authors identify the source before submission.** Replace the second and
third sentences with the confirmed source, citation and licence, and update
`datasets/registry.py`, the dataset cards and the model card to match.

**Option C — minimal.** State the line counts and filtering, and add to Limitations that the
origin and licence of the pretraining text are undocumented. (Weaker: a reader of the method
section is not told.)

Under every option:
- "curated monolingual corpora" (Abstract, Sec. 3.2, Sec. 3.4) becomes "monolingual corpora";
- the Ethical Considerations sentence "All datasets used are publicly available" is restricted
  to the downstream benchmarks, and the pretraining text is named as the exception;
- a correction-notice row records the change (see `correction_notice.md`, row C-DATA).
