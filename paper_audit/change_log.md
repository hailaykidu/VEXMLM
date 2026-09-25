# Change log: published draft → corrected paper

Every substantive content change, its reason and its evidence is a row of `correction_notice.md`
(IDs in brackets below). This file records the changes to structure and production.

## Structure
| Change | Reason |
|---|---|
| Title → *Vocabulary Expansion for Low-Resource African Languages: A Case Study in Amharic and Tigrinya* | Author instruction |
| Table 2: parity (11 languages × 3 tokenizers) → fertility, compression, round-trip (Amharic, Tigrinya × 3 tokenizers) | Parity invalid; other languages never evaluated [T2-1] |
| Table 3: 11 language rows + average → Tigrinya row only; same columns | Only Tigrinya has an OOV-split evaluation [T3-1] |
| Table 4: Glot500 column removed; rows split per dataset; scores in % with ± SD | Glot500 untraceable; published single values match no dataset [T4-*] |
| Table 5: same rows; values ± SD; Δ column shows change from previous row | 5-seed results [T5-*] |
| Figure 1: English bars removed; values from stored results (`results/spmerge_tokenizer_metrics.json`, `results/provenance/tokenizer_metrics_2026-08-15.json`) | No English measurement [F1-7] |
| Figure 2 removed | No generating script; mislabelled base model [S-5] |
| Appendix: Table 6 (19-language list) → Table 6 (macro-F1, entity-F1); new Sec. 9 Reproducibility | Draft footnote promised macro-F1 [T4-15]; language list unsupported [S-1] |
| Resources link → `huggingface.co/Hailay/VEXMLM` | Collection link returns 404 [S-6] |
| Tables 3 and 5: "OOV" relabelled "words fragmented by XLM-R"; Table 3 columns Other / Fragm. | The word set is defined by the expanded tokenizer [T3-6, T3-7] |
| Results reported exactly as in GitHub's result files and the Hugging Face card (same values, precision, units; differences from unrounded means, as there) | No inconsistency with the public artifacts [K-11] |
| References: WECHSEL reworded; Mitchell, Richburg, Weber removed; Gebremedhin venue, Petrov year, Glot500 wording corrected; AfriSenti dataset paper added | [R-1 … R-9] |
| Limitations: single-seed baseline, ablation confound, data provenance, QA trailing-space answers added | [S-7] |

## Production
| Change | Reason |
|---|---|
| All result numbers are macros from `paper/generated/results_macros.tex`; tables `\inputtable` generated bodies | Traceability; `scripts/make_paper_artifacts.py` |
| `\inputtable` (primitive input) for table bodies | `\input` before `\bottomrule` fails to compile |
| Font: Times New Roman if installed, else Nimbus Roman; `inconsolata` only if installed | Neither is installed on the build machine |
| Bibliography reconstructed from the published list; two garbled entries corrected; three flagged | See `draft_audit.md` §6 |

## Repository additions (uncommitted)
`paper/`, `paper_audit/`, `REPRODUCE.md`, `scripts/make_paper_artifacts.py`,
`scripts/slurm_xlmr_baseline_5seeds.sh` (not run), `tests/test_paper_artifacts.py`,
`results/baselines/xlmr_seed42/`, `results/provenance/` (copies of earlier run records), `paper_audit/`
(including `paper_audit/verification/`: runs made during the correction, not used as paper sources).
No existing file was modified; the implementation is identical to GitHub `main` (a pipeline-script
fix was drafted and reverted: the implementation is documented, not changed). The only computation run was the tokenizer evaluation
(`evaluation/run_intrinsic.py`, as instructed), parameter counting, dataset row counting, the QA
span check, the word-set composition count and a re-run of token selection (identical output);
no training or fine-tuning was run.

## Not produced
`paper_audit/draft_vs_final.pdf` (latexdiff): no LaTeX source of the published draft exists and
`latexdiff` is not installed. `correction_notice.md` lists every change instead.
