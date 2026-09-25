# Audit of the published draft

Draft: *Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and
Tigrinya* (published PDF; no LaTeX source was available, so the corrected LaTeX was reconstructed
from the PDF text). Evidence: repository at commit `1e56e55` plus the tracked copies in
`results/provenance/` and `results/baselines/` (copies of earlier run records).

Claim-by-claim audit of the draft: `draft_claims_ledger.csv` (89 rows; built by an audit agent and
spot-checked — where it differs from this file or `correction_notice.md`, those take precedence:
it calls the SA difference "sign-reversed" although it is within one standard deviation, and it
cites single-seed Glot500 runs that are not reported).

## 1. Numeric discrepancies
All draft → verified values, with sources, are in `correction_notice.md` (rows F1-*, T2-*, T3-*,
T4-*, T5-*, M-*, TK-*). In summary:

| Draft element | Finding |
|---|---|
| Figure 1 | No published value matches any stored tokenizer evaluation; English was never measured |
| Table 2 (parity) | Invalid without a parallel corpus; 9 of 11 languages never evaluated; Glot500 column untraceable |
| Table 3 (OOV, 11 languages) | No evaluation exists for 10 of 11 languages; Tigrinya values contradicted by the 5-seed ablation |
| Table 4 | VEXMLM and XLM-R values contradicted by result files; Glot500 column untraceable |
| Table 5 | All four values contradicted by the 5-seed ablation; body-text deltas (+2.3, +7.1) do not even match the draft's own table (+0.5, +0.4) |
| Sec. 5.1 vs 5.4 | Average OOV gain stated as +5.9 and +12.9; neither is supported |
| Table 1 | fp16 (actual bf16); 56 epochs (60 configured, epoch 56 released); frozen body (all parameters trained) |
| Sec. 4.1 | 279M / 301M mix counting conventions (278.3M / 301.4M under one convention) |
| Sec. 4.3 | MasakhaNER listed with MISC (Amharic MasakhaNER has DATE, not MISC) |

## 2. Untraceable claims (no support in the repository)
| Claim | Action |
|---|---|
| "Improvements … transfer to 17 languages"; evaluation "across 19 languages" (Table 6 lists 20) | Removed |
| All Glot500 downstream values | Removed (S-4) |
| Tokenizer sizes "calibrated to minimize OOV rates on held-out evaluation corpora" | Removed |
| Mean initialization gives "faster convergence" | Removed |
| Figure 2 (vocabulary size per language, labelled xlm-roberta-large) | Removed; no generating script |
| "Curated monolingual corpora"; "All datasets used are publicly available" | Replaced; data statement pending (C-DATA) |

## 3. Method mismatches (the paper now describes the code)
| Draft | Code / run records |
|---|---|
| Embeddings initialized "via subword-averaging" (abstract) | Global mean of all original embeddings (Eq. 2) |
| New tokens = all pieces absent from XLM-R, yielding 30,000 | Top 20,000 per language by corpus frequency, rank-interleaved, deduplicated to 30,000 |
| Integration unspecified | Pieces merged into XLM-R's SentencePiece model |
| Fine-tuning updates only embeddings and head | All parameters are fine-tuned |
| fp16 | bf16 |
| "Constant learning rate … and linear schedule" | Linear decay after warmup |
| Model selection unspecified | NER: best validation macro-F1; SA: best validation accuracy; QA: final epoch |
| QA evaluated on "test" (implied) | QA scored on the development splits |
| Ablation "VEXMLM (Full) … + fine-tuning" as a separate step | All four ablation models are fine-tuned identically |
| "Low-resource language data is upsampled" | Both languages resampled with p ∝ n^0.5 (Amharic up, Tigrinya down) |

## 4. Overclaims
| Draft | Why unsupported |
|---|---|
| "Substantially outperforms XLM-R and Glot500 across all evaluated tasks" | QA below XLM-R; SA comparable; no Glot500 downstream result |
| "Consistently outperforms strong multilingual baselines" (Conclusion) | as above |
| "Every stage of the method contributes" / monotonic ablation | Expansion alone lowers accuracy |
| "Glot500 … larger parameter count … greater representational capacity" explanation of NER | Explains a result that does not exist |
| "Critically, improvements … transfer to 17 languages" | No evaluation |

No significance test exists in the repository; the corrected paper makes no significance claim and
describes differences within one standard deviation of VEXMLM as comparable.

## 5. Missing reproducibility details (added)
Seeds and aggregation (5 seeds, mean ± SD); single-seed XLM-R baseline; evaluation splits per task;
model-selection rule; token-selection procedure; tokenizer settings; corpus filtering and split
sizes; Stage 1 sampling and validation split; hardware; parameter-counting convention; exact
commands (`REPRODUCE.md`); a generated-artifact test (`tests/test_paper_artifacts.py`).

Still missing (recorded, not filled in): origin and licence of the pretraining text (disclosed as
undocumented); `transformers`, `datasets` and `sentencepiece` versions used for training. The
release tag `v1.1-correction` is named in the paper and still has to be created.

## 6. References
See `correction_notice.md` rows R-1 to R-9. Resolved: WECHSEL misattribution (R-1), Richburg and
Carpuat (R-3), Weber et al. (R-4), Petrov year (R-6), Glot500 wording (R-7), AfriSenti dataset
paper (R-8), Conneau and Wang entries (R-9). R-2 is resolved with author-supplied
references (Phillipson 2012; Getatchew Haile 1996). R-5: venue corrected; the four counts that
could not be verified (347, 45, 24 of 29, 26) are removed on the authors' decision.

## 7. External review (2026-09-24) — disposition
| Point | Disposition |
|---|---|
| WECHSEL misattributes mean initialization | Accepted (R-1). Not adopted: citing Hong et al. (2024) for the method — MULTI-INIT averages each token's own subword embeddings, whereas VEXMLM gives every new token one global mean |
| Mitchell, Richburg, Weber citations | Accepted (R-2, R-3, R-4) |
| "OOV" set is circular | Confirmed: all 3,491 types enter by the "more pieces than the expanded tokenizer" clause. Renamed "fragmented words"; clause counts and tag mix reported (T3-6, T3-7) |
| Deltas computed from unrounded values | Superseded by the authors' requirement that the paper match GitHub and Hugging Face exactly: results are given at the published precision and differences are computed from unrounded means, as in the public artifacts; the convention is stated in Sec. 5 (K-11) |
| Co-author name "Nejd"/"nejd1" | Not an error: source and compiled PDF read "Nejdl"/"nejdl"; text-extraction artefact |
| "Rekasaz", "Karagaran" | Not errors in this version: bibliography has Rekabsaz and Kargaran |
| "Imani-Googhari" | Line-break hyphenation; `\hyphenation{ImaniGooghari}` added |
| Gebremedhin venue | Accepted (R-5); cited figures flagged |
| Petrov 2023, Glot500 wording, AfriSenti dataset paper | Accepted (R-6, R-7, R-8) |
| AmQA 2,622 vs 2,628; QA on development split | Stated in Sec. 4.3 (M-9) |
| Table 4 scrambled | Not an error in the rendered PDF; extraction artefact |
| Promote Table 6 into the main results | Not done (structure kept for the formal correction); macro-F1 gains now stated in Sec. 5.2 — author decision |
