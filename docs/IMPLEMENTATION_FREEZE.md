# Implementation Freeze

The VEXMLM implementation is frozen. No further architectural changes will be
made without an explicit decision to lift the freeze for a named component.

| Field | Value |
|---|---|
| **Commit** | `3298cf321c4ba67178146d116c84781264306bf3` |
| Short hash | `3298cf3` |
| Tree hash | `25c86ea7a4bbc0159dec2f68597da420c059c61a` |
| **Date** | 2026-08-14 20:55:45 +0200 |
| Tracked files | 88 |
| Python modules | 33 files, 5,548 lines |
| Tests | 25, all passing |

## Frozen components

| Module | Path | Lines | SHA-256 (12) | Role |
|---|---|---|---|---|
| Vocabulary Expansion | `vocabulary_expansion/expand_xlmr_vocab.py` | 183 | `34469de5c106` | expansion + encoding gate + firing check |
| Mean-Based Initialization | `vocabulary_expansion/initialization.py` | 206 | `62c550f83514` | global_mean default; random/constituent_mean/mixed |
| Init Verification | `vocabulary_expansion/verify_vocab.py` | 125 | `861772b2a896` | identifies init method from weights |
| Vocabulary Merge | `vocabulary_expansion/merge_vocabularies.py` | 109 | `0bbd207bf62c` | rank-interleaved cross-language merge |
| Amharic Tokenizer | `tokenizer/train_amharic_tokenizer.py` | 61 | `2cd3262a4a55` | SentencePiece 32K |
| Tigrinya Tokenizer | `tokenizer/train_tigrinya_tokenizer.py` | 61 | `843923fcd59b` | SentencePiece 50K |
| Joint Tokenizer | `tokenizer/train_joint_tokenizer.py` | 117 | `a21e760fe89d` | alpha-rebalanced |
| Token Selection | `tokenizer/extract_new_tokens.py` | 145 | `1ff56de085be` | raw-Unicode gate, frequency ranking |
| MLM Pipeline (Stage 1) | `pretraining/run_mlm.py` | 250 | `d1e401c0d430` | continued MLM, all params trainable |
| QA Pipeline | `finetuning/qa/run_qa.py` | 307 | `0f61392e73da` | span extraction, EM/F1, Ge'ez normalization |
| NER Pipeline | `finetuning/ner/run_ner.py` | 228 | `deda126493ff` | token classification, accuracy/macro-F1/entity-F1 |
| Sentiment Pipeline | `finetuning/sentiment/run_sentiment.py` | 205 | `819a264688bc` | sequence classification, accuracy/macro-F1 |
| Dataset Manager | `datasets/manager.py` | 348 | `d0ea28577fa7` | registry-driven loading, hashing, cards |
| Evaluation (intrinsic) | `evaluation/metrics/intrinsic.py` | 132 | `d6f20bbf0e8f` | fertility, compression, parity, OOV |
| Table Generation | `evaluation/generate_tables.py` | 246 | `7b8b1bd92173` | Tables 2-5 from run records |
| Aggregation | `evaluation/aggregate.py` | 105 | `43a0bfad0bc8` | multi-seed mean/std/min/max/median |
| Device Layer | `src/vexmlm/device.py` | 135 | `ba3b5d1d1a05` | CUDA/MPS/CPU, bf16/fp16, DDP |
| Tracking | `src/vexmlm/tracking.py` | 189 | `e3d90e5e0ba4` | MLflow + JSON fallback, full provenance |
| Experiment Modes | `src/vexmlm/modes.py` | 91 | `15e4bd09139a` | debug / research / official |

## What "frozen" means

**Frozen** — the method and its interfaces:

- the vocabulary expansion procedure and its encoding/firing gates;
- mean-based initialization as the official default, and the four strategies;
- the two-stage training structure;
- the QA / NER / sentiment pipelines and their metric definitions;
- config schema and the paper's Table 1 hyperparameters;
- dataset registry semantics and provenance recording.

**Not frozen** — things that must stay changeable:

- bug fixes, with the defect recorded in the commit message;
- documentation, reports, and dataset cards;
- new config or ablation files (adding a config is not an architecture change);
- results, tables, and model cards produced by runs.

**Requires lifting the freeze**, i.e. an explicit decision recorded here:

- Stage 1 block-chunking (Option B in
  [`OFFICIAL_TRAINING_PLAN.md`](OFFICIAL_TRAINING_PLAN.md)) — a scoped change to
  `build_corpus()` in `pretraining/run_mlm.py`;
- any change to initialization, expansion, or metric definitions;
- any change to the paper hyperparameters in `configs/base.yaml`.

## Verified at freeze time

Executed, not asserted:

| Check | Result |
|---|---|
| Test suite | 25 passed |
| Mean-based init | distance to centroid **exactly 0.0** |
| Init verification (independent) | verdict `global_mean`, relative error 0.0 |
| New-token firing rate | **83.0%** on held-out Tigrinya |
| Encoding gate | aborts with exit 2, creates no output |
| Original XLM-R vocab | preserved (250,002 → 250,044, append-only) |
| TIGQA end-to-end | 67 validation questions scored |
| AmQA end-to-end | 600 validation questions scored |
| MasakhaNER end-to-end | 9 labels, splits match manifest |
| Tigrinya NER end-to-end | 11 labels after `B-LO` repair |
| AfriSenti end-to-end | ran, metrics emitted |
| Stage 1 MLM | ran, loss + perplexity emitted |
| Intrinsic metrics | amh fertility 1.53 vs tir 2.31, parity 0.663 |
| Dataset checksums | 14/14 match source |

## Known limitations

Carried into training deliberately, each documented:

1. **MLM corpus provenance UNKNOWN.** HornMT ruled out on scale, overlap, and
   structure. Blocks publication, not training.
   → [MLM_CORPUS_ANALYSIS.md](MLM_CORPUS_ANALYSIS.md)
2. **MLM corpus is small** — ~2.3 M tokens after deduplication. Ten epochs risks
   memorization; held-out perplexity will be optimistic.
3. **Stage 1 padding waste.** Median line is 30–40 characters against
   `max_seq_length` 256, so ~90% of each sequence is padding under the current
   one-line-per-example construction. Fix identified, not applied (frozen).
4. **GPU paths never executed.** This is a CPU-only node. CUDA, BF16, and DDP
   code is written and reviewed but unrun; the first job should be a short
   calibration run.
5. **MLflow not installed.** The JSON fallback is exercised and working.
6. **No model cards.** No models trained yet; cards are generated from run
   artifacts.
7. **MasakhaNER is CC BY-NC 4.0.** Non-commercial; the constraint propagates to
   any NER checkpoint trained on it.
8. **Tigrinya NER has no licence.** Cannot be bundled; must be downloaded.
9. **Parity needs a parallel corpus.** Without sentence-aligned text, Table 2
   parity is computed but explicitly marked invalid.
10. **Upstream data defect.** Tigrinya NER `train.conll:1350` carries `B-LO`,
    repaired in memory at read time; the source file is untouched.
11. **Paper values unavailable for comparison.** Tables emit
    `NOT YET MEASURED` rather than estimates.

## Status

| Milestone | State |
|---|---|
| Implementation Complete | ✅ |
| Data Integrated | ✅ |
| Provenance Reviewed | ✅ |
| Licensing Reviewed | ✅ |
| Training Planned | ✅ |
| Official GPU Training | ⏳ awaiting allocation and decisions 1–4 |

**Milestone reached: Training Ready.** Not Public Release — that is gated on the
provenance and licensing items above.
