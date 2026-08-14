# Implementation Status

What is built and verified, versus what needs data or GPU time to run.
Generated against commit at time of writing; re-check with `pytest -q`.

## Legend

| Mark | Meaning |
|---|---|
| ✅ | Implemented and verified by execution or test |
| ⏸ | Implemented; awaiting data or GPU time to execute |
| ❌ | Not implemented |

## Pipeline

| Step | Component | Status | Evidence |
|---|---|---|---|
| 1 | Amharic SentencePiece (32K) | ✅ | trained end-to-end; 94.1% Ethiopic pieces |
| 1 | Tigrinya SentencePiece (50K) | ✅ | trained end-to-end |
| 1 | Joint tokenizer with α-rebalancing | ✅ | `train_joint_tokenizer.py` |
| 2 | New-token selection | ✅ | 49/68 already-present tokens correctly dropped |
| 2 | Cross-language merge | ✅ | 3 shared tokens deduplicated |
| 2 | Vocabulary expansion | ✅ | 250,002 → 250,044 on test; originals preserved |
| 2 | Encoding gate | ✅ | aborts (exit 2) on byte-level input |
| 2 | Firing verification | ✅ | 83.02% firing rate on held-out Tigrinya |
| 3 | Mean-based init (`global_mean`) | ✅ | distance to centroid exactly 0.0 |
| 3 | `random` (Table 5 arm) | ✅ | test-asserted signature |
| 3 | `constituent_mean` | ✅ | test-asserted against subword means |
| 3 | Init verification from weights | ✅ | verdict `global_mean`, rel. error 0.0 |
| 4 | Stage 1 continued MLM | ✅ | ran end-to-end; loss + perplexity emitted |
| 5 | QA fine-tuning | ✅ | ran end-to-end; EM/F1 emitted |
| 5 | NER fine-tuning | ✅ | ran end-to-end; accuracy/macro-F1/entity-F1 |
| 5 | Sentiment fine-tuning | ✅ | ran end-to-end; accuracy/macro-F1 |

## Evaluation

| Component | Status | Evidence |
|---|---|---|
| Fertility, compression | ✅ | amh 1.53 vs tir 2.31 on parallel test text |
| Parity (validity-checked) | ✅ | 0.663; marked invalid without `--parallel` |
| OOV accuracy | ✅ | round-trip and no-UNK rates |
| Multi-seed aggregation | ✅ | mean/std/min/max/median; missing seeds warned |
| Tables 2–5 generation | ✅ | CSV + RESULTS.md; unmeasured cells labelled |

## Infrastructure

| Component | Status | Notes |
|---|---|---|
| GPU auto-detection | ✅ | CUDA → MPS → CPU; verified degrading to CPU here |
| BF16/FP16 policy | ✅ | BF16 on SM 8.0+, else FP16 |
| Multi-GPU (DDP/torchrun) | ⏸ | implemented; needs a multi-GPU node |
| Checkpoint resume | ✅ | `get_last_checkpoint` on Stage 1 and Stage 2 |
| Experiment tracking | ✅ | MLflow with JSON fallback; fallback exercised |
| Config inheritance + hashing | ✅ | `inherits:` resolution, CLI overrides |
| SLURM scripts | ✅ | parse-checked; single-GPU, multi-GPU, seed array |
| One-command pipeline | ✅ | `reproduce_paper.sh`, dry-run verified |
| Test suite | ✅ | 25 tests passing |

## Blocked on data

These are implemented but cannot produce paper numbers until corpora arrive:

| Item | Needs |
|---|---|
| Tokenizers at full 32K/50K | Amharic and Tigrinya corpora at scale |
| Stage 1 pretraining | the same corpora |
| QA results | TIGQA, AmQA |
| NER (Tigrinya) | a named Tigrinya NER corpus |
| Table 2 parity (valid) | sentence-aligned parallel text |
| All Table 4/5 numbers | the above, plus GPU time for 5 seeds × 4 arms |

See [DATA_SETUP.md](DATA_SETUP.md) and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Verification performed

Executed during development, on real data where available:

- 25 unit tests, including a regression test against the reference artifact
- full chain: corpus → tokenizers → extraction → merge → expansion → verification
- all four training pipelines run end-to-end (short runs, real data)
- dataset manager loaded 107,549 real AfriSenti rows and generated a card
- table generation exercised on real run records

No numbers in this repository are estimated or invented. Metrics with no run
behind them read `NOT YET MEASURED`.
