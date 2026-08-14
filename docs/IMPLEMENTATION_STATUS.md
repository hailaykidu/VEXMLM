# Implementation Status

> **Status: TRAINING READY** — implementation frozen at commit `3298cf3`
> (see [IMPLEMENTATION_FREEZE.md](IMPLEMENTATION_FREEZE.md)).
> Datasets integrated and licence-reviewed; official GPU runs awaiting
> allocation and the decisions in [OFFICIAL_TRAINING_PLAN.md](OFFICIAL_TRAINING_PLAN.md).

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

## Data status

All designated datasets are integrated from local copies, checksum-verified, and
documented. See [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md).

| Dataset | Integrated | Verified end-to-end |
|---|---|---|
| TIGQA | ✅ 797 QA pairs | ✅ 67 val questions scored |
| AmQA | ✅ 1,723/600/299 | ✅ 600 val questions scored |
| MasakhaNER (amh) | ✅ 1,750/250/500 | ✅ 9 labels |
| Tigrinya NER | ✅ 4,562/570/571 | ✅ 11 labels after repair |
| AfriSenti | ✅ Hub loader | ✅ ran |
| MLM corpora | ✅ 200K lines each | ✅ Stage 1 ran |

## Remaining blockers

| Item | Blocks | Reference |
|---|---|---|
| MLM corpus provenance UNKNOWN | publication, not training | [MLM_CORPUS_ANALYSIS.md](MLM_CORPUS_ANALYSIS.md) |
| Tigrinya NER has no licence | bundling | [DATASET_REDISTRIBUTION.md](DATASET_REDISTRIBUTION.md) |
| MasakhaNER CC BY-NC 4.0 | commercial use of NER models | [DATASET_REDISTRIBUTION.md](DATASET_REDISTRIBUTION.md) |
| Parallel corpus absent | valid Table 2 parity | [DATA_SETUP.md](DATA_SETUP.md) |
| GPU allocation | all official runs | [OFFICIAL_TRAINING_PLAN.md](OFFICIAL_TRAINING_PLAN.md) |

See [RELEASE_AUDIT.md](RELEASE_AUDIT.md) for component-level readiness.

## Verification performed

Executed during development, on real data where available:

- 25 unit tests, including a regression test against the reference artifact
- full chain: corpus → tokenizers → extraction → merge → expansion → verification
- all four training pipelines run end-to-end (short runs, real data)
- dataset manager loaded 107,549 real AfriSenti rows and generated a card
- table generation exercised on real run records

No numbers in this repository are estimated or invented. Metrics with no run
behind them read `NOT YET MEASURED`.
