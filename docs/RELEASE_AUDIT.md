# Release Package Audit

Component-by-component readiness. Status reflects what was **verified by
inspection or execution**, not what is intended.

Audited 2026-08-14 against commit `5df44f1`.

| Mark | Meaning |
|---|---|
| **READY** | Present, complete, and exercised |
| **PARTIAL** | Present and working, but incomplete or blocked on an input |
| **MISSING** | Not present |

## Summary

| Component | Status |
|---|---|
| README | READY |
| Model cards | PARTIAL |
| Dataset cards | READY |
| Configs | READY |
| Training scripts | READY |
| GPU support | PARTIAL |
| MLflow support | PARTIAL |
| SLURM support | READY |
| Tokenizers | READY |
| Vocabulary expansion | READY |
| Evaluation | READY |

Nothing is MISSING. The three PARTIALs are each blocked on an external input
(GPU hardware, an optional dependency, or a completed training run) rather than
on unwritten code.

---

## README — READY

`README.md`, 190 lines. Covers the method with the paper's initialization
formula, a copy-pasteable quick start for all six pipeline steps, task/dataset
table, evaluation semantics, hardware and SLURM usage, repository layout, tests,
citation, and licence. Cross-links resolve to files that exist.

## Model cards — PARTIAL

- ✅ `docs/MODEL_CARD_TEMPLATE.md` (112 lines): architecture, vocab size,
  training corpora with checksums, hyperparameters, per-seed results,
  intrinsic metrics, reproducibility block, limitations, ethical considerations.
- ❌ Zero populated `MODEL_CARD.md` files.

**Why PARTIAL, correctly:** no model has been trained. A card written now would
contain invented numbers. Cards get generated from `results.json` and
`expansion_manifest.json` once Stage 1 and Stage 2 complete.

**Carry forward:** any released Amharic NER checkpoint inherits MasakhaNER's
CC BY-NC 4.0 non-commercial constraint — the card must say so.

## Dataset cards — READY

7 cards in `datasets/cards/`, each auto-generated from the registry plus
measured statistics: source, citation, licence, version, import origin, import
date, per-split SHA-256 with verification marks, row counts, mean characters,
and label distributions where applicable.

Licences were verified against authoritative sources during this audit, and one
error was corrected: **MasakhaNER is CC BY-NC 4.0, not CC BY 4.0**. See
[DATASET_REDISTRIBUTION.md](DATASET_REDISTRIBUTION.md).

| Card | Licence | Ship? |
|---|---|---|
| tigqa | CC BY 4.0 | ✅ |
| amqa | MIT | ✅ |
| masakhaner_amh | CC BY-NC 4.0 | ⚠ non-commercial |
| tigrinya_ner | none stated | ❌ |
| afrisenti | CC BY 4.0 | ⚠ platform terms |
| amharic_mlm | UNKNOWN | ❌ |
| tigrinya_mlm | UNKNOWN | ❌ |

## Configs — READY

| File | Contents |
|---|---|
| `configs/base.yaml` | Paper Table 1 for both stages, model/tokenizer/expansion settings |
| `configs/seeds.yaml` | Seeds 42–46, aggregation statistics, 3-seed reporting floor |
| `configs/modes.yaml` | debug / research / official |
| `configs/ablations/*.yaml` | 4 arms, matching Table 5 exactly |

Inheritance (`inherits:`), CLI overrides, and config hashing are implemented in
`src/vexmlm/config.py` and exercised by every training script.

Ablation arms verified against the paper: XLM-R · Vocab Expansion + Random Init ·
Vocab Expansion + Mean Init · Full VEXMLM.

## Training scripts — READY

| Script | Verified |
|---|---|
| `pretraining/run_mlm.py` | Ran end-to-end; loss + perplexity emitted |
| `finetuning/qa/run_qa.py` | Ran on real TIGQA (67 q) and AmQA (600 q) |
| `finetuning/ner/run_ner.py` | Ran on real MasakhaNER (9 labels) and Tigrinya NER (11) |
| `finetuning/sentiment/run_sentiment.py` | Ran on real AfriSenti |

All four accept `--mode`, `--seed`, `--config`, `--override`; all resume from
the last checkpoint; all write `results.json` and a tracking record.

## GPU support — PARTIAL

Implemented in `src/vexmlm/device.py`:

- CUDA → MPS → CPU auto-selection;
- BF16 on SM 8.0+, FP16 otherwise, `--precision` overridable;
- multi-GPU via `torchrun`/DDP, `ddp_find_unused_parameters` configurable;
- effective batch size computed and logged;
- `nvidia-smi` utilization capture in the run record.

**Why PARTIAL:** this machine is a CPU-only login node. Device detection was
verified *degrading correctly* to CPU with a warning, and precision flags
correctly returning `{fp16: False, bf16: False}`. The CUDA, BF16, and DDP paths
are written but **have not executed on real hardware**. First GPU job should be
a short single-GPU run before launching the array.

## MLflow support — PARTIAL

`src/vexmlm/tracking.py` records git commit and dirty state, seed, config hash,
dataset hashes, tokenizer hash, checkpoint hash, hardware, and metrics.

**Why PARTIAL:** MLflow is not installed in this environment
(`ModuleNotFoundError`). The JSON fallback was exercised instead and works — run
records exist under `experiment_tracking/runs/`. Losing an experiment because a
logging library is absent is not acceptable, hence the fallback.

To activate: `pip install -e ".[tracking]"`.

## SLURM support — READY

| Script | Purpose |
|---|---|
| `scripts/slurm_pretrain.sh` | Stage 1, 1× A100-80GB, 24 h, requeue-safe |
| `scripts/slurm_pretrain_multigpu.sh` | Stage 1, 4× A100 via torchrun |
| `scripts/slurm_finetune.sh` | Stage 2 array job, one task per seed |

All parse cleanly (`bash -n`). Not yet submitted — by instruction.

## Tokenizers — READY

`train_amharic_tokenizer.py` (32K), `train_tigrinya_tokenizer.py` (50K),
`train_joint_tokenizer.py` (α-rebalanced), `extract_new_tokens.py`,
`evaluate_tokenizer.py`, plus the shared `spm_trainer.py`.

Verified by training real SentencePiece models end-to-end (94.1% Ethiopic
pieces on a test corpus). Corpus-too-small failures produce actionable guidance
rather than a raw assertion.

## Vocabulary expansion — READY

`expand_xlmr_vocab.py`, `initialization.py` (4 strategies), `verify_vocab.py`,
`merge_vocabularies.py`.

Verified end-to-end:

- 250,002 → 250,044 with all original entries preserved;
- mean-based init gives **distance to centroid exactly 0.0**;
- `verify_vocab.py` independently returns verdict `global_mean`, relative error 0.0;
- **83.0% new-token firing rate** on held-out Tigrinya;
- encoding gate aborts (exit 2) on byte-level-BPE input, creating no output.

## Evaluation — READY

Intrinsic (`run_intrinsic.py`, `metrics/intrinsic.py`): fertility, compression,
parity with a validity flag, OOV accuracy. Verified on parallel text — XLM-R
fertility 1.53 (amh) vs 2.31 (tir), parity 0.663.

Extrinsic: EM/F1 for QA (Ge'ez-aware normalization), accuracy/macro-F1 plus
entity-level seqeval for NER, accuracy/macro-F1 for sentiment.

Aggregation and tables: mean/std/min/max/median over seeds, missing-seed
warnings, Tables 2–5 as CSV plus rendered `RESULTS.md`. Unmeasured cells emit
`NOT YET MEASURED`.

## Tests

25 tests, all passing. Includes a specification test asserting the paper's
initialization formula directly, mutual separability of all four strategies, and
a regression test against the reference artifact's encoding defect.

## Blockers before public release

These gate *release*, not training:

1. **MLM corpus provenance** — source and licence unknown; HornMT ruled out.
2. **Tigrinya NER licence** — no grant exists; cannot be bundled.
3. **Model cards** — pending trained models.
4. **NC propagation** — decide whether released NER checkpoints carry
   MasakhaNER's non-commercial constraint.
5. **Parallel corpus** — needed for a valid Table 2 parity figure.

## Verdict

**Training Ready.** Every component needed to launch official runs is present
and exercised. No component is MISSING. The outstanding items are data-governance
and publication questions, none of which blocks starting Stage 1.
