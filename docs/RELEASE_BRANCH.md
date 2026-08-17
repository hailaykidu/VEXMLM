# Verified Release Branch

**Date**: 2026-08-16
**Status**: informational — this document records a decision. No branch, config,
result, or artifact was modified to create it.

---

## Decision

**`release/v1.0-paper` is the authoritative VEXMLM implementation.**

It is the only branch in `hailaykidu/VEXMLM` whose contents correspond to the
verified experimental artifacts. All reported VEXMLM results should be traced to
this branch.

At the time of writing it is in sync with `origin/release/v1.0-paper`
(`0` ahead, `0` behind) at commit `c66c1a5`.

---

## Repository branch inventory

| Branch | Head | Date | Contents |
|---|---|---|---|
| **`release/v1.0-paper`** | `c66c1a5` | — | **Full VEXMLM method — authoritative** |
| `master` (local only) | `9e7478f` | — | "Prepare official implementation release" |
| `origin/main` | `f2e38ed` | 2026-08-04 | Glot500 sentiment baseline |
| `origin/refactor/paper-aligned-architecture` | `23d05cf` | 2026-06-08 | EXLMR prototype + `vexmlm/` package |
| `origin/refactor/phase-2-training-evaluation` | `83e2caf` | 2026-06-08 | EXLMR prototype |

### The branches do not share history

| Branch | Root commit | Merge-base with `release/v1.0-paper` |
|---|---|---|
| `release/v1.0-paper` | `4b6cc53` | — |
| `origin/main` | `005857f` | **none** |
| `origin/refactor/paper-aligned-architecture` | `83e2caf` | **none** |
| `origin/refactor/phase-2-training-evaluation` | `83e2caf` | **none** |
| `master` | — | `5e39c92` |

`git merge-base` returns empty for all three remote branches. The repository
holds **four unrelated histories**. Only local `master` shares an ancestor with
the release branch.

**Consequence**: reconciling these branches is not a merge — it is a wholesale
replacement. Git cannot assist, and it cannot be performed without either a
force-push or an orphan-branch operation. Neither is authorized.

---

## Why the other branches are not authoritative

**`origin/main`** — tracked code is the Glot500 3-class sentiment baseline
(`finetune_M_model.py`, `evaluation.py`, `hf_space/`). Its README is written as
VEXMLM documentation, but the README itself states the artifacts are *"the
Glot500 sentiment baseline on `main`, not the VEXMLM model described in the
paper."* It further documents a `vexmlm/` package (`tokenization/tokenizer.py`,
`modeling/vexmlm.py`) that does not exist in that branch's tree, and describes
three embedding-init strategies (`mean` / `random` / `mixed`) where the shipped
model is verified `global_mean`.

**`origin/refactor/*`** — the EXLMR prototype (`EXLMR.py`, `train.csv`,
`vocab.json`), dated 2026-06-08, predating the Stage 1 / Stage 2 work by roughly
two months.

---

## What makes `release/v1.0-paper` verified

Each item below is traced to an artifact file in this branch.

| Component | Evidence |
|---|---|
| SP-Merge tokenizer, 280,002 vocab (30,000 added) | `results/spmerge_tokenizer_metrics.json` |
| Embedding init = `global_mean`, confidence high | `results/init_verification.json` — all 30,000 new rows identical to the global mean |
| Stage 1 MLM, best checkpoint epoch 56 | `checkpoints/vexmlm-stage1-spm/` — `global_step` 24,808 of 26,580 |
| Stage 2 fine-tuning, 5 seeds (42–46) | `checkpoints/vexmlm-spm-stage2/` |
| 190 multilingual evaluation runs | `results/multilingual_evaluation/` |
| Uniform hyperparameters across all arms | `training_args.bin`, config hash `2c53359ea691` |

---

## Known discrepancies — recorded, not resolved

These are documented so they are not mistaken for settled facts. **No file was
changed to reconcile them.**

1. **`reports/RESULTS_SUMMARY.md` conflicts with the measured artifacts.**
   Table 3 reports a VEXMLM OOV win; `spmerge_tokenizer_metrics.json` and
   `PAPER_TABLES_STATUS.md` disagree. Table 5's ablation arms are recorded in
   `PAPER_TABLES_STATUS.md` as never run. Table 2 parity is `valid: false` in the
   artifact (no sentence-aligned corpus). Treat verified artifacts as
   authoritative over this summary.

2. **`results/downstream_task_metrics.csv` is stale.** It holds pre-repair values
   (Amharic NER macro-F1 0.46, AmQA 20.30 EM) superseded by the SP-merge runs.

3. **`configs/base.yaml` states 10 pretraining epochs.** The released checkpoint
   is epoch 56 of a 60-epoch configuration, via
   `scripts/internal/slurm_pretrain_spm.sh` (`--override
   pretraining.num_train_epochs=60`).

4. **Eval-output drift.** Six result files across three datasets changed for
   fixed seeds within roughly one hour on 2026-08-16. Two AmQA runs
   (seeds 42, 43) were restored to their committed values. Five files remain
   uncommitted pending provenance review — including
   `datasets/metadata/afrisenti.json`, whose `config` field flipped from `tir` to
   `arq`. Cause not yet established.

---

## Constraints in force

- `release/v1.0-paper` is authoritative; do not merge or replace it.
- Do not modify `main` or any other branch.
- Never force-push. Use a normal `git push`; stop if the remote has diverged.
- Preserve the 5 uncommitted files until their provenance is verified.
- Verified artifacts and paper-reported values take precedence over unsupported
  documented results.
