# VEXMLM Repository Map

Audit pin: commit `1e56e55ef3f67426c2ea17e672b3a11cafef55de`, branch `fix/deterministic-evaluation`, working tree clean (verified via `git status --short` and `git rev-parse HEAD`).

**Important provenance caveat**: `.gitignore` excludes `reports/internal/`, `scripts/internal/`, `checkpoints/`, `tokenizer/artifacts/`, `vocabulary_expansion/artifacts/`, `datasets/raw/`, `datasets/processed/`, `datasets/cache/`, `legacy/`, `experiment_tracking/mlruns/`, and `experiment_tracking/runs/`. Files under these paths are **untracked working-tree artifacts, not part of git history** — `git log` returns nothing for them, so "commit" is genuinely N/A for anything under those trees (including the ~80 self-audit docs in `reports/internal/`). They are still real, present-on-disk evidence at this HEAD, but cannot be dated/attributed via git.

---

## 1. Training scripts

| Purpose | Path | Notes |
|---|---|---|
| Per-language SentencePiece tokenizer training (Tigrinya) | `tokenizer/train_tigrinya_tokenizer.py` | argparse default `--vocab-size 50000` (matches `configs/base.yaml`) |
| Per-language SentencePiece tokenizer training (Amharic) | `tokenizer/train_amharic_tokenizer.py` | argparse default `--vocab-size 32000` (matches `configs/base.yaml`) |
| Joint tokenizer utility | `tokenizer/train_joint_tokenizer.py` | |
| New-token candidate extraction | `tokenizer/extract_new_tokens.py` | Ethiopic / byte-BPE validation gates → `vocabulary_expansion/artifacts/new_tokens_{amh,tir}.json` (gitignored) |
| Tokenizer evaluation (fertility/compression/parity/OOV) | `tokenizer/evaluate_tokenizer.py` | |
| Vocabulary merge (per-language candidates → 30,000) | `vocabulary_expansion/merge_vocabularies.py` | rank-interleaved dedup+trim to `--target-total` |
| Vocab expansion of XLM-R + embedding init | `vocabulary_expansion/expand_xlmr_vocab.py` | preserves original rows 0–250,001; new pieces added only if `tok not in existing` (enforces disjointness) |
| Embedding initialization strategies | `vocabulary_expansion/initialization.py` | `global_mean_init` (paper's official Sec 3.3 method), `constituent_mean_init` (explicitly docstring-labeled "RESEARCH EXTENSION — not the paper's method"), `random_init`, `mixed_init` ("REFERENCE ONLY... do not use for results") |
| Vocab/init forensic verification | `vocabulary_expansion/verify_vocab.py` | identifies which init strategy produced a checkpoint from embedding-row statistics; writes `results/init_verification.json` |
| Continued MLM pretraining (Stage 1) | `pretraining/run_mlm.py` | reads `configs/base.yaml` `pretraining:` block; docstring: "All parameters are trainable (paper Sec. 4)" |

## 2. Fine-tuning and evaluation scripts per downstream task

| Task | Fine-tuning script | Notes |
|---|---|---|
| Sentiment Analysis (AfriSenti) | `finetuning/sentiment/run_sentiment.py` | accuracy primary metric; hardcodes `logging_steps=50`, `metric_for_best_model="accuracy"` (not sourced from `base.yaml`, which has no such key under `finetuning:`) |
| NER (MasakhaNER-Amharic, Tigrinya NER) | `finetuning/ner/run_ner.py` | seqeval entity-level F1 + accuracy + macro-F1; hardcodes `logging_steps=50`, `metric_for_best_model="macro_f1"` |
| QA (TIGQA, AmQA, TiQuAD) | `finetuning/qa/run_qa.py` | SQuAD-style EM/F1, sliding-window `doc_stride`; hardcodes `logging_steps=50` and `eval_strategy="no"` (comment: "QA eval needs custom postprocessing"), overriding `base.yaml`'s `finetuning.eval_strategy: epoch` |

No `requires_grad`/freeze code exists in any of the three `run_*.py` scripts, nor in `pretraining/run_mlm.py` (`grep -rn "requires_grad\|freeze"` returns zero hits in all four) — confirms full-model fine-tuning throughout, contradicting the paper's "partially frozen" claim.

## 3. Intrinsic evaluation scripts

| Metric | Script | Output |
|---|---|---|
| Fertility, compression, continuation rate, UNK rate, parity | `evaluation/metrics/intrinsic.py` (`measure`, `parity` functions) | consumed by `run_intrinsic.py` |
| OOV word accuracy (round-trip definition) | `evaluation/metrics/intrinsic.py::oov_accuracy` | consumed by `run_intrinsic.py` |
| Driver for Table 2/3 intrinsic metrics | `evaluation/run_intrinsic.py` | `--out results/spmerge_tokenizer_metrics.json` (or `results/tokenizer_metrics.json` per older invocations); language-agnostic (`langs = list(corpora)`, no hardcoded list); computes `parity` only when `len(langs) == 2` |
| NER-task OOV/Non-OOV accuracy split (Table 5 ablation metric) | `evaluation/run_ner_oov.py` | Tigrinya-NER-only; requires `--reference-tokenizer`; writes `results/ablation/{arm}-seed{N}.json` |
| Zero-shot sentiment transfer | `evaluation/run_zeroshot_sentiment.py` | used for AfriSenti languages lacking a train split (Oromo, Tigrinya) |
| No perplexity-specific script | — | perplexity is logged as `eval_loss`/`perplexity` inside `checkpoints/*/stage1_summary.json` (gitignored) by the HF `Trainer` inside `pretraining/run_mlm.py`, not a standalone eval script |

## 4. Config files

- `configs/base.yaml` — top-level keys: `model:`, `tokenizer:`, `vocabulary_expansion:`, `pretraining:`, `finetuning:`, `qa:`, `hardware:`, `tracking:`, `seed:`. Config hash stamped into run records as `2c53359ea691` (per `reports/internal/PAPER_PROVENANCE_MAP.md`, untracked) / `ce27cc194946` (per `reports/RESULTS_SUMMARY.md`, tracked) for different pipeline stages.
- `configs/modes.yaml` — scale/reporting modes; header states "Paper hyperparameters... are identical across all three [modes]."
- `configs/seeds.yaml` — multi-seed protocol: `seeds: [42, 43, 44, 45, 46]`; explicit comment "Single-seed results must not be reported."
- `configs/ablations/arm1_xlmr_baseline.yaml` — Table 5 arm 1: `vocabulary_expansion.enabled: false`, `pretraining.enabled: false`, `finetuning.model_path: xlm-roberta-base`.
- `configs/ablations/arm2_expansion_random.yaml` — Table 5 arm 2: `vocabulary_expansion.enabled: true`, `initialization: random`, `pretraining.enabled: false`.
- `configs/ablations/arm3_expansion_mean.yaml` — Table 5 arm 3: `vocabulary_expansion.enabled: true`, `initialization: global_mean`, `pretraining.enabled: false`. Comment: "The paper's initialization (Sec. 3.3) WITHOUT continued pretraining."
- `configs/ablations/arm4_full_vexmlm.yaml` — Table 5 arm 4 (full VEXMLM): `vocabulary_expansion.enabled: true`, `initialization: global_mean`, `pretraining.enabled: true`.

**Hardcoded-outside-config hyperparameters** (not overridden by YAML, i.e. bypass `configs/`):
- `logging_steps=50` literal in `finetuning/ner/run_ner.py`, `finetuning/qa/run_qa.py`, `finetuning/sentiment/run_sentiment.py` TrainingArguments calls (`base.yaml` has no `logging_steps` key at all).
- `metric_for_best_model` / `greater_is_better` hardcoded per-task in `run_ner.py` / `run_sentiment.py` rather than read from config.
- `run_qa.py` hardcodes `eval_strategy="no"`, overriding `base.yaml`'s `finetuning.eval_strategy: epoch`.
- AdamW `eps=1e-8, beta1=0.9, beta2=0.999` are never set anywhere in config or code (`grep -rn "adam_beta\|adam_epsilon"` across `pretraining/`, `finetuning/`, `configs/`, `src/` — zero hits); values match the paper only via unpinned HF `transformers.TrainingArguments` library defaults, not an explicit repo setting.
- Stage-1 pretraining epochs: `configs/base.yaml pretraining.num_train_epochs: 10`, but the actual released checkpoint (`vexmlm-stage1-spm`) was launched with `--override pretraining.num_train_epochs=60` via `scripts/slurm_pretrain_spm.sh` — an undocumented CLI override never reflected back into the YAML.

## 5. Data preparation scripts, dataset registry, licensing/provenance docs

| Purpose | Path |
|---|---|
| Dataset acquisition (local-first, checksummed) | `scripts/acquire_datasets.py` |
| MLM corpus normalization + train/dev split + OOV list extraction | `datasets/prepare.py` (`dev_fraction` default 0.02, seed 42 — governs only the amh/tir MLM corpora, ~98:2 train:dev, no test split) |
| Unified dataset loader/cache/hash manager | `datasets/manager.py` |
| Declarative dataset registry (source, licence, task, languages, loader) | `datasets/registry.py` — only dataset using an explicit 80:10:10-style split is **TIGQA** (797 pairs → 644/67/86, context-level split, seed 42); all others ("AmQA", "MasakhaNER-Amharic", "Tigrinya NER", "AfriSenti") use official upstream splits |
| TiQuAD normalization to SQuAD schema | `scripts/prepare_tiquad.py` |
| Dataset cards (one per dataset) | `datasets/cards/{afrisenti,amharic_mlm,amqa,masakhaner_amh,tigqa,tigrinya_mlm,tigrinya_ner}.md` |
| Per-dataset metadata (checksums, splits, licence) | `datasets/metadata/{acquisition,afrisenti,amqa,masakhaner2,masakhaner_amh,tigqa,tiquad_644,tiquad}.json` |
| Full origin/licence/checksum/provenance record | `docs/DATASET_PROVENANCE.md` — flags the amh/tir **MLM pretraining corpora provenance as UNRESOLVED** (designated source HornMT, ~2,030 pairs, cannot explain the actual ~200,000-line monolingual files — two orders of magnitude mismatch; licence UNKNOWN) |
| Redistribution/licensing verdicts per dataset | `docs/DATASET_REDISTRIBUTION.md` — TIGQA (CC BY 4.0) and AmQA (MIT) safe to ship; AfriSenti (CC BY 4.0 + Twitter ToS complication), MasakhaNER (CC BY-NC 4.0, non-commercial — corrects an earlier wrong "CC BY 4.0" registry entry), Tigrinya NER (**no licence stated — no rights granted**) must be downloaded at build time; MLM corpora flagged DO NOT SHIP |
| Practical data setup guide | `docs/DATA_SETUP.md` |
| Tigrinya NER citation | "Yohannes and Amagasa (2022)", *ACM SIGAPP Applied Computing Review* 22(3):56–68 — confirmed in `docs/DATASET_PROVENANCE.md`, `docs/DATASET_REDISTRIBUTION.md`, `datasets/registry.py`, `datasets/cards/tigrinya_ner.md`. Source: `github.com/mehari-eng/Tigrinya-NER`. |

## 6. Result files with provenance

| Result file | Producing script | Config/seeds | Commit (or N/A if gitignored) |
|---|---|---|---|
| `results/aggregate.json` | `evaluation/generate_tables.py` (writes `(out/"aggregate.json")`) | mixed runs under `checkpoints/` | `5e39c92` "v1.0-paper: Release for LM4UC Workshop, IJCAI 2026" |
| `results/downstream_task_metrics.csv` | `evaluation/export_spm_results.py` (`--csv` default) | seeds 42–46, `vexmlm-stage1-spm` model | `228dcf8` "Add authoritative SP-Merge 5-seed evaluation and regenerate downstream table" |
| `results/gpu_environment.json` | `scripts/gpu_probe.py` (pre-Stage-1 hardware gate) | — | `5e39c92` (also `af4f4b3`) |
| `results/init_verification.json` | `vocabulary_expansion/verify_vocab.py` | — | `ce87877` "Stage 1 calibration passed; launch full MLM pretraining" |
| `results/mlm_corpus_analysis.json` | `scripts/analyze_mlm_corpora.py` | — | `c66c1a5` "Correct masakhaner_amh provenance and strip local paths from metadata" |
| `results/spmerge_tokenizer_metrics.json` | `evaluation/run_intrinsic.py` (`--out` CLI arg) | amh/tir dev corpora, XLM-R + VEXMLM tokenizers | `d825163` "Prepare official implementation release" |
| `results/table5_ablation.csv` | `evaluation/aggregate_ablation.py` (**current content**; `evaluation/generate_tables.py::table5_ablation` produces an older, differently-shaped/stale version of the same filename with a different column header — do not confuse the two) | seeds 42–46, 4 arms | `bbaf6fc` "Add completed Tigrinya NER OOV ablation, 4 arms x 5 seeds" |
| `results/ablation/{arm}-seed{N}.json` (20 files: 4 arms × seeds 42–46) | `evaluation/run_ner_oov.py`, aggregated by `evaluation/aggregate_ablation.py` | seeds 42–46; arms `xlmr_baseline`, `expansion_random_init`, `expansion_mean_init`, `full_vexmlm`; all `"dataset": "tigrinya_ner", "split": "test"` | `bbaf6fc` |
| `results/spm_stage2/{task}-seed{N}.json` (30 files: tasks `afrisenti`, `amqa`, `masakhaner_amh`, `tigqa`, `tigrinya_ner`, `tiquad` × seeds 42–46) + `PROVENANCE.json` | `evaluation/export_spm_results.py` (copies from gitignored `checkpoints/`) | seeds 42–46 | `228dcf8` |
| `results/multilingual_evaluation/{sentiment,ner,sentiment_zeroshot,qa}/**/results.json` | Expanded, non-paper multilingual eval harness (`evaluation/aggregate_multilingual.py` + `scripts/slurm_multilingual_{sa,ner,qa}.sh`) | seeds 42–46 where applicable | `1a486b6` "Complete expanded multilingual evaluation; fix empty-split dataset loading" (and related `dbd4b88`, `c59fcda`, `34c642e`) — **explicitly headed "Not paper results" in `results/multilingual_evaluation/aggregated/MULTILINGUAL_RESULTS.md`** |
| `results/multilingual_evaluation/aggregated/{per_language,not_run,aggregate_averages,language_coverage}.csv`, `MULTILINGUAL_RESULTS.md` | `evaluation/aggregate_multilingual.py` | — | `1a486b6` and successors |

## 7. Table/figure-generation scripts

- `evaluation/generate_tables.py` — generates Tables 2–5 + `results/RESULTS.md` from run records under `checkpoints/`; excludes `NON_VEXMLM_DIRS = ("baseline-", "tiquad-diagnostic")` and `SPM_DIR = "vexmlm-spm-stage2"` from generic aggregation to avoid attribution errors (documented fix in `reports/internal/PAPER_TABLES_STATUS.md`, untracked). Emits `NOT YET MEASURED` for absent ablation arms rather than fabricating values.
- `evaluation/aggregate.py` — generic multi-seed aggregation module (`collect_runs`, `group_and_aggregate`; mean/std/min/max/median) consumed by `generate_tables.py`.
- `evaluation/aggregate_ablation.py` — Table 5 ablation-specific aggregator (see §6).
- `evaluation/aggregate_multilingual.py` — aggregates the expanded (non-paper) multilingual eval.
- `evaluation/export_spm_results.py` — exports authoritative SP-merge Stage 2 results (see §6).
- **No plotting/figure-generation script exists anywhere in the repository.** `grep` for `matplotlib`, `seaborn`, `plotly` across all `*.py` returns zero hits. Only two SVGs exist in the whole repo: `reports/figures/stage1_loss.svg` and `reports/figures/stage1_perplexity.svg` (both MLM training-curve figures from the original, non-SP-merge 10-epoch run — flagged stale in `reports/internal/PAPER_PROVENANCE_MAP.md`, untracked). Neither is a fertility, parity, or downstream-comparison figure. No figure supporting the paper's Figure 1 (fertility) exists in any form.
- Canonical end-to-end pipeline: `scripts/reproduce_paper.sh` (7 stages: 1 data prep, 2 tokenizer training amh/tir, 3 vocab expansion + mean init, 4 Stage 1 MLM pretraining, 5 Stage 2 fine-tuning across 5 seeds, 6 intrinsic evaluation, 7 table generation). Supports `--dry-run`, `--force`, `--stage N`, `--seeds`, `--config`; every stage is two-language (amh, tir) only — no 19-language stage exists.

## 8. Environment files and pinned library versions

Only `pyproject.toml` exists — **no `requirements.txt`, `environment.yml`, or `setup.py`** anywhere in the repo.

```
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "datasets>=2.14",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20",
    "scikit-learn>=1.3",
    "seqeval>=1.2.2",
    "pyyaml>=6.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "accelerate>=0.26",
    "safetensors>=0.4",
]
```

All bounds are lower-bound only (`>=`); there are **no exact pins and no upper bounds** for any library, including `transformers`, `torch`, `sentencepiece`, or `datasets`. No `tokenizers` or `evaluate` package is listed explicitly (transitively pulled in via `transformers`/`sentencepiece`). Optional extras: `tracking = ["mlflow>=2.9"]`, `dev = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.3"]`. Last touched at commit `4b6cc53` "Official VEXMLM implementation" (2026-08-14).

---

## Appendix: key primary sources consulted directly

- `reports/internal/CLAIMS_AUDIT.md` (untracked, dated 2026-08-15 in-document) — self-audit with SUPPORTED/PARTIALLY SUPPORTED/UNSUPPORTED verdicts on 20 claims. Predates commits `bbaf6fc`/`3d96f3b`/`228dcf8` (2026-08-17), so several of its findings (e.g. "0 of 4 ablation arms measured") describe a state since superseded by later commits — always cross-check its date against the file/commit under discussion.
- `reports/internal/PAPER_TABLES_STATUS.md` (untracked, dated 2026-08-15) — same-era caveat as above.
- `reports/internal/PAPER_PROVENANCE_MAP.md` (untracked, dated 2026-08-15) — end-to-end pipeline provenance table; also pre-dates the 2026-08-17 ablation/table fixes.
- `reports/internal/PAPER_CLAIMS_REWRITE.md` (untracked, dated 2026-08-15) — overstated-claim rewrite guide; explicitly notes "no manuscript file exists in this repository."
- `reports/internal/PAPER_METHOD_ALIGNMENT.md` (untracked, dated 2026-08-15) — states plainly: "No VEXMLM manuscript, LaTeX source, or PDF exists in this repository or the surrounding workspace."
- `reports/internal/EVALUATION_19_LANGUAGES_RECONSTRUCTION.md` (untracked, dated 2026-08-15) — finds zero 19-language artifacts at that date; superseded one day later by commit `1a486b6`'s expanded (but explicitly non-paper) 27-language multilingual evaluation.
- `reports/RESULTS_SUMMARY.md` (tracked) — current authoritative, hand-verified results summary; states "Every number below is generated from a run artifact in this repository by a script in `evaluation/`."
