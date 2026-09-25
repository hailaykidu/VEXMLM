# Reproducing the paper

*Vocabulary Expansion for Low-Resource African Languages: A Case Study in Amharic and Tigrinya.*
Every number, table and figure in `paper/` is generated from result files in this repository.
Evidence snapshot: commit `1e56e55` plus the files listed below; the corrected release is to be tagged `v1.1-correction`.

## 1. Regenerate the paper from stored results (no GPU)

```bash
python3 scripts/make_paper_artifacts.py --results results --out paper/generated
python3 -m pytest tests/test_paper_artifacts.py      # fails if paper/generated is stale
cd paper && lualatex acl_lualatex && bibtex acl_lualatex && lualatex acl_lualatex && lualatex acl_lualatex
```

`make_paper_artifacts.py` reads result files only, writes `paper/generated/results_macros.tex`
(one `\newcommand` per number, each with its source file and key in a comment), the table bodies
`table_*.tex`, and `fig_fertility.pdf`. It also checks that its aggregation reproduces
`results/downstream_task_metrics.csv` and `results/table5_ablation.csv`.

Formatting (applied only in the script): every result is given exactly as in the public artifacts —
`results/downstream_task_metrics.csv` (NER/SA accuracy and F1 as fractions, 4 decimals; QA EM/F1 in %,
2 decimals), `results/table5_ablation.csv` (%, 2 decimals, Δ as published) and
`results/spmerge_tokenizer_metrics.json` (4 decimals), which the Hugging Face model card repeats.
The single-seed XLM-R baseline uses the same unit and precision; differences are computed from
unrounded means, as in those files. `python3 paper_audit/check_public_consistency.py` compares the
paper's values with GitHub `origin/main` and the Hugging Face card.

## 2. Table and figure provenance

Every number comes from stored results of runs made before the correction (GitHub `origin/main`,
the Hugging Face card, or unchanged copies of earlier local run records in `results/provenance/` and
`results/baselines/`); `python3 paper_audit/check_result_provenance.py` verifies this. Runs made
during the correction are verification only (`paper_audit/verification/`).

| Paper element | Result file(s) | Produced by | Command |
|---|---|---|---|
| Table 1 (hyperparameters) | `results/provenance/training_args.json`, `configs/base.yaml` | `pretraining/run_mlm.py`, `finetuning/*/run_*.py` | see §3 |
| Table 2, Figure 1 (tokenizer metrics) | `results/spmerge_tokenizer_metrics.json` (XLM-R, VEXMLM; on GitHub); `results/provenance/tokenizer_metrics_2026-08-15.json` (Glot500; earlier run) | `evaluation/run_intrinsic.py` | `paper_audit/verification/tokenizer_rerun/README.md` |
| Table 3 (Tigrinya accuracy on words fragmented by XLM-R / other words) | `results/ablation/{xlmr_baseline,full_vexmlm}-seed42..46.json` | `evaluation/run_ner_oov.py` | `sbatch scripts/slurm_ablation_tigrinya_oov.sh` |
| Table 4, Table 6 (downstream), VEXMLM | `results/spm_stage2/<task>-seed42..46.json` (aggregate: `results/downstream_task_metrics.csv`) | `finetuning/{qa,ner,sentiment}/run_*.py`; `evaluation/export_spm_results.py` | `sbatch scripts/slurm_stage2_spm_seeds.sh`, then `python3 evaluation/export_spm_results.py` |
| Table 4, Table 6 (downstream), XLM-R (1 seed) | `results/baselines/xlmr_seed42/<task>/results.json` | same runners, `--model xlm-roberta-base --seed 42` | `results/baselines/xlmr_seed42/README.md` |
| Table 5 (ablation) | `results/ablation/<arm>-seed42..46.json` (aggregate: `results/table5_ablation.csv`) | `evaluation/run_ner_oov.py`; `evaluation/aggregate_ablation.py` | `sbatch scripts/slurm_ablation_tigrinya_oov.sh` |
| Sec. 3.2 (token selection, tokenizer settings) | `results/provenance/new_tokens*.json`, `tokenizer_manifest_*.json`, `expansion_manifest_spm.json` | `tokenizer/extract_new_tokens.py`, `vocabulary_expansion/merge_vocabularies.py`, `vocabulary_expansion/expand_xlmr_vocab.py` | `scripts/reproduce_paper.sh --stage 3` (see note) |
| Sec. 3.4, 4.2 (pretraining record) | `results/provenance/stage1_summary_spm.json`, `stage1_spm_trainer_state_best.json`, `stage1_spm_data.log.txt` | `pretraining/run_mlm.py` | see §3 |
| Sec. 4.1 (parameter count, vocabulary) | `results/provenance/hf_model_card_Hailay_VEXMLM.md` (Hugging Face card) | — | — |
| Sec. 4.3 (corpus and split sizes) | `results/provenance/preparation_report.json`, `datasets/metadata/*.json` | `datasets/prepare.py`, `datasets/manager.py` | `python3 datasets/prepare.py --config configs/base.yaml` |

Expected values are the macros in `paper/generated/results_macros.tex`; the test in §1 compares them.

## 3. Re-running the experiments (GPU)

The released artifacts were produced with the repository's code as it is, run with the commands
below. Each command is taken from a run record (manifest, log or job script) or confirmed by
re-running it; the verification is noted.

| Step | Command used for the released artifact | Record |
|---|---|---|
| Data | `python3 datasets/prepare.py --config configs/base.yaml` | `results/provenance/preparation_report.json` |
| Tokenizers | `python3 tokenizer/train_amharic_tokenizer.py --input datasets/processed/amharic.train.txt --output-dir tokenizer/artifacts/amharic` (Tigrinya analogous) | `results/provenance/tokenizer_manifest_*.json` (`input_files`) |
| Candidate tokens | `python3 tokenizer/extract_new_tokens.py --spm tokenizer/artifacts/amharic/amharic_sp.model --corpus datasets/processed/amharic.train.txt --max-new-tokens 20000 --out vocabulary_expansion/artifacts/new_tokens_amh.json` (Tigrinya analogous) | re-run 2026-09-25: output identical to `results/provenance/new_tokens_{amh,tir}.json` |
| Selection | `python3 vocabulary_expansion/merge_vocabularies.py --inputs vocabulary_expansion/artifacts/new_tokens_amh.json vocabulary_expansion/artifacts/new_tokens_tir.json --target-total 30000 --out vocabulary_expansion/artifacts/new_tokens.json` | re-run 2026-09-25: output identical to `results/provenance/new_tokens.json` |
| Expanded models | `python3 vocabulary_expansion/expand_xlmr_vocab.py --new-tokens vocabulary_expansion/artifacts/new_tokens.json --init global_mean --seed 42 --output checkpoints/vexmlm-expanded-spm`; the ablation arm uses `--init random` and `--output checkpoints/vexmlm-expanded-spm-random` (default `--integration sentencepiece_merge`). The manifests also record a token-firing and round-trip check on 2,000 / 300 sentences, run with `--validation-corpus`; the file passed is not recorded | `results/provenance/expansion_manifest_spm{,_random}.json` |
| Stage 1 | `python3 pretraining/run_mlm.py --config configs/base.yaml --model checkpoints/vexmlm-expanded-spm --amharic datasets/processed/amharic.train.txt --tigrinya datasets/processed/tigrinya.train.txt --block-chunk --mode official --output checkpoints/vexmlm-stage1-spm --override pretraining.num_train_epochs=60 pretraining.save_total_limit=1 pretraining.logging_steps=50` | `results/provenance/stage1_spm_data.log.txt`, `training_args.json` |
| Stage 2 (VEXMLM, 5 seeds) | `sbatch scripts/slurm_stage2_spm_seeds.sh` | `results/spm_stage2/PROVENANCE.json` |
| XLM-R baseline (seed 42) | the Stage 2 runners with `--model xlm-roberta-base --seed 42 --mode official`; a five-seed version is prepared in `scripts/slurm_xlmr_baseline_5seeds.sh` (not run) | `results/baselines/xlmr_seed42/README.md` |
| Table 5 ablation | `sbatch scripts/slurm_ablation_tigrinya_oov.sh` | `results/ablation/*.json` |
| Tokenizer metrics | command in `paper_audit/verification/tokenizer_rerun/README.md` | stored results `results/spmerge_tokenizer_metrics.json` and `results/provenance/tokenizer_metrics_2026-08-15.json`; re-run 2026-09-24: identical |
| Aggregation | `python3 evaluation/export_spm_results.py`; `python3 evaluation/aggregate_ablation.py`; `python3 scripts/make_paper_artifacts.py` | `results/downstream_task_metrics.csv`, `results/table5_ablation.csv`, `paper/generated/` |

`scripts/reproduce_paper.sh` (the one-command pipeline in the repository) issues the commands
above. Print them all without running anything with:

```bash
bash scripts/reproduce_paper.sh --dry-run
```

Until 2026-09-25 the script ran a different configuration from the released one — the raw corpus
files instead of the training splits, 15,000 candidate tokens instead of 20,000, the
`checkpoints/vexmlm-expanded` / `-stage1` names of the superseded `add_tokens` path, the
`configs/base.yaml` default of 10 epochs instead of 60, no `--block-chunk` and no `--mode
official`, no random-init ablation model, and `evaluation/generate_tables.py` for the tables. It
therefore reproduced the abandoned run while appearing to succeed. The script now issues the
recorded commands. The released artifacts are unchanged and were **not** re-run.

Three differences from the released run remain, by design:

| Step | `reproduce_paper.sh` | Released artifact |
|---|---|---|
| Output location | everything under `--run-root` (default `runs/repro/`) | `checkpoints/…`, `results/`, `logs/` |
| Stage 2 launcher | the runners in a loop over seeds | `sbatch scripts/slurm_stage2_spm_seeds.sh` (same runners, same flags) |
| Tokenizer metrics | XLM-R and the expanded model | XLM-R, Glot500 and `vexmlm-stage1-spm`, with OOV word lists |

The output location is deliberate. The script must not write into the released run's directories,
so that the experimental record of the original run stays exactly as it is: a re-run produces a
parallel set of artifacts under `runs/` (git-ignored) for comparison, and never edits, overwrites
or relabels the originals. `--run-root` refuses `checkpoints/`, `results/`, `logs/`,
`tokenizer/artifacts/`, `vocabulary_expansion/artifacts/`, `datasets/processed/` and the
repository root, and `--dry-run` writes nothing at all.

The pipeline cannot in any case be executed end to end outside the original environment: the
pretraining text is of undocumented origin and unknown licence and is not redistributed
(`docs/DATASET_PROVENANCE.md`). The script's value is as an accurate specification of what was
run, which `--dry-run` prints in full.

## 4. Data

- Downstream: AfriSenti (Hugging Face Hub, Amharic config), MasakhaNER Amharic, Tigrinya NER,
  AmQA, TIGQA — see `docs/DATA_SETUP.md` and `datasets/registry.py`. QA scores use the
  development splits; NER and SA scores use the test splits.
- Pretraining text: origin and licence undocumented (`docs/DATASET_PROVENANCE.md`); not
  redistributed.

## 5. Environment

`pyproject.toml` lists minimum versions only. Recorded for the released runs: PyTorch
2.5.1+cu118, CUDA 11.8, NVIDIA A100 (`results/provenance/stage1_summary_spm.json`). The
`transformers`, `datasets` and `sentencepiece` versions used for training were not recorded.
Paper artifacts in §1 were regenerated with Python 3, transformers 4.51.3 and matplotlib 3.11.1.
