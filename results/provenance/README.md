# Provenance records for method-section numbers

Copies of small run records that live in gitignored directories, so that every number in
the paper's method and setup sections can be traced from a tracked file. Contents are
unchanged except that the home-directory path is replaced by `<home>`.

| File | Copied from | Paper use |
|---|---|---|
| `new_tokens.json` | `vocabulary_expansion/artifacts/new_tokens.json` | the 30,000 added pieces; per-language composition |
| `new_tokens_amh.json`, `new_tokens_tir.json` | `vocabulary_expansion/artifacts/` | candidate selection (top 20,000 by corpus frequency among pieces absent from XLM-R) |
| `tokenizer_manifest_amh.json`, `tokenizer_manifest_tir.json` | `tokenizer/artifacts/*/manifest.json` | SentencePiece settings and training data |
| `expansion_manifest_spm.json` | `checkpoints/vexmlm-expanded-spm/` | SentencePiece-merge integration, global-mean initialization |
| `expansion_manifest_spm_random.json` | `checkpoints/vexmlm-expanded-spm-random/` | random-initialization ablation arm |
| `stage1_summary_spm.json` | `checkpoints/vexmlm-stage1-spm/` | continued pretraining run (60 configured epochs, hardware) |
| `stage1_spm_trainer_state_best.json` | `checkpoints/vexmlm-stage1-spm/checkpoint-24808/trainer_state.json` | best checkpoint: epoch 56, step 24,808, eval loss 3.7120 |
| `preparation_report.json` | `datasets/processed/preparation_report.json` | corpus filtering and train/dev split counts |
| `stage1_spm_data.log.txt` | `logs/stage1_spm_60674.err` (data lines) and the `run_mlm.py` call in `scripts/internal/slurm_pretrain_spm.sh` | Stage 1 input files, language sampling (alpha = 0.5), 1% validation split, block counts |
| `tokenizer_metrics_2026-08-15.json` | `reports/internal/development_metrics/tokenizer_metrics.json` (earlier run of `evaluation/run_intrinsic.py`, 2026-08-15) | Glot500 row of Table 2 / Figure 1 |
| `hf_model_card_Hailay_VEXMLM.md` | Hugging Face card `Hailay/VEXMLM` (revision in `hf_model_card_Hailay_VEXMLM.revision.txt`) | parameter count and vocabulary size (Sec. 4.1) |
| `training_args.json` | selected fields of `training_args.bin` for the Stage 1 run and the seed-42 Stage 2 runs (`checkpoints/vexmlm-stage1-spm/`, `checkpoints/vexmlm-spm-stage2/*-seed42/`), read with `torch.load` | optimizer, schedule, precision, model selection |

Snapshot taken at commit `1e56e55`.
