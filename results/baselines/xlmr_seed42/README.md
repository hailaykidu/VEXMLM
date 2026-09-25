# XLM-R baseline — single seed (42)

Raw outputs of the XLM-R (`xlm-roberta-base`) downstream baseline, copied unchanged from
`checkpoints/baseline-xlmr-phase-a/<task>-seed42/` (gitignored). `SOURCE_SHA256.txt` lists
the checksums of the source `results.json` files; the copies are byte-identical. Logs are
from `logs/phase-a-xlmr-*-42.log`, with the home-directory path replaced by `<home>`.

| Task | Dataset (split) | Metric | Value |
|---|---|---|---|
| QA | AmQA (600 questions) | EM / F1 | 41.50 / 58.46 |
| QA | TigQA (67 questions) | EM / F1 | 13.43 / 27.90 |
| NER | MasakhaNER Amharic (test) | Accuracy / Macro-F1 / Entity-F1 | 0.9366 / 0.7109 / 0.6248 |
| NER | Tigrinya NER (test) | Accuracy / Macro-F1 / Entity-F1 | 0.9431 / 0.7875 / 0.6817 |
| SA | AfriSenti Amharic (test, 1,999 tweets) | Accuracy / Macro-F1 | 0.5073 / 0.5036 |

Produced by `scripts/internal/run_phase_a_xlmr_seed42.sh` (SLURM jobs 60654, 60659, 60661),
which calls `finetuning/{qa,ner,sentiment}/run_*.py --model xlm-roberta-base --seed 42
--mode official` with `configs/base.yaml` — the same runners, configuration, datasets and
splits as the VEXMLM runs in `results/spm_stage2/`.

Caveats for any comparison:
- One seed only. VEXMLM results are means over seeds 42–46.
- Run on 2026-08-15, before commit `f117c33` (2026-08-16) made fine-tuning bit-reproducible
  (`enable_full_determinism`). Re-running seed 42 now may differ slightly.
