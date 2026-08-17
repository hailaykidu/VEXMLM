# VEXMLM — Results

Every number below is generated from a run artifact in this repository by a script
in `evaluation/`. Nothing is transcribed by hand.

**Model.** `checkpoints/vexmlm-stage1-spm` — XLM-R base extended to a 280,002
subword vocabulary (30,000 Ge'ez-script tokens merged natively into the
SentencePiece model), `global_mean` embedding initialization, then continued MLM
pretraining.

**Protocol.** Seeds 42–46, configuration hash `ce27cc194946`, one A100-PCIE-40GB,
bf16. Fine-tuning is bit-reproducible (`enable_full_determinism`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `dataloader_num_workers=0`).

---

## Tokenizer (intrinsic)

Source: [`results/spmerge_tokenizer_metrics.json`](../results/spmerge_tokenizer_metrics.json),
measured on the Amharic and Tigrinya development corpora (2,588 / 2,811 sentences).

| Metric | Language | XLM-R | VEXMLM | Change |
|---|---|---|---|---|
| Fertility (subwords/word) ↓ | amh | 2.0692 | **1.4888** | −28.0% |
| | tir | 3.1300 | **1.6928** | **−45.9%** |
| Compression (chars/token) ↑ | amh | 2.2950 | **3.1896** | +39.0% |
| | tir | 1.4591 | **2.6979** | **+84.9%** |
| Continuation rate ↓ | amh | 0.5167 | **0.3283** | −36.5% |
| | tir | 0.6805 | **0.4093** | −39.9% |
| OOV word round-trip ↑ | amh | 1.0000 | 1.0000 | — |
| | tir | 0.9954 | **0.9987** | +0.33 pt |

Added tokens carry 24.2% (amh) and 45.5% (tir) of token mass, so they are
genuinely used rather than merely present.

**Parity is not reported.** It is valid only on a sentence-aligned parallel
corpus; none is available here, and the artifact records `valid: false`. On
non-parallel text the ratio reflects content differences rather than tokenizer
fairness.

---

## Continued MLM pretraining

Source: `checkpoints/vexmlm-stage1*/stage1_summary.json`.

| Run | Tokenizer integration | Epochs | Eval loss | Perplexity |
|---|---|---|---|---|
| `vexmlm-stage1` | `add_tokens` | 10 | 4.5850 | 98.00 |
| **`vexmlm-stage1-spm`** | SentencePiece merge | 56 of 60 configured | **3.7120** | **41.67** |

The released model is the best-by-validation-loss checkpoint at epoch 56 (24,808
of 26,580 steps). The two runs differ in both tokenizer integration and training
budget, so their difference cannot be attributed to either alone.

---

## Downstream task performance

Source: [`results/downstream_task_metrics.csv`](../results/downstream_task_metrics.csv),
generated from the 30 run records in [`results/spm_stage2/`](../results/spm_stage2/)
by `evaluation/export_spm_results.py`. Mean ± standard deviation over seeds 42–46.

| Task | Dataset | Metric | VEXMLM |
|---|---|---|---|
| NER | MasakhaNER Amharic | Accuracy | 0.9413 ± 0.0026 |
| | | Macro-F1 | 0.7423 ± 0.0122 |
| | | Entity-F1 | 0.6347 ± 0.0148 |
| NER | Tigrinya NER | Accuracy | 0.9515 ± 0.0005 |
| | | Macro-F1 | 0.8219 ± 0.0069 |
| | | Entity-F1 | 0.7282 ± 0.0079 |
| QA | AmQA | EM | 32.57 ± 0.77 |
| | | F1 | 48.85 ± 0.96 |
| QA | TIGQA | EM | 2.39 ± 0.82 |
| | | F1 | 9.76 ± 0.97 |
| SA | AfriSenti (Amharic) | Accuracy | 0.4978 ± 0.0331 |
| | | Macro-F1 | 0.4971 ± 0.0193 |

### Supplementary

**TiQuAD is supplementary** — a diagnostic task, not a paper benchmark. It is
flagged as such in the CSV's `Status` column.

| Task | Dataset | Metric | VEXMLM |
|---|---|---|---|
| QA | TiQuAD *(supplementary)* | EM | 50.24 ± 0.48 |
| | | F1 | 58.90 ± 0.66 |

TIGQA's 67 test questions are too few to carry a QA claim alone; TiQuAD's 926
questions are reported alongside it for that reason.

---

## Ablation — downstream NER OOV accuracy

Source: [`results/table5_ablation.csv`](../results/table5_ablation.csv), generated
from the 20 run records in [`results/ablation/`](../results/ablation/) by
`evaluation/aggregate_ablation.py`. Tigrinya NER test split, seeds 42–46.

A word is out-of-vocabulary when the baseline `xlm-roberta-base` tokenizer emits
`<unk>`, fails to round-trip it, or fragments it into more subwords than the fixed
expanded reference tokenizer. That definition depends only on the baseline and the
reference — never on the arm being scored — so all four arms are evaluated on an
identical set: **3,491 of 4,677 word types (74.6%)**, covering 7,194 first-subword
positions.

| Configuration | OOV Acc. (%) | Δ |
|---|---|---|
| XLM-R baseline | 94.57 ± 0.16 | — |
| + VocabExp (Random Init) | 87.04 ± 0.20 | **−7.52** |
| + VocabExp (Mean Init) | 87.63 ± 0.14 | +0.59 |
| + Continued Pretraining | **95.66 ± 0.09** | **+8.02** |

**Vocabulary expansion alone hurts.** Adding 30,000 embedding rows costs 7.52
points: until Stage 1 adapts them, the new rows are noise the classifier must work
around. Mean-based initialization recovers 0.59 of that over random
initialization — a real but small effect, consistent with both variants starting
from untrained rows.

**Continued pretraining is what makes expansion pay off.** It contributes +8.02
points, recovering the expansion loss and finishing 1.09 above the baseline. The
net effect of the full method over XLM-R on this metric is **+1.09 points**, and
the mechanism is adaptation of the new embeddings rather than the vocabulary
addition itself.

---

## Scope and limitations

**Baselines are single-seed.** The XLM-R and Glot500 comparison runs under
`checkpoints/baseline-*` exist for seed 42 only. They are not aggregated into the
tables above, and no multi-seed head-to-head claim is made from them.

**A separate multilingual evaluation set** lives in
`results/multilingual_evaluation/` (AfriSenti and MasakhaNER2 across additional
languages, 5 seeds). It uses different run configurations and is never pooled with
the SP-Merge results above. Two of its sentiment configurations
(`afrisenti-swa`, `afrisenti-tso`) are degenerate majority-class collapses and are
not capability measurements.

**Not measured here.** Tokenizer parity (no sentence-aligned corpus), multi-seed
baseline comparisons, and cross-lingual transfer beyond the zero-shot Amharic
configurations in the multilingual set.

---

## Reproducing these numbers

```bash
sbatch scripts/slurm_stage2_spm_seeds.sh          # 6 tasks × 5 seeds
python3 evaluation/export_spm_results.py          # -> downstream_task_metrics.csv

sbatch scripts/slurm_ablation_tigrinya_oov.sh     # 4 arms × 5 seeds
python3 evaluation/aggregate_ablation.py          # -> table5_ablation.csv
```

See the [README](../README.md) for installation, data preparation, and the full
pipeline.
