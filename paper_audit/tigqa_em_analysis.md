# TIGQA EM 2.39 — what the number is, and what it can support

Compiled 2026-10-10 from stored run outputs. **No experiment was run for this note.**

## The number

`results/downstream_task_metrics.csv` reports TIGQA EM `2.39 ± 0.82` for VEXMLM. This is the mean
and sample standard deviation of the five per-seed runs in `results/spm_stage2/`:

| Seed | EM (%) | F1 (%) | Questions answered exactly |
|---|---|---|---|
| 42 | 1.492537 | 10.5695 | 1 |
| 43 | 2.985075 | 10.2176 | 2 |
| 44 | 2.985075 | 8.4166 | 2 |
| 45 | 2.985075 | 9.0652 | 2 |
| 46 | 1.492537 | 10.5533 | 1 |
| **mean** | **2.3881** | 9.7644 | 1.6 |
| **sd** | **0.8175** | | |

The published value is therefore arithmetically correct and traceable to the per-seed files.

## Why it cannot support a conclusion

**The scored split has 67 questions.** `finetuning/qa/run_qa.py:249` selects
`eval_split = "validation" if "validation" in ds else "test"`, so QA is scored on the validation
split — 67 questions for TIGQA (`datasets/metadata/tigqa.json`).

**One question is worth 1.4925 EM points** (100/67). Every observed value is an exact multiple:
1.492537% = 1 question, 2.985075% = 2 questions. The metric cannot take any value between them.

Consequently:

- The whole result rests on **one or two correct answers per run**. The five seeds split 2/3
  between "one question right" and "two questions right".
- The reported `± 0.82` is not sampling noise around a stable quantity; it is the spread between
  those two adjacent lattice points.
- No comparison at this resolution is meaningful. A difference of one question moves EM by 1.49
  points — nearly two standard deviations.

## Context

| Model | TIGQA EM | AmQA EM |
|---|---|---|
| VEXMLM (5 seeds) | 2.39 | 32.57 |
| XLM-R (seed 42) | 13.43 | 41.50 |

VEXMLM scores about 5.6× below its own baseline on TIGQA, while on AmQA — the other extractive QA
dataset, same runner, same hyperparameters — both models score an order of magnitude higher. The
gap is specific to TIGQA, not to QA in general.

F1 ≈ 9.8 against EM ≈ 2.4 indicates the model frequently produces spans that overlap the gold
answer without matching it. That is a systematic span-boundary failure rather than random error.

## What the paper should (and does) say

The paper reports the value as published and states that the cause of the QA gap is not analysed
in this work. Two further facts belong with it, and are supported by stored records alone:

1. The TIGQA evaluation split holds 67 questions, so EM resolves in steps of 1.49 points.
2. Our TIGQA copy contains only the 797 span-extractable pairs of the published release's 2,685
   (`datasets/cards/tigqa.md`), so the scores are not comparable with results on the full release.

**Recommendation.** Do not attribute the low EM to vocabulary expansion, tokenization, or any
property of VEXMLM. The measurement lacks the resolution to attribute it to anything. Report it,
state the split size, and leave the cause to future work on an adequately sized split.

## Not determined here

Why the spans are mis-bounded is **not** established. Answering it would require inspecting
per-example predictions, which is a new analysis run and was not performed. It is recorded as
future work in the Limitations section.

## Correction to an earlier reading

An earlier informal reading of these numbers used the 86-question **test** split as the
denominator and described the result as "2 of 86". The runner scores the **validation** split, so
the correct denominator is 67. The conclusion is unchanged and slightly strengthened: one question
is worth 1.49 EM points, not 1.16.
