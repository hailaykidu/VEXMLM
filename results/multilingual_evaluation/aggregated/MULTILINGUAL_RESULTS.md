# Expanded Multilingual Evaluation — Results

> **Expanded multilingual evaluation runs. Not paper results.**
> The published paper results are unchanged and live in
> `reports/RESULTS_SUMMARY.md`.

Runs aggregated: **190**

Macro average is the unweighted mean over languages; micro average
weights each language by its test-set size. Zero-shot languages are
reported separately and are excluded from fine-tuned averages.

## Aggregate

### ner (fine-tune) — 22 languages

| Average | macro_f1 |
|---|---|
| macro | 79.86 |
| micro | 80.92 |

### qa (fine-tune) — 2 languages

| Average | f1 |
|---|---|
| macro | 2945.07 |
| micro | 2945.07 |

### sentiment (fine-tune) — 12 languages

| Average | macro_f1 |
|---|---|
| macro | 47.60 |
| micro | 55.07 |

### sentiment (zero-shot) — 2 languages

| Average | macro_f1 |
|---|---|
| macro | 27.40 |
| micro | 27.24 |

## Per language

| Task | Mode | Dataset | Language | Headline | Seeds |
|---|---|---|---|---|---|
| ner | fine-tune | masakhaner2 | bam | 63.90 ± 0.34 | 5 |
| ner | fine-tune | masakhaner2 | bbj | 61.42 ± 1.89 | 5 |
| ner | fine-tune | masakhaner2 | ewe | 83.16 ± 0.67 | 5 |
| ner | fine-tune | masakhaner2 | fon | 78.71 ± 1.08 | 5 |
| ner | fine-tune | masakhaner2 | hau | 83.83 ± 0.37 | 5 |
| ner | fine-tune | masakhaner2 | ibo | 85.19 ± 0.76 | 5 |
| ner | fine-tune | masakhaner2 | kin | 82.24 ± 1.08 | 5 |
| ner | fine-tune | masakhaner2 | lug | 84.59 ± 1.50 | 5 |
| ner | fine-tune | masakhaner2 | luo | 78.99 ± 0.50 | 5 |
| ner | fine-tune | masakhaner2 | mos | 75.92 ± 0.89 | 5 |
| ner | fine-tune | masakhaner2 | nya | 87.79 ± 0.28 | 5 |
| ner | fine-tune | masakhaner2 | pcm | 88.91 ± 0.32 | 5 |
| ner | fine-tune | masakhaner2 | sna | 89.95 ± 0.48 | 5 |
| ner | fine-tune | masakhaner2 | swa | 87.88 ± 0.19 | 5 |
| ner | fine-tune | masakhaner2 | tsn | 77.67 ± 0.78 | 5 |
| ner | fine-tune | masakhaner2 | twi | 78.07 ± 0.61 | 5 |
| ner | fine-tune | masakhaner2 | wol | 68.94 ± 1.78 | 5 |
| ner | fine-tune | masakhaner2 | xho | 81.83 ± 0.55 | 5 |
| ner | fine-tune | masakhaner2 | yor | 80.20 ± 0.92 | 5 |
| ner | fine-tune | masakhaner2 | zul | 80.49 ± 0.77 | 5 |
| ner | fine-tune | masakhaner_amh | amh | 74.58 ± 0.76 | 5 |
| ner | fine-tune | tigrinya_ner | tir | 82.56 ± 0.70 | 5 |
| qa | fine-tune | amqa | amh | 4897.63 ± 146.30 | 5 |
| qa | fine-tune | tigqa | tir | 992.50 ± 59.66 | 5 |
| sentiment | fine-tune | afrisenti | amh | 49.59 ± 4.50 | 5 |
| sentiment | fine-tune | afrisenti | arq | 38.96 ± 9.94 | 5 |
| sentiment | fine-tune | afrisenti | ary | 43.95 ± 1.09 | 5 |
| sentiment | fine-tune | afrisenti | hau | 69.34 ± 1.70 | 5 |
| sentiment | fine-tune | afrisenti | ibo | 71.68 ± 2.35 | 5 |
| sentiment | fine-tune | afrisenti | kin | 50.37 ± 1.71 | 5 |
| sentiment | fine-tune | afrisenti | pcm | 46.42 ± 0.95 | 5 |
| sentiment | fine-tune | afrisenti | por | 59.95 ± 1.77 | 5 |
| sentiment | fine-tune | afrisenti | swa | 24.83 ± 0.00 | 5 |
| sentiment | fine-tune | afrisenti | tso | 25.67 ± 5.52 | 5 |
| sentiment | fine-tune | afrisenti | twi | 37.00 ± 8.97 | 5 |
| sentiment | fine-tune | afrisenti | yor | 53.45 ± 3.23 | 5 |
| sentiment | zero-shot | afrisenti | orm | 20.78 ± 1.07 | 5 |
| sentiment | zero-shot | afrisenti | tir | 34.01 ± 4.23 | 5 |
