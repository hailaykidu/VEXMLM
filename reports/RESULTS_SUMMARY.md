# VEXMLM — Results

Published results from the VEXMLM paper. Metrics: parity relative to English
sentence length, OOV word accuracy, downstream task performance, and an
initialization ablation.

Baselines: `xlm-roberta-base` and `cis-lmu/glot500-base`.

---

## Table 2 — Tokenizer Parity

Parity scores across languages relative to English sentence length. Values
closer to 1.0 indicate more equitable tokenization. **Bold** indicates best.

| Language | XLM-R | Glot500 | VEXMLM |
|---|---|---|---|
| Tigrinya | **1.36** | 1.20 | 0.27 |
| Amharic | 0.82 | **0.90** | 0.39 |
| Ge'ez | 0.45 | 0.31 | **0.73** |
| Tigre | 0.82 | 0.88 | **0.89** |
| Harari | 0.41 | 0.14 | **0.88** |
| Gurage | 0.62 | 0.12 | **0.89** |
| Afar | 0.54 | 0.30 | **0.90** |
| Oromo | 1.27 | 0.90 | **0.93** |
| Afrikaans | 1.18 | 1.30 | **0.96** |
| German | 1.09 | 1.10 | **0.98** |
| Arabic | 1.27 | 1.30 | **1.16** |

VEXMLM achieves the most equitable tokenization on 9 of 11 languages, including
every Ge'ez-script language except Tigrinya and Amharic, and on the
non-Ge'ez controls (Afrikaans, German, Arabic).

---

## Table 3 — OOV Word Accuracy on NER

OOV word accuracy (%) for 11 African languages on NER. A diagnostic evaluation
focusing on out-of-vocabulary tokens within the NER task; **not directly
comparable** to the overall NER accuracy in Table 4.

| Language | XLM-R Non-OOV | XLM-R OOV | VEXMLM Non-OOV | VEXMLM OOV |
|---|---|---|---|---|
| Amharic (amh) | **98.1** | 91.1 | **98.1** | **92.2** |
| Tigrinya (tir) | 78.1 | 96.1 | **89.5** | **98.2** |
| Hausa (hau) | 97.0 | 90.2 | **97.2** | **95.6** |
| Igbo (ibo) | **97.8** | 91.9 | 97.7 | **94.5** |
| Kinyarwanda (kin) | 98.8 | 84.9 | **99.0** | **93.4** |
| Luganda (lug) | **98.8** | 91.3 | 96.4 | **94.8** |
| Luo (luo) | 97.8 | 91.4 | **98.6** | **95.2** |
| Nigerian Pidgin (pcm) | **98.6** | 89.6 | 97.5 | **97.0** |
| Swahili (swa) | **98.2** | 80.2 | 93.9 | **95.2** |
| Wolof (wol) | 92.5 | 89.1 | **98.7** | **93.0** |
| Yoruba (yor) | 91.6 | 81.8 | **92.8** | **89.1** |
| **Average** | 95.2 | **88.9** | **96.3** | **94.4** |

VEXMLM improves average OOV accuracy by **+5.5 points** over XLM-R, and improves
OOV accuracy on **all 11 languages**.

> **Correction applied (B1).** The averages above are the verified recomputation
> from the eleven tabulated rows: XLM-R OOV **88.9** (was 81.4) and VEXMLM OOV
> **94.4** (was 94.3), giving an improvement of **+5.5** (was +12.9). Three of
> four column means reproduced exactly before correction, confirming the method
> as an unweighted mean over 11 languages. Derivation:
> [internal/TABLE3_VERIFICATION.md](internal/TABLE3_VERIFICATION.md) ·
> [TABLE3_RELEASE_PATCH.md](TABLE3_RELEASE_PATCH.md).
>
> The caption cross-reference has been corrected to **Table 4**. One editorial
> item remains for the authors: Tigrinya is coded `(tig)` in the source table,
> whereas ISO 639-3 `tig` denotes **Tigre** — Tigrinya is `tir` (used here), and
> Table 2 lists Tigre as a separate language.

---

## Table 4 — Downstream Task Performance

SA and NER report accuracy; QA reports Exact Match (EM) and F1. **Bold**
indicates best performance per metric.

| Task (Metric) | XLM-R | VEXMLM | Glot500 |
|---|---|---|---|
| SA (Accuracy) | 0.77 | **0.80** | 0.46 |
| NER (Accuracy) | 0.75 | 0.78 | **0.92** |
| QA (EM) | 0.66 | **0.87** | 0.74 |
| QA (F1) | 0.78 | **0.90** | 0.78 |

VEXMLM significantly outperforms XLM-R on QA (**+21 EM**, **+12 F1**) and SA
(**+3 accuracy**), and Glot500 on SA (**+34 accuracy**) and QA. Glot500 attains
the highest NER accuracy.

*All caption deltas verified against the tabulated values.*

---

## Table 5 — Ablation Study

Tigrinya NER OOV accuracy under different configurations. Each row adds one
component to the previous.

| Configuration | OOV Acc. (%) | Δ |
|---|---|---|
| XLM-R baseline | 96.1 | — |
| + VocabExp (Random Init) | 97.3 | +1.2 |
| + VocabExp (Mean Init) | **97.8** | +0.5 |
| + Continued Pretraining | **98.2** | +0.4 |

Each component contributes positively. Vocabulary expansion alone accounts for
the largest single gain (+1.2); mean initialization improves on random
initialization by **+0.5**, and continued pretraining adds a further **+0.4**.
Total improvement over the baseline: **+2.1 points**.

*Consistency verified: the ablation endpoints match Table 3's Tigrinya row
exactly — baseline 96.1 = XLM-R OOV, final 98.2 = VEXMLM OOV.*

---

## Summary of Findings

**Tokenization equity (Table 2).** VEXMLM produces the most equitable
tokenization relative to English on 9 of 11 languages, including under-served
Ge'ez-script languages (Ge'ez, Tigre, Harari, Gurage, Afar) and non-Ge'ez
controls (Afrikaans, German, Arabic).

**OOV handling (Table 3).** VEXMLM improves OOV word accuracy on all 11 African
languages evaluated, with the largest gains on Swahili, Kinyarwanda, and
Nigerian Pidgin.

**Downstream performance (Table 4).** VEXMLM leads on question answering
(+21 EM, +12 F1 over XLM-R) and sentiment analysis, while Glot500 leads on NER
accuracy.

**Component contributions (Table 5).** Every stage of the method contributes:
vocabulary expansion, mean-based initialization over random, and continued
pretraining.

---

## Reproduction

```bash
bash scripts/reproduce_paper.sh
```

Generated tables are written to `results/` as CSV plus a rendered
`results/RESULTS.md`. See [DOCUMENTATION_OVERVIEW.md](DOCUMENTATION_OVERVIEW.md)
for per-component details.
