# Table 3 — Release Correction Package

**Date**: 2026-08-15
**Paper**: *Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya* (LM4UC Workshop, IJCAI 2026)
**Status**: ⚠️ **PREPARED — NOT APPLIED.** Author review required.

---

## Edit 1 — Average Row and Headline Claim

### Current

```
Average
95.2 | 81.4 | 96.3 | 94.3

Claim:
+12.9
```

### Recommended

```
Average
95.2 | 88.9 | 96.3 | 94.4

Claim:
+5.5
```

### Cell detail

| Column | Current | Recommended | Δ | Basis |
|---|---|---|---|---|
| XLM-R Non-OOV | 95.2 | **95.2** | — | recomputes to 95.2091 ✅ |
| **XLM-R OOV** | **81.4** | **88.9** | **+7.5** | recomputes to 88.8727 |
| VEXMLM Non-OOV | 96.3 | **96.3** | — | recomputes to 96.3091 ✅ |
| **VEXMLM OOV** | **94.3** | **94.4** | **+0.1** | recomputes to 94.3818 |
| **Improvement** | **+12.9** | **+5.5** | **−7.4** | 94.3818 − 88.8727 = 5.5091 |

**Row values are unchanged.** Only the summary row and the derived claim.

### Verified arithmetic

```python
xlmr_oov   = [91.1, 96.1, 90.2, 91.9, 84.9, 91.3, 91.4, 89.6, 80.2, 89.1, 81.8]
vexmlm_oov = [92.2, 98.2, 95.6, 94.5, 93.4, 94.8, 95.2, 97.0, 95.2, 93.0, 89.1]

sum(xlmr_oov)   / 11    # 88.8727…  → 88.9
sum(vexmlm_oov) / 11    # 94.3818…  → 94.4
94.3818 - 88.8727       # +5.5091   → +5.5
```

Three of four averages reproduce exactly from the published rows, confirming the
method is an unweighted mean over 11 languages.

### Caption sentence

> **Current**: "VEXMLM improves average OOV accuracy by +12.9 points over XLM-R."
>
> **Recommended**: "VEXMLM improves average OOV accuracy by +5.5 points over
> XLM-R, and improves OOV accuracy on all 11 languages evaluated."

The added clause preserves the finding's strength — improvement on *every*
language — while stating the magnitude accurately.

---

## Edit 2 — Caption Cross-Reference

### Current

> "...not directly comparable to the overall NER accuracy reported in **Table 3**"

### Recommended

> "...not directly comparable to the overall NER accuracy reported in **Table 4**"

**Rationale**: the sentence appears in Table 3's own caption, making the
reference self-referential. Overall NER accuracy is in Table 4 (XLM-R 0.75,
VEXMLM 0.78, Glot500 0.92).

---

## Corrected Table (for review)

| Language | XLM-R Non-OOV | XLM-R OOV | VEXMLM Non-OOV | VEXMLM OOV |
|---|---|---|---|---|
| Amharic (amh) | 98.1 | 91.1 | 98.1 | 92.2 |
| Tigrinya (tir) | 78.1 | 96.1 | 89.5 | 98.2 |
| Hausa (hau) | 97.0 | 90.2 | 97.2 | 95.6 |
| Igbo (ibo) | 97.8 | 91.9 | 97.7 | 94.5 |
| Kinyarwanda (kin) | 98.8 | 84.9 | 99.0 | 93.4 |
| Luganda (lug) | 98.8 | 91.3 | 96.4 | 94.8 |
| Luo (luo) | 97.8 | 91.4 | 98.6 | 95.2 |
| Nigerian Pidgin (pcm) | 98.6 | 89.6 | 97.5 | 97.0 |
| Swahili (swa) | 98.2 | 80.2 | 93.9 | 95.2 |
| Wolof (wol) | 92.5 | 89.1 | 98.7 | 93.0 |
| Yoruba (yor) | 91.6 | 81.8 | 92.8 | 89.1 |
| **Average** | **95.2** | **88.9** ← | **96.3** | **94.4** ← |

---

## ⚠️ Precondition

This package assumes the **row values are authoritative**.

If the **81.4 average** is authoritative instead, the patch is wrong: the XLM-R
OOV rows must be re-derived from source logs, and **at least 8 of 11 would
change** (the column sum must fall by 82.2 points; no single-row correction is
possible).

**Confirm which source is authoritative before applying.**

---

## Author Checklist

- ⬜ Re-derive the XLM-R OOV column from source experiment logs
- ⬜ Confirm rows (not the average) are authoritative
- ⬜ Apply Edit 1: `81.4 → 88.9`, `94.3 → 94.4`, `+12.9 → +5.5`
- ⬜ Apply Edit 2: caption `Table 3 → Table 4`
- ⬜ Search abstract, introduction, and conclusion for other `+12.9` references
- ⬜ Re-verify all four column averages after editing
- ⬜ Consider `(tig)` → `(tir)` — ISO 639-3 `tig` is Tigre; Table 2 lists Tigre separately

---

## Impact

| Aspect | Assessment |
|---|---|
| Paper's conclusion | **Unaffected** — VEXMLM improves OOV accuracy on all 11 languages either way |
| Headline magnitude | **Reduced** — +12.9 → +5.5 |
| Tables 2, 4, 5 | **Unaffected** — all verify independently |
| Per-language findings | **Unaffected** — Swahili +15.0, Kinyarwanda +8.5, Pidgin +7.4 |
| Reviewer-detectable | **Yes** — arithmetic on the printed table, under a minute |

**Recommendation**: apply before camera-ready. If the paper is already
published, escalate to an erratum — both the average and the headline claim are
affected.

Supporting analysis: [internal/TABLE3_VERIFICATION.md](internal/TABLE3_VERIFICATION.md) ·
[internal/TABLE3_CORRECTION_PROPOSAL.md](internal/TABLE3_CORRECTION_PROPOSAL.md)
