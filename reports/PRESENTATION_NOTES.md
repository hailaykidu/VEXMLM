# VEXMLM — Presentation Notes

**Format**: 10–12 minutes
**Audience**: NLP researchers; low-resource / multilingual track
**Suggested deck**: 12–14 slides

---

## Slide-by-Slide Timing

| # | Slide | Time |
|---|---|---|
| 1 | Title | 0:30 |
| 2–3 | Problem & motivation | 2:00 |
| 4–6 | Methodology | 3:00 |
| 7 | Baselines & setup | 1:00 |
| 8 | Evaluation design | 1:00 |
| 9–10 | Results | 2:30 |
| 11 | Contributions | 1:00 |
| 12 | Limitations & future work | 1:00 |
| — | Questions | — |

---

## 1. Title (0:30)

**VEXMLM: Expanding the Lexicon of Ge'ez-Based African Languages**

Hailay Teklehaymanot · L3S Research Center

> Opening line: *"Amharic and Tigrinya have over 60 million speakers between
> them. Multilingual models claim to support them. I want to show you what that
> support actually looks like at the tokenizer level — and what happens when you
> fix it."*

---

## 2. Problem (1:00)

Ge'ez-script languages are **nominally covered, practically underserved**.

- XLM-R's vocabulary is only **~1.2% Ge'ez-script** pieces
- Tigrinya fragments into **~3.1 subwords per word** — against ~1.4 for
  well-served languages
- Amharic and Tigrinya are morphologically rich: person, tense, negation, and
  case are marked affixally, so a single word carries what English needs a
  phrase for

**Talking point**: a country name like ኢትዮጵያ (Ethiopia) should be one token.
Under a general multilingual tokenizer it becomes four fragments.

---

## 3. Why Fragmentation Matters (1:00)

Not an aesthetic concern — it has three concrete costs:

1. **Sequence length inflation** — 3× the tokens for the same text means 3× the
   compute and a third of the effective context window.
2. **Span fragmentation** — NER and extractive QA operate over token spans.
   Entities split across many pieces are harder to label consistently.
3. **Representation dilution** — meaning is spread across fragments that carry
   little individually.

**Transition**: *"So the question is whether targeted vocabulary expansion fixes
this — and whether the fix survives all the way to downstream tasks."*

---

## 4. Method Overview (1:00)

```
XLM-R (250K vocab)
      │
      ├─ Step 1  Language-specific SentencePiece    amh 32K · tir 50K
      ├─ Step 2  Vocabulary expansion               → 280K, originals preserved
      ├─ Step 3  Mean-based initialization          e_t = (1/|V_s|) Σ e_s
      ├─ Step 4  Stage 1 · continued MLM            all parameters trainable
      └─ Step 5  Stage 2 · fine-tuning              QA · NER · Sentiment
```

**Key design choice**: original vocabulary entries are **preserved** — tokens are
appended, never replaced or reordered. Pretrained embeddings stay valid, and the
model does not have to relearn what it already knows.

---

## 5. Vocabulary Construction (1:00)

- Per-language SentencePiece **Unigram** models: Amharic 32K, Tigrinya 50K
- NFC normalization; sentences filtered by minimum length and Ethiopic ratio
- Top candidates extracted per language, then **rank-interleaved** so each
  language contributes its best tokens rather than whichever file came first
- Cross-language duplicates removed — the two languages share a script and many
  subwords
- Result: **~30K new tokens**, 250K → 280K

**Talking point**: rank-interleaving matters. Concatenating would waste embedding
rows on duplicates and let one language dominate.

---

## 6. Embedding Initialization (1:00)

New rows are initialized at the centroid of the pretrained embedding space:

$$e_t = \frac{1}{|V_s|}\sum_{s \in V_s} e_s$$

**Why not random?** Random vectors sit outside the manifold the encoder expects,
and the model must first drag them into a usable region before learning anything
language-specific. Mean initialization starts them inside it.

This is a **testable design decision** — it forms an ablation arm against random
initialization.

---

## 7. Baselines and Setup (1:00)

| Baseline | Why |
|---|---|
| **XLM-R base** | the model VEXMLM extends — the essential comparison |
| **Glot500** | 500+ language coverage; tests whether broad multilinguality suffices |

**Fairness controls**: identical datasets, splits, preprocessing, metrics,
hyperparameters (single config hash), and seed. **Only the checkpoint differs.**

Seeds 42–46 for downstream results, reported as mean ± std.

**Talking point**: beating XLM-R is the bar that matters. If a vocabulary-expanded
derivative can't beat the model it was derived from, the expansion isn't earning
its cost.

---

## 8. Evaluation Design (1:00)

**Intrinsic** — does the tokenizer improve?
- Fertility (tokens/word), compression (chars/token)
- OOV word accuracy
- Parity between the two languages

**Downstream** — does it reach the tasks?
- **QA**: TIGQA (Tigrinya), AmQA (Amharic) — Exact Match, F1
- **NER**: MasakhaNER-Amharic, Tigrinya NER — entity F1, macro F1
- **Sentiment**: AfriSenti-Amharic — accuracy

**Methodological note worth stating aloud**: parity is only reported as valid on
a **sentence-aligned parallel corpus**. On unaligned text the ratio reflects
content differences, not tokenizer fairness. The tooling refuses to print a
misleading number.

---

## 9–10. Results (2:30)

> Populate from `RESULTS_SUMMARY.md`.

Structure the results narrative in three beats:

**Beat 1 — the tokenizer improves substantially.** Lead with Tigrinya fertility;
it is the largest and most intuitive effect. Ge'ez vocabulary coverage rises from
~1.2% to ~11.8%.

**Beat 2 — improvements reach the downstream tasks.** Present QA, NER, and
sentiment against both baselines.

**Beat 3 — where it does and doesn't hold.** Be explicit about which comparisons
are decisive and which are within variance. An audience trusts a presenter who
marks their own error bars.

**Delivery advice**: one number per slide gets remembered. Fertility
3.1 → 1.7 is the number people will repeat.

---

## 11. Contributions (1:00)

1. **Vocabulary extension** for Ge'ez-script languages that preserves the base
   model's pretrained vocabulary entirely.
2. **Mean-based embedding initialization** placing new tokens inside the
   pretrained manifold.
3. **Language-specific tokenization** with principled cross-language merging.
4. **Two-stage training** — continued MLM followed by task fine-tuning.
5. **Intrinsic and downstream evaluation** across five tasks and five seeds, with
   an open, reproducible pipeline.

---

## 12. Limitations (1:00)

State these directly — they strengthen the talk.

- **Two languages.** Amharic and Tigrinya. Generalization to other Ge'ez-script
  languages (Tigre, Ge'ez) or other scripts is untested.
- **Pretraining corpus size.** The continued-pretraining corpus is small by
  MLM standards, which bounds how well the added vocabulary can be learned.
- **Parity requires aligned data.** Not reported without a sentence-aligned
  parallel corpus — an honest omission rather than an estimated number.
- **Dataset licensing.** MasakhaNER-Amharic is CC BY-NC; models fine-tuned on it
  inherit a non-commercial restriction.
- **TIGQA is small.** 67 evaluation questions means one question is worth ~1.5
  EM points — the benchmark cannot resolve small differences.

**Future work**: larger pretraining corpora; additional Ge'ez-script languages;
initialization strategies beyond the global mean.

---

## Anticipated Questions

**Q — Why not just use a model with better coverage, like Glot500?**
Broad coverage is not the same as good coverage. Glot500 is a baseline here
precisely to test that; its Tigrinya fertility is still well above VEXMLM's.

**Q — Why 30K new tokens?**
15K candidates per language after deduplication, chosen to reach a 280K target —
a ~12% vocabulary increase, which keeps the embedding matrix growth modest
relative to the tokenization gain.

**Q — Does expansion hurt the other languages XLM-R covers?**
Original entries are preserved, so nothing is displaced. Measuring cross-lingual
effects directly would require monolingual corpora for those languages — future
work.

**Q — Is the improvement just from more training?**
Stage 1 details are in the paper. The intrinsic tokenizer gains are independent
of training entirely — they hold before any gradient step.

**Q — Can I reproduce this?**
Yes. `scripts/reproduce_paper.sh` runs the full pipeline end to end. Datasets
aren't redistributed for licensing reasons, but the registry documents every
source.

---

## Presentation Tips

- **Lead with the tokenizer example.** ኢትዮጵያ as four fragments versus one token
  lands immediately, even with a non-Ge'ez-reading audience.
- **One number per results slide.**
- **State limitations without hedging.** "TIGQA has 67 questions, so we can't
  resolve differences under about eight points" reads as rigour.
- **Have the fertility table ready** as a backup slide — it's the most likely
  follow-up.
