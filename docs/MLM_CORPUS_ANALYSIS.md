# MLM Corpus Analysis

Origin investigation for the two Stage 1 pretraining corpora. Every figure below
is measured by [`scripts/analyze_mlm_corpora.py`](../scripts/analyze_mlm_corpora.py);
raw output in [`results/mlm_corpus_analysis.json`](../results/mlm_corpus_analysis.json).

```bash
python scripts/analyze_mlm_corpora.py --out results/mlm_corpus_analysis.json
```

Classification scheme:

| Label | Meaning |
|---|---|
| **VERIFIED** | A checksum or exact content match ties the file to a named source. |
| **LIKELY** | Strong distinguishing evidence, short of an exact match. |
| **UNKNOWN** | No sufficient evidence. No guess is recorded. |

---

## Verdict

| Corpus | Classification | HornMT? |
|---|---|---|
| Amharic | **UNKNOWN** | Ruled out |
| Tigrinya | **UNKNOWN** | Ruled out |

Neither corpus is labelled HornMT anywhere in this repository. Neither is
assigned a speculative origin.

## Corpus facts

| Field | Amharic | Tigrinya |
|---|---|---|
| Import path | `datasets/raw/amharic/amharic.txt` | `datasets/raw/tigrinya/tigrinya.txt` |
| Origin path (read-only) | `~/lgse-repro/data/lapt/amharic.txt` | `~/lgse-repro/data/lapt/tigrinya.txt` |
| Language | Amharic (amh) | Tigrinya (tir) |
| Bytes | 23,338,664 | 18,026,579 |
| SHA-256 (prefix) | `4d16a0e63d3a0f01…` see JSON | see JSON |
| Lines (total) | 200,002 | 200,000 |
| Lines (non-empty) | 200,001 | 200,000 |
| Lines (unique) | 129,864 | 141,015 |
| **Duplicate rate** | **35.07%** | **29.49%** |
| Tokens (whitespace) | 1,902,258 | 1,531,950 |
| Characters | 9,091,793 | 6,982,723 |
| Mean tokens/line | 9.51 | 7.66 |
| Ge'ez share (non-space) | 94.3% | 96.0% |
| Distinct Ethiopic codepoints | 318 | 345 |

## Domain composition

Share of sampled lines containing at least one marker term. Markers are coarse
probes: they establish that material of a kind is **present**, not that it
dominates. Categories overlap and do not sum to 100%.

| Domain | Amharic | Tigrinya |
|---|---|---|
| Religious (Christian) | 7.1% | 8.9% |
| Religious (Islamic) | 2.6% | 0.1% |
| News / politics | 0.8% | 0.9% |
| Web / commerce | 1.5% | 0.4% |
| Unmatched by any probe | ~88% | ~90% |

Qualitatively, manual sampling shows Qur'anic and Biblical translation passages
alongside general web prose — a mixture typical of a broad crawl rather than a
curated single-domain collection.

## Why HornMT is ruled out

The designated reference source was
<https://github.com/asmelashteka/HornMT>. Three independent lines of evidence
exclude it:

**1. Scale.** The HornMT copies present on this machine hold **2,030 unique
lines**. Each corpus here holds **200,000** — 99× larger. A 2K-pair parallel
corpus cannot be the source of a 200K-line monolingual file.

| HornMT copy on this machine | Unique lines |
|---|---|
| `~/amseg/data/corpus/hornmt_tir.txt` | 2,029 |
| `~/TigrinyaTokenizer/EnTiMT/01_collection/raw/hornmt.ti` | 2,029 |
| `~/TigrinyaTokenizer/EnTiMT/01_collection/raw/hornmt.en` | 2,030 |

**2. Zero content overlap.** Exact line intersection between each corpus and
every HornMT copy above: **0 lines (0.00%)**. Had these corpora been built from
HornMT — even as a subset, even after shuffling — some exact lines would
survive. None do.

**3. Structure.** HornMT is a *parallel* MT corpus with aligned segments across
languages. These two files are monolingual, independently sized (200,001 vs
200,000 lines), and have unequal token counts. The round 200,000 figure in both
indicates a deliberate cap applied during preparation, not a natural boundary.

## Why the true origin is UNKNOWN

The evidence rules a source *out* without establishing one. What is missing:

- no manifest, README, or preparation script accompanies these two files in the
  originating workspace — unlike every other dataset there, which carries a
  `manifest.json` recording source and checksums;
- no checksum match against any corpus available on this machine;
- content is consistent with a CC-100 / OSCAR-style crawl (religious
  translations plus general web text, the usual composition for Amharic and
  Tigrinya in those collections), but **consistency is not identification** —
  many crawls would look similar, and no exact match was found.

Recording "probably CC-100" would be a guess. It stays **UNKNOWN**.

## Licence evidence

| Corpus | Licence evidence found |
|---|---|
| Amharic | **None.** No licence file, header, or statement accompanies the file. |
| Tigrinya | **None.** Same. |

Both are marked `license: UNKNOWN — blocker` in
[`datasets/registry.py`](../datasets/registry.py).

## Consequences

1. **Neither corpus may be redistributed.** Without a known source, its licence
   cannot be determined, so it cannot ship with the repository or a model.
2. **Models trained on them cannot carry a complete data statement.** The corpus
   row of every model card stays `UNKNOWN` until this is resolved.
3. **Stage 1 experiments on this machine are unaffected.** Local use for
   training and internal evaluation is fine; only redistribution and publication
   claims are blocked.
4. **Deduplication is required before use** — see the 35%/29% duplicate rates in
   [`reports/mlm_data_quality_report.md`](../reports/mlm_data_quality_report.md).

## What would resolve this

Any one of the following, in descending order of strength:

1. The preparation script or command that produced these two files.
2. A named source plus version (e.g. "CC-100 amh, 2020 release, first 200K lines
   after filtering"), which can then be checksum-verified against the public
   release.
3. Author confirmation of the source, recorded here with the date and the person
   confirming.

If HornMT genuinely was intended, the corpora must be rebuilt from it — and at
~2,030 sentence pairs, that will not support ten epochs of continued pretraining
in any meaningful sense. That trade-off should be settled before Stage 1 runs.
