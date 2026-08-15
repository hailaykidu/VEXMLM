# License Final Status

**Date**: 2026-08-15
**Scope**: pretraining corpora licence classification for public release.

---

## HornMT — Verified

| Field | Value |
|---|---|
| Name | HornMT — *Machine Translation Benchmark Dataset for Languages in the Horn of Africa* |
| Repository | `github.com/asmelashteka/HornMT` |
| **Licence** | **CC BY 4.0** — declared in the README |
| Size | **2,030 records** (`HornMT.json`) |
| Type | Parallel (multi-way translation benchmark) |

**Verified quotation**: *"This work is licensed under a Creative Commons
Attribution 4.0 International License."*

⚠️ GitHub's API reports `license: null` and `/license` returns 404, because the
terms are stated in prose rather than a `LICENSE` file. **The README is
authoritative.** Anyone auditing via the API alone will incorrectly conclude the
repository is unlicensed.

**HornMT itself is redistributable with attribution.**

---

## The Scale Discrepancy

| Source | Size | Type |
|---|---|---|
| HornMT | **2,030 records** | parallel |
| VEXMLM Amharic corpus | **200,000 lines** | monolingual |
| VEXMLM Tigrinya corpus | **200,000 lines** | monolingual |

HornMT accounts for approximately **1%** of each corpus — a **98.5× gap**.

`datasets/registry.py` records the same finding independently: content sampling
shows religious translations plus general web text, consistent with a
CC-100/OSCAR-style crawl.

**HornMT is a confirmed and permissively licensed component, but not the origin
of the corpora as a whole.**

---

## Classification

### Amharic MLM corpus (`amharic_mlm`)

| Field | Value |
|---|---|
| Size | 200,000 lines / 9,192,496 characters |
| Confirmed component | HornMT (CC BY 4.0) — ~1% |
| Remainder | ~198,000 lines, source unidentified |
| **Classification** | ⚠️ **Licence clarification still required** |

### Tigrinya MLM corpus (`tigrinya_mlm`)

| Field | Value |
|---|---|
| Size | 200,000 lines / 6,982,894 characters |
| Confirmed component | HornMT (CC BY 4.0) — ~1% |
| Remainder | ~198,000 lines, source unidentified |
| **Classification** | ⚠️ **Licence clarification still required** |

### Why not the other classifications

**Not "redistributable"** — the CC BY 4.0 grant covers the ~2,030 HornMT records,
not the ~198,000 additional lines per language.

**Not "link-only"** — HornMT is linkable, but linking to it would not let a user
reconstruct the corpora actually used for pretraining.

---

## Task Datasets — for completeness

| Dataset | Licence | Classification |
|---|---|---|
| TIGQA | CC BY 4.0 | link-only |
| AmQA | MIT | link-only |
| TiQuAD | CC BY 4.0 | link-only |
| AfriSenti | CC BY 4.0 | link-only |
| MasakhaNER-Amharic | CC BY-NC 4.0 ⚠️ | link-only, non-commercial |
| Tigrinya NER | none upstream ⚠️ | licence clarification required |

⚠️ MasakhaNER's non-commercial term propagates to any model fine-tuned on it.
`vexmlm-stage1` is unaffected; NER fine-tunes are, which is one reason they are
not released.

---

## Final Release Recommendation

### Repository (GitHub) — ✅ **proceed**

No datasets are distributed. `datasets/registry.py`,
`docs/DATASET_PROVENANCE.md`, and `docs/DATA_SETUP.md` document sources,
licences, and acquisition. The repository states plainly what is known (HornMT,
CC BY 4.0) and what is not (the remaining ~99%).

Recommended action: update the registry notes for both corpora from
`UNKNOWN — blocker` to reflect the confirmed HornMT component and the
outstanding remainder.

### Model (Hugging Face) — ⚠️ **author decision**

A model card must state its training data. Two defensible paths:

1. **Publish with explicit disclosure** — state that the corpora comprise HornMT
   (CC BY 4.0) plus additional material whose provenance is under review. Honest,
   consistent with the repository's posture, and does not overclaim.
2. **Delay** until the remaining ~99% is traced.

**Recommended: option 1.** The disclosure is accurate, the corpora are not
redistributed, and the alternative postpones release indefinitely for a
question that may resolve favourably.

### Highest-value next step

Identify the source of the ~198,000 additional lines per language. If the
registry's CC-100/OSCAR hypothesis is correct, that material is permissively
licensed and would move both corpora to **redistributable**, closing the
question entirely.
