# Dataset Distribution Plan

**Date**: 2026-08-15
**Policy**: **no dataset files ship.** Every dataset is link-only or blocked.

---

## Classification

| Dataset | Licence | Classification |
|---|---|---|
| TIGQA | CC BY 4.0 | **Link-only** |
| AmQA | MIT | **Link-only** |
| TiQuAD | CC BY 4.0 | **Link-only** |
| AfriSenti | CC BY 4.0 | **Link-only** |
| MasakhaNER-Amharic | CC BY-NC 4.0 | **Link-only** ⚠️ non-commercial |
| Tigrinya NER | none upstream | **Requires licence verification** ⚠️ |
| **Amharic MLM corpus** | **UNKNOWN** | **Requires licence verification** 🔴 |
| **Tigrinya MLM corpus** | **UNKNOWN** | **Requires licence verification** 🔴 |

**Redistributable: none.** Even the permissively licensed sets are link-only —
redistribution adds no value when the originals are stable and citable, and it
creates a maintenance and attribution burden.

---

## Link-Only Datasets

| Dataset | Source | Splits |
|---|---|---|
| TIGQA | Zenodo (record 11423987) | 644 / 67 / 86 questions |
| AmQA | GitHub — semantic-systems/amharic-qa | 1,723 / 600 / 299 |
| TiQuAD | HF Hub — `fgaim/tiquad` | 4,452 / 926 |
| AfriSenti | GitHub — afrisenti-semeval | 5,984 / 1,497 / 1,999 |
| MasakhaNER-Amharic | HF Hub — `masakhane/masakhaner2` | 1,750 / 250 / 500 sentences |

Each is declared in `datasets/registry.py` with source URL, licence, citation,
and expected splits. `datasets/manager.py` records row counts and content hashes
on load, so users can verify they obtained identical data.

### ⚠️ MasakhaNER non-commercial constraint

CC BY-NC 4.0. **Any model fine-tuned on it inherits a non-commercial
restriction.** The Stage 1 base model is unaffected; NER fine-tuned checkpoints
are — which is one reason they are not released.

---

## Requires Licence Verification

### 🔴 Amharic MLM corpus

| Field | Value |
|---|---|
| Registry key | `amharic_mlm` |
| Licence | **UNKNOWN — blocker** |
| Size | 200,001 lines / 9,192,496 characters |
| Nominal source | HornMT |
| **Problem** | HornMT is a ~2,030-sentence parallel corpus and cannot be the origin of a 200,000-line monolingual file. Content sampling shows religious translations plus general web text, consistent with a CC-100/OSCAR-style crawl. |
| Status | Provenance unresolved; treated as a release blocker |

### 🔴 Tigrinya MLM corpus

| Field | Value |
|---|---|
| Registry key | `tigrinya_mlm` |
| Licence | **UNKNOWN — blocker** |
| Size | 200,000 lines / 6,982,894 characters |
| Nominal source | HornMT |
| **Problem** | Same discrepancy |
| Status | Provenance unresolved; treated as a release blocker |

**Neither is distributed.** Both are gitignored and excluded from every release
artifact.

#### Why this matters beyond the repository

These corpora trained the released model. A Hugging Face model card **must state
its training data**. Publishing a card that asserts unverified provenance is a
liability — which is why B2/B3 block the model release, not just the repository.

**Required action** (authors only):
1. Identify the true source of both files.
2. Determine the licence.
3. Update `datasets/registry.py` and `docs/DATASET_PROVENANCE.md`.
4. If the licence forbids derivative distribution, reassess the model release.

### ⚠️ Tigrinya NER

No licence file in the upstream repository (`mehari-eng/Tigrinya-NER`). Not
distributed; documented as unlicensed-upstream. Users assume their own risk.

---

## Release Recommendations

1. **Ship no data.** Already enforced — `datasets/{raw,processed,cache}/` are
   gitignored.
2. **Keep the registry authoritative.** `datasets/registry.py` is machine-readable
   and already records the unresolved licences honestly.
3. **Surface the licence table** in both the README and
   `docs/DATASET_PROVENANCE.md`.
4. **Keep the fail-loud behaviour.** `scripts/reproduce_paper.sh` halts with
   `die "pretraining corpora absent"` rather than proceeding — correct.
5. **Resolve B2/B3 before the model release**, not merely before the repository
   release.
6. **State the MasakhaNER constraint** wherever NER results or checkpoints are
   discussed.

---

## Verification

| Check | Result |
|---|---|
| Dataset files tracked in git | ✅ **0** |
| `datasets/raw/` gitignored | ✅ |
| `datasets/processed/` gitignored | ✅ |
| Unknown-licence data redistributed | ✅ **no** |
| Licences recorded in registry | ✅ all 8 |
| Acquisition instructions | ✅ `docs/DATA_SETUP.md` |

**No unknown-licence data is redistributed.**
