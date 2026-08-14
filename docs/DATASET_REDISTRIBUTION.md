# Dataset Redistribution Review

Whether each dataset may ship inside this repository, or must be downloaded at
build time. Licences were read from the authoritative source (Zenodo record,
repository LICENSE file, or upstream README) and are quoted where they matter.

Reviewed 2026-08-14.

## Verdict

| Dataset | Licence | Redistribute? | Verdict |
|---|---|---|---|
| TIGQA | CC BY 4.0 | ✅ yes, with attribution | **SAFE TO SHIP** |
| AmQA | MIT | ✅ yes, with notice | **SAFE TO SHIP** |
| AfriSenti | CC BY 4.0 | ⚠ licence yes; platform terms apply | **DOWNLOAD AT BUILD TIME** |
| MasakhaNER | **CC BY-NC 4.0** | ⚠ non-commercial only | **DOWNLOAD AT BUILD TIME** |
| Tigrinya NER | **none stated** | ❌ no permission granted | **DOWNLOAD AT BUILD TIME** |
| Amharic MLM corpus | **unknown** | ❌ source unidentified | **DO NOT SHIP** |
| Tigrinya MLM corpus | **unknown** | ❌ source unidentified | **DO NOT SHIP** |

**Current repository state:** `datasets/raw/` is git-ignored, so *no* dataset is
committed today. Nothing needs to be removed; what follows governs what may be
added to a public release.

---

## TIGQA — SAFE TO SHIP

| Field | Value |
|---|---|
| Source | <https://zenodo.org/records/11423987> |
| Licence | **CC BY 4.0** (verified on the Zenodo record) |
| Redistribution | Allowed, including commercially and as derivatives |
| Restriction | Attribution required |
| Citation | Required — Teklehaymanot, Fazlija, Ganguly, Patro, Nejdl |
| Download needed | No |

CC BY 4.0 permits redistribution and adaptation provided the creators are
credited. The prepared 80/10/10 split is a derivative and may ship, as long as
attribution and a statement of changes accompany it (this repository's changes
are recorded in [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md)).

## AmQA — SAFE TO SHIP

| Field | Value |
|---|---|
| Source | <https://github.com/semantic-systems/amharic-qa> |
| Licence | **MIT** |
| Redistribution | Allowed |
| Restriction | Include the MIT licence text and copyright notice |
| Citation | Requested — Taffa et al. (2024) |
| Download needed | No |

If AmQA is bundled, copy the upstream `LICENSE` file alongside the data as
`datasets/raw/amqa/LICENSE`.

## AfriSenti — DOWNLOAD AT BUILD TIME

| Field | Value |
|---|---|
| Source | <https://github.com/afrisenti-semeval/afrisent-semeval-2023> |
| Licence | **CC BY 4.0** |
| Redistribution | Permitted by the dataset licence |
| Restriction | Attribution; **plus** platform terms on the underlying tweets |
| Citation | Required — Muhammad et al. (2023) |
| Download needed | Recommended |

The dataset licence alone would allow shipping. The complication is upstream:
the corpus consists of tweet text, and X/Twitter's developer terms have
historically required redistribution as *tweet IDs* rather than hydrated text.
The AfriSenti repository does not address this, so the constraint is unresolved
rather than absent.

Loading from the Hub (`shmuhammad/AfriSenti-twitter-sentiment`) is already the
default path in `datasets/registry.py`, which sidesteps the question entirely.
Keep it that way; no local copy is needed.

## MasakhaNER — DOWNLOAD AT BUILD TIME

| Field | Value |
|---|---|
| Source | <https://github.com/masakhane-io/masakhane-ner> |
| Licence | **CC BY-NC 4.0** — non-commercial |
| Redistribution | Permitted for non-commercial use, with attribution |
| Restriction | **No commercial use** |
| Citation | Required — Adelani et al. (2021) |
| Download needed | Recommended |

Verbatim from the upstream README:

> "The license of the NER dataset is in
> [CC-BY-4.0-NC](https://creativecommons.org/licenses/by-nc/4.0/)"

Two consequences worth stating plainly:

1. **This is the only non-commercial dataset in the set.** An earlier revision of
   `datasets/registry.py` recorded it as plain CC BY 4.0; that was wrong and has
   been corrected.
2. **The constraint propagates.** A model fine-tuned on MasakhaNER is a
   derivative of NC-licensed data. Any released Amharic NER checkpoint should
   therefore carry a non-commercial notice in its model card, and the repository
   licence (Apache-2.0) does not override it.

The upstream README also notes the underlying monolingual text carries
per-news-site licences, which are not enumerated. Loading from the Hub avoids
re-publishing the text.

## Tigrinya NER — DOWNLOAD AT BUILD TIME

| Field | Value |
|---|---|
| Source | <https://github.com/mehari-eng/Tigrinya-NER> |
| Licence | **None stated** — no LICENSE file, no licence statement |
| Redistribution | ❌ **Not permitted** — no rights are granted |
| Citation | Requested — Yohannes & Amagasa (2022), *ACM SIGAPP Applied Computing Review* 22(3):56–68 |
| Download needed | **Yes** |

Absence of a licence is not permission. Under default copyright, publishing
without an explicit grant reserves all rights, so this dataset **must not be
bundled**, even though it is freely downloadable and the authors clearly intend
academic use (they request citation).

The pipeline must fetch it from the source repository at build time. Users may
download it; this project may not redistribute it.

**To upgrade this to SAFE TO SHIP:** ask the authors to add an explicit licence
(CC BY 4.0 would match the field norm), or obtain written permission and record
it here with the date.

## MLM corpora — DO NOT SHIP

| Field | Amharic | Tigrinya |
|---|---|---|
| Claimed source | HornMT (ruled out) | HornMT (ruled out) |
| Actual source | **UNKNOWN** | **UNKNOWN** |
| Licence | **UNKNOWN** | **UNKNOWN** |
| Redistribution | ❌ Not permitted | ❌ Not permitted |

An unidentified source cannot be licence-cleared. See
[MLM_CORPUS_ANALYSIS.md](MLM_CORPUS_ANALYSIS.md) for the evidence ruling out
HornMT (2,030 reference lines vs 200,000 here, zero exact overlap).

Local use for Stage 1 training is unaffected. What is blocked is redistribution,
and any published claim about what the model was trained on.

---

## Build-time acquisition

`scripts/acquire_datasets.py` already implements the required flow: it searches
locally first and reports the official URL when a dataset is absent. For a
public release, the non-shippable datasets should be fetched by the user:

```bash
python scripts/acquire_datasets.py --report   # lists what is missing and where to get it
```

Recommended packaging for a public release:

| Ships in the repository | Fetched by the user |
|---|---|
| TIGQA (CC BY 4.0) | AfriSenti (Hub) |
| AmQA (MIT, with LICENSE) | MasakhaNER (Hub) |
| | Tigrinya NER (source repo) |
| | MLM corpora (blocked pending provenance) |

## Attribution block for the release

Every release must carry these citations; all five datasets require or request
attribution:

- **TIGQA** — Teklehaymanot, Fazlija, Ganguly, Patro, Nejdl (2024). CC BY 4.0.
- **AmQA** — Taffa et al. (2024). MIT.
- **MasakhaNER** — Adelani et al. (2021), *TACL*. CC BY-NC 4.0 (non-commercial).
- **Tigrinya NER** — Yohannes & Amagasa (2022), *ACM SIGAPP ACR* 22(3):56–68.
- **AfriSenti** — Muhammad et al. (2023), SemEval-2023 Task 12. CC BY 4.0.

BibTeX for each is in [`datasets/registry.py`](../datasets/registry.py) and
rendered into the per-dataset cards under `datasets/cards/`.

## Open items

| # | Item | Owner |
|---|---|---|
| 1 | MLM corpus provenance and licence | authors |
| 2 | Tigrinya NER licence grant | upstream authors |
| 3 | Decide whether released NER checkpoints inherit the NC constraint | authors |
| 4 | Confirm AfriSenti tweet-text redistribution stance if bundling is ever wanted | authors |
