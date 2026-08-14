# Dataset Provenance

Origin, licence, checksum, and preprocessing for every dataset used by VEXMLM.

Regenerate the machine-readable record with:

```bash
python scripts/acquire_datasets.py --copy      # writes datasets/metadata/acquisition.json
```

## Acquisition policy

1. Search this repository's `datasets/raw/`.
2. Search other local repositories on this machine — **read-only**.
3. Only if absent, download from the official source URL.

Data is **copied** into this repository, never moved and never symlinked. Source
repositories are opened read-only and are never modified. Nothing synthetic is
ever generated.

All six datasets were resolved from the local workspace; **nothing was
downloaded**. Every copy's SHA-256 was verified to match its source (14/14).

## Summary

| Dataset | Task | Lang | Splits | Licence | Status |
|---|---|---|---|---|---|
| TIGQA | QA | tir | 644/67/86 q | see Zenodo record | ✅ integrated |
| AmQA | QA | amh | 1,723/600/299 q | MIT | ✅ integrated |
| MasakhaNER | NER | amh | 1,750/250/500 s | CC BY 4.0 | ✅ integrated |
| Tigrinya NER | NER | tir | 4,562/570/571 s | see source repo | ✅ integrated |
| AfriSenti | Sentiment | amh/tir | per config | CC BY 4.0 | ✅ Hub loader |
| Amharic corpus | MLM | amh | 200,001 lines | **UNKNOWN** | ⚠ blocker |
| Tigrinya corpus | MLM | tir | 200,000 lines | **UNKNOWN** | ⚠ blocker |

---

## TIGQA

| Field | Value |
|---|---|
| Source URL | <https://zenodo.org/records/11423987> |
| Citation | Teklehaymanot et al. (2024) |
| Version | TIGQA-1.0 |
| Language | Tigrinya (tir) |
| Imported from | `~/lgse-repro/data/qa/tigqa_squad` (read-only) |
| Upstream raw SHA-256 | `4d089b392276062390c201a966ebd56c17a3c4c4bad032d3e10db0bc8706912a` |

**Splits.** 797 question–answer pairs over 365 contexts, split 80/10/10 **by
context** with seed 42:

| Split | Contexts | Questions |
|---|---|---|
| train | 292 | 644 |
| validation | 36 | 67 |
| test | 37 | 86 |

**Preprocessing (upstream).** From the full release, 1,039 abstractive and 272
unanswerable items were excluded: they carry `answer_start == -1`, meaning the
answer does not appear verbatim in the context and cannot be scored by
span-extraction EM/F1. The prepared files contain **no** negative offsets
(verified). Splitting by context, not by question, prevents the same passage
appearing in both train and test.

**Format note.** TIGQA stores articles flat (`title`/`context`/`qas`) with no
intermediate `paragraphs` level, unlike canonical SQuAD. `datasets/manager.py`
handles both layouts.

## AmQA

| Field | Value |
|---|---|
| Source URL | <https://github.com/semantic-systems/amharic-qa> |
| Citation | Taffa et al. (2024) |
| Licence | MIT |
| Format | SQuAD v2.0 |
| Language | Amharic (amh) |
| Imported from | `~/lgse-repro/data/qa/amqa` (read-only) |

| Split | Questions | Upstream raw SHA-256 (prefix) |
|---|---|---|
| train | 1,723 | `80e216fe3594e02e` |
| validation | 600 | `889f4aeb478005a5` |
| test | 299 | `4bff7f83d9d72091` |

Official splits, used as released. Upstream repairs: 20 answer offsets
corrected; one article storing `paragraphs` as a dict wrapped into a list
(content unchanged). No questions dropped.

## MasakhaNER (Amharic)

| Field | Value |
|---|---|
| Source URL | <https://github.com/masakhane-io/masakhane-ner> |
| Citation | Adelani et al. (2021) |
| Licence | CC BY 4.0 |
| Imported from | `~/lgse-repro/data/ner/amharic` (read-only) |

| Split | Sentences | Tokens | SHA-256 (prefix) |
|---|---|---|---|
| train | 1,750 | 25,819 | `44d6e06d6bedd2b9` |
| validation | 250 | 3,749 | `f7561345b55d41cc` |
| test | 500 | 7,449 | `3968ef19c31c4808` |

CoNLL BIO tags: PER, ORG, LOC, DATE. Official splits, used as released.

## Tigrinya NER

| Field | Value |
|---|---|
| Source URL | <https://github.com/mehari-eng/Tigrinya-NER> |
| Citation | Yohannes and Amagasa (2022) |
| Licence | see source repository — confirm before redistribution |
| Imported from | `~/lgse-repro/data/ner/tigrinya` (read-only) |

| Split | Sentences | Tokens | SHA-256 (prefix) |
|---|---|---|---|
| train | 4,562 | 88,102 | `01a9968d78ce30c9` |
| validation | 570 | 11,003 | `e3f86c999f177e1f` |
| test | 571 | 10,818 | `19fa30537db6b88a` |

Tigrinya is **not** covered by MasakhaNER v1 or v2; this is a separate resource.

**Label statistics (train).**

| Tag | Count | | Tag | Count |
|---|---|---|---|---|
| O | 73,894 | | B-ORG | 1,498 |
| B-LOC | 2,998 | | I-PER | 1,135 |
| I-ORG | 1,947 | | I-LOC | 852 |
| B-DATE | 1,901 | | B-MISC | 351 |
| I-DATE | 1,742 | | I-MISC | 229 |
| B-PER | 1,554 | | **B-LO** | **1** ← defect |

**Known defect — repaired on read.** `train.conll` line 1350 carries the tag
`B-LO`, plainly a truncated `B-LOC`: the token is `ቺለ፡` ("Chile"), surrounded by
`ኣርጀንቲና`/`ሜክሲኮን`/`ጀርመንን`, all tagged `B-LOC`. Left as-is it would create a
spurious 12th label class, inflating the label set and depressing macro-F1 by
adding a class with a single training example and no test support.

`datasets/manager.py` maps it to `B-LOC` at read time via `CONLL_TAG_REPAIRS`
and logs the substitution. **The source file is never modified.**

## AfriSenti

| Field | Value |
|---|---|
| Source URL | <https://github.com/afrisenti-semeval/afrisent-semeval-2023> |
| Hub | `shmuhammad/AfriSenti-twitter-sentiment` |
| Citation | Muhammad et al. (2023), SemEval-2023 Task 12 |
| Licence | CC BY 4.0 |

3-class (positive/negative/neutral). Amharic (`amh`) is a native subtask. Loads
from the Hub; no local copy required.

---

## MLM pretraining corpora — PROVENANCE UNRESOLVED

| Field | Value |
|---|---|
| Designated source | <https://github.com/asmelashteka/HornMT> |
| Imported from | `~/lgse-repro/data/lapt/` (read-only) |
| Amharic | 200,001 lines, 9,192,496 characters |
| Tigrinya | 200,000 lines, 6,982,894 characters |
| Licence | **UNKNOWN** |

**The designated source does not match the imported files.** This is recorded
rather than papered over.

Evidence:

1. **Scale.** HornMT is a parallel MT corpus of roughly **2,030** sentence pairs
   (`~/amseg/data/corpus/hornmt_tir.txt`: 2,030 lines). The imported corpora are
   **200,000 lines each** — two orders of magnitude larger. HornMT cannot be
   their origin.
2. **Structure.** HornMT is parallel (aligned across languages). These files are
   monolingual and independently sized, and the round 200,000-line count in both
   indicates a deliberate cap applied during preparation.
3. **Content.** Sampling shows Qur'anic and Biblical translations alongside
   general web prose — the characteristic composition of a **CC-100 / OSCAR**
   style crawl, not of a curated MT corpus.
4. **No upstream record.** The originating workspace carries no manifest or
   documentation naming a source for these two files, unlike every other dataset
   there.

**Consequence.** Both corpora are marked `license: UNKNOWN — blocker` in
`datasets/registry.py`. They are usable for Stage 1 experiments on this machine,
but:

- they must **not** be redistributed until the source and licence are known;
- any model trained on them cannot be released with a complete data statement;
- the corpus row of every model card must stay `UNKNOWN` until resolved.

**To resolve**, the authors need to confirm which corpus was actually used —
CC-100, OSCAR, an in-house crawl, or something else — after which the registry
entry and this section should be corrected. If HornMT is genuinely intended, the
Stage 1 corpora must be rebuilt from it, and the resulting ~2K-sentence scale
will not support 10 epochs of continued pretraining in any meaningful sense.

## Preprocessing applied by this repository

`datasets/prepare.py` (all corpora):

- NFC normalization (Ethiopic is precomposed; NFKC would destructively fold);
- zero-width and bidi control characters stripped;
- whitespace collapsed;
- lines shorter than 10 characters dropped;
- lines below 50% Ethiopic characters dropped;
- exact duplicates removed;
- 2% held out as a dev split (seed 42), with an OOV word list derived from words
  present in dev but absent from train.

Counts before and after are written to
`datasets/processed/preparation_report.json`.
