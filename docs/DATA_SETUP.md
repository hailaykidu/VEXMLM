# Data Setup

Which datasets load automatically, which you must supply, and in what format.

## Status

| Dataset | Task | Loader | Action |
|---|---|---|---|
| MasakhaNER 2.0 (amh) | NER | Hub: `masakhane/masakhaner2` | automatic |
| AfriSenti | Sentiment | Hub: `shmuhammad/AfriSenti-twitter-sentiment` | automatic |
| TIGQA | QA (tir) | none verified | **supply locally** |
| AmQA | QA (amh) | none verified | **supply locally** |
| Tigrinya NER | NER (tir) | none | **supply locally** |
| Amharic corpus | Stage 1 MLM | none | **supply locally** |
| Tigrinya corpus | Stage 1 MLM | none | **supply locally** |

MasakhaNER covers Amharic but **not Tigrinya** in either v1 or v2 — the Tigrinya
NER corpus is a separate resource that must be named and supplied.

## Layout

```
datasets/raw/
├── amharic/*.txt          # one sentence or paragraph per line, UTF-8
├── tigrinya/*.txt
├── tigqa/{train,validation,test}.json
├── amqa/{train,validation,test}.json
└── tigrinya_ner/{train,validation,test}.conll
```

`datasets/prepare.py` reads the two corpus directories and writes normalized,
deduplicated splits plus an OOV word list to `datasets/processed/`.

## Formats

**MLM corpora** — plain UTF-8 text, one segment per line. Preparation applies
NFC normalization, strips zero-width and bidi controls, drops lines shorter than
10 characters or less than 50% Ethiopic, and removes exact duplicates.

**QA** — SQuAD-style JSON (nested `data → paragraphs → qas`) or JSONL with flat
records. `answer_start` must be a **character** offset into `context`:

```json
{"id": "q1", "context": "ኣዲስ ኣበባ ናይ ኢትዮጵያ ዋና ከተማ እያ።",
 "question": "ናይ ኢትዮጵያ ዋና ከተማ ኣየነይቲ እያ?",
 "answers": {"text": ["ኣዲስ ኣበባ"], "answer_start": [0]}}
```

**NER** — CoNLL: one `TOKEN TAG` pair per line, blank line between sentences,
BIO tags.

```
ኣዲስ	B-LOC
ኣበባ	I-LOC
ከተማ	O
```

## Parallel corpus for parity

Table 2's parity metric requires **sentence-aligned** Amharic–Tigrinya text:

```
datasets/processed/parallel.amh.txt
datasets/processed/parallel.tir.txt
```

Line *i* of each file must be a translation of the other. Pass `--parallel` to
`evaluation/run_intrinsic.py`; without it, parity is computed but explicitly
marked invalid, because on non-parallel text the ratio measures content
differences rather than tokenizer fairness.

## Corpus size

The paper's tokenizer targets (32K Amharic, 50K Tigrinya) need a corpus on the
order of hundreds of thousands of sentences. SentencePiece fails with a clear
message if `vocab_size` exceeds what the corpus can support:

```
Corpus too small for vocab_size=32000.
  ... Please set it to a value <= 88.
Either supply more text, or lower --vocab-size.
```

## Licences

Every dataset's licence is recorded in `datasets/registry.py` and rendered into
`datasets/cards/`. Check the registry for the licence of every dataset before
use.
