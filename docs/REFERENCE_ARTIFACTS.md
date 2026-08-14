# Reference Artifacts

Pre-existing Ge'ez XLM-R artifacts on this machine were analysed before this
implementation was written. They are **reference material only** — none of them
is a VEXMLM model, and none of their design decisions is carried forward. This
document records what they are and what was learned, because two of the findings
directly justify safeguards in the code.

Provenance for every file: [`legacy/PROVENANCE.json`](../legacy/PROVENANCE.json)
(source path, size, SHA-256, mtime). Regenerate with:

```bash
python scripts/forensic_import.py --copy
python scripts/analyze_legacy_artifacts.py --out results/legacy_analysis.json
```

Every number below comes from that second script.

## Inventory

| Artifact | Location | Size | Role |
|---|---|---|---|
| `EXLMR.py` | `~/EXLMR/` | 7,415 B | Prototype: vocab expansion + Tigrinya binary classification |
| `vocab.json` | `~/EXLMR/` | 918,848 B | 30,522 candidate tokens; no generating script |
| Expanded checkpoint | `~/VEXMLM_Model/` | 1.2 GB | `vocab_size: 280147`, pre-fine-tune |
| Release candidate | `~/EXLMR_release/` | 976 MB | Fine-tuned Tigrinya classifier |

## Finding 1 — the shipped vocabulary is inert

The 30,145 added tokens are **byte-level-BPE encoded**, but were added to XLM-R's
**SentencePiece** tokenizer, which matches on raw Unicode.

| Test | Result |
|---|---|
| Tokens containing Ethiopic as stored | **0** of 30,145 |
| Tokens decoding to Ethiopic via byte-level BPE | **29,819** (98.9%) |
| Example | `'áĪ¨'` → `'ረ'` (U+1228) |

The consequence, measured on 2,000 real Tigrinya sentences:

| Metric | Value |
|---|---|
| New-token firings | **0** across 84,101 tokens |
| Sentences using ≥1 new token | **0** |
| Tokens, expanded model | 84,101 |
| Tokens, stock XLM-R | 83,953 |

The expanded tokenizer segments Ge'ez **identically to stock XLM-R**, and in
aggregate slightly worse. The 30,145 embedding rows are unreachable.

**Consequence for this repository.** Token selection and expansion both enforce
an encoding gate that aborts on byte-level-BPE input, and expansion runs a
firing check on held-out text afterwards. A vocabulary that cannot fire is
treated as a fatal error, not a warning:

```
ABORT: 12/50 tokens are byte-level-BPE encoded (e.g. ['"áį¡', ...] -> mojibake).
Added to a SentencePiece tokenizer these can NEVER match input text.
```

## Finding 2 — no SentencePiece model was ever trained

`sentencepiece.bpe.model` is byte-identical across `~/VEXMLM_Model/`,
`~/EXLMR_Model/`, and `~/EXLMR_release/`:

```
sha256 cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865
piece_size 250000        # stock XLM-R
```

Neither a 32,000-piece Amharic model nor a 50,000-piece Tigrinya model exists
anywhere on this machine. This repository implements that training from scratch
(`tokenizer/train_{amharic,tigrinya}_tokenizer.py`).

## Finding 3 — the reference checkpoint uses a non-paper initialization

Measured directly from `~/VEXMLM_Model/model.safetensors`, embedding shape
(280147, 768):

| Quantity | Observed |
|---|---|
| Pretrained rows, per-dim std | 0.2123 |
| New rows, per-dim std | 0.5010 |
| E‖e_new − ē‖ | 13.8599 |
| Rows equal to the global mean | 0 |

Against the closed forms:

| Strategy | Predicted E‖e_new − ē‖ | Relative error |
|---|---|---|
| `global_mean` (paper) | 0.0 | — |
| **`mixed` = (N(0,1)+ē)/2** | **13.8564** | **0.000255** |
| `random` N(0,1) | 27.7128 | 0.4999 |

The reference checkpoint uses `(random + mean)/2`, agreeing with that prediction
to 0.03%. It is **not** the paper's mean-based initialization, under which every
new row would sit exactly at the centroid (distance 0).

**Consequence for this repository.** The paper's method is the default. The
`mixed` strategy is retained solely so `verify_vocab.py` can recognize
externally-supplied checkpoints built this way; it is not a VEXMLM method and
must not be used for results.

## Finding 4 — vocabulary size

| Quantity | Value |
|---|---|
| Candidates in `vocab.json` | 30,522 |
| Already present in XLM-R (skipped) | 377 |
| Actually added | 30,145 |
| Final vocab size | 250,002 + 30,145 = **280,147** |

The arithmetic is self-consistent. The paper states a ~30,000/280,000 target;
280,147 is that target as realized by one particular candidate list, differing
by 147. This repository treats 30,000 as a **configurable target**
(`vocabulary_expansion.target_new_tokens`) rather than a hardcoded constant, and
`merge_vocabularies.py --target-total` trims to it exactly.

## What was not carried forward

- the byte-level-encoded vocabulary (unusable)
- the mixed initialization (not the paper's method)
- single-seed, untracked runs with no random seed set anywhere
- hardcoded absolute paths
- the binary Tigrinya classification dataset of undocumented provenance
  (see [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md))
