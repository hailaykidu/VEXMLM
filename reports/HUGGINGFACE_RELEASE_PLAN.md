# Hugging Face Release Plan — `vexmlm-stage1`

**Date**: 2026-08-15
**Artifact**: `vexmlm-stage1` — the official public baseline accompanying the paper
**Paper**: *Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya* (LM4UC Workshop, IJCAI 2026)

---

## 1. Model Description

`vexmlm-stage1` is `xlm-roberta-base` adapted to Amharic and Tigrinya through
vocabulary expansion and continued masked-language-model pretraining.

| Property | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Architecture | XLMRobertaForMaskedLM, 12 layers, hidden 768, 12 heads |
| Vocabulary | 250,002 → **280,002** (~30K Ge'ez-derived subwords added) |
| Languages | Amharic (amh), Tigrinya (tir) |
| Training | Stage 1 continued MLM, all parameters trainable |
| Size | ~1.2 GB |
| Precision | BF16 training; FP32 weights |

**Intended use**: a base encoder for Amharic and Tigrinya downstream tasks —
question answering, named entity recognition, sentiment analysis, and further
fine-tuning. It is **not** an instruction-tuned or generative model.

---

## 2. Model Card Draft

```markdown
---
language:
  - am
  - ti
license: apache-2.0
base_model: xlm-roberta-base
tags:
  - vocabulary-expansion
  - amharic
  - tigrinya
  - geez
  - low-resource
  - fill-mask
library_name: transformers
pipeline_tag: fill-mask
---

# VEXMLM (Stage 1)

Vocabulary-extended XLM-R for Amharic and Tigrinya.

Official model for *Expanding the Lexicon of Ge'ez Based African Languages:
A Comparative Study of Amharic and Tigrinya* (LM4UC Workshop, IJCAI 2026).

## Model Details

- **Base**: `xlm-roberta-base`
- **Vocabulary**: 250,002 → 280,002 (~30K Ge'ez-derived subwords appended;
  all original entries preserved)
- **Initialization**: new embeddings set to the centroid of the pretrained
  embedding space, e_t = (1/|V_s|) Σ e_s
- **Training**: continued MLM pretraining on Amharic and Tigrinya, all
  parameters trainable
- **Languages**: Amharic (amh), Tigrinya (tir)

## Why

XLM-R's vocabulary is only ~1.2% Ge'ez-script, and Tigrinya fragments into
roughly 3.1 subwords per word. VEXMLM reduces that fragmentation and improves
out-of-vocabulary handling.

## Intended Use

Base encoder for Amharic and Tigrinya: QA, NER, sentiment analysis, and further
fine-tuning. Not instruction-tuned; not a generative model.

## Known Limitation — Tokenizer Decoding

This checkpoint's vocabulary was installed via HuggingFace `added_tokens` rather
than merged into the SentencePiece model. As a result, `decode()` does not
exactly reconstruct Ge'ez-script input:

```
input : ኢትዮጵያ
decode: ኢት ዮጵያ
```

**This does not affect encoding, training, or evaluation.** Token IDs are
deterministic and character offsets are correct (TIGQA span recovery 96.3%), so
fine-tuning, feature extraction, and span-based tasks (QA, NER) behave exactly
as reported in the paper. Every published result derives from token IDs and
offsets, never from decoded text.

**Affected**: workflows that read back decoded text — `pipeline()` output, text
reconstruction, or re-tokenizing decoded strings.

This is the artifact **as evaluated in the paper**, released for reproducibility.
A corrected tokenizer has been built and validated (100% round-trip); it will be
published separately once trained, so this artifact remains retrievable.

## Limitations

- Vocabulary construction and continued pretraining cover **two languages**.
  Broader intrinsic evaluation (19 languages) is tokenizer-level only.
- Continued pretraining used a modest corpus; downstream fine-tuning is
  recommended for task use.
- Glot500 outperforms this model on NER accuracy (see the paper).
- ⚠️ Fine-tuning on MasakhaNER (CC BY-NC 4.0) makes the resulting checkpoint
  non-commercial. This base model is unaffected.

## Training Data

Amharic and Tigrinya monolingual corpora. Dataset sources and licences are
recorded in the repository (`datasets/registry.py`). No datasets are
redistributed here.

## Citation

See below.
```

---

## 3. Required Model Files

| File | Purpose |
|---|---|
| `config.json` | architecture and `vocab_size: 280002` |
| `model.safetensors` | weights (~1.1 GB) |
| `generation_config.json` | if present |

---

## 4. Tokenizer Files

| File | Purpose |
|---|---|
| `sentencepiece.bpe.model` | SentencePiece model |
| `tokenizer.json` | fast tokenizer |
| `tokenizer_config.json` | tokenizer settings |
| `special_tokens_map.json` | special token mapping |

Also required: `added_tokens.json` (834 KB) — this tokenizer depends on it.

⚠️ **Verify before upload** that `len(tokenizer) == 280002` matches
`config.json`. Note that this tokenizer **does not** round-trip Ge'ez text; that
is the documented B5 limitation, disclosed in the model card, and is expected
rather than a defect to fix before upload.

---

## 5. Config Files

`config.json` must state: `model_type: xlm-roberta`, `vocab_size: 280002`,
`num_hidden_layers: 12`, `hidden_size: 768`, `num_attention_heads: 12`,
`architectures: ["XLMRobertaForMaskedLM"]`.

---

## 6. Example Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained("Hailay/vexmlm-stage1")
model = AutoModelForMaskedLM.from_pretrained("Hailay/vexmlm-stage1")

text = "ኢትዮጵያ ኣብ ቀርኒ ኣፍሪቃ እትርከብ ሃገር እያ።"
print(tok.tokenize(text))
```

Masked prediction:

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="Hailay/vexmlm-stage1")
fill(f"ኣዲስ ኣበባ {fill.tokenizer.mask_token} ኢትዮጵያ እያ።")
```

Fine-tuning for token classification:

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "Hailay/vexmlm-stage1", num_labels=9)
```

---

## 7. Repository Links

| Resource | Location |
|---|---|
| Code | `https://github.com/hailaykidu/VEXMLM` |
| Reproduction | `scripts/reproduce_paper.sh` |
| Results | `reports/RESULTS_SUMMARY.md` |
| Documentation | `reports/DOCUMENTATION_OVERVIEW.md` |
| Dataset registry | `datasets/registry.py` |

---

## 8. Citation Block

```bibtex
@inproceedings{teklehaymanot2026vexmlm,
  title     = {Expanding the Lexicon of Ge'ez Based African Languages:
               A Comparative Study of Amharic and Tigrinya},
  author    = {Teklehaymanot, Hailay},
  booktitle = {Proceedings of the Workshop on Language Models for
               Underserved Communities (LM4UC), IJCAI},
  year      = {2026}
}
```

---

## 9. License Considerations

**Model licence**: Apache-2.0, inherited from `xlm-roberta-base` and consistent
with the repository.

### ⚠️ Blocking condition

The model card must state its training data. Two pretraining corpora currently
have **UNKNOWN** licences (blockers B2, B3). **Do not publish until their
provenance is confirmed** — a public model card asserting unverified training
data is a liability.

| Constraint | Effect |
|---|---|
| Amharic MLM corpus — UNKNOWN | 🔴 blocks publication |
| Tigrinya MLM corpus — UNKNOWN | 🔴 blocks publication |
| MasakhaNER CC BY-NC 4.0 | ⚠️ does **not** affect this model; affects NER fine-tunes only |

---

## 10. Release Checklist

### Pre-upload

- ⬜ **B2/B3 resolved** — pretraining corpus licences confirmed 🔴
- ⬜ Model card states training data and licences accurately
- ✅ Model card carries the **Known Limitation — Tokenizer Decoding** section (B5)
- ⬜ Verify `len(tokenizer) == 280002 == config.vocab_size`
- ✅ Round-trip behaviour characterized and disclosed (B5) — no fix required pre-upload
- ⬜ Load-test `from_pretrained` from a clean directory
- ⬜ Confirm no training artifacts included (optimizer state, checkpoints)

### Upload

- ⬜ Create `Hailay/vexmlm-stage1`
- ⬜ Upload model, tokenizer, and config files
- ⬜ Publish the model card with venue and citation
- ⬜ Set language tags (`am`, `ti`), licence (`apache-2.0`), `base_model`

### Post-upload

- ⬜ Verify the example snippets run against the live model
- ⬜ Link the model from the repository README
- ⬜ Link the repository from the model card
- ⬜ Tag the corresponding repository release

---

## Downstream Checkpoints — Not Released

The 25 fine-tuned task checkpoints are **excluded**, for three reasons:

1. **Regenerable** — each reproduces in minutes via `scripts/slurm_finetune.sh`.
2. **Licence contamination** — NER checkpoints inherit MasakhaNER's CC BY-NC 4.0.
3. **Not the paper's artifact** — the paper's contribution is the vocabulary-
   expanded base model; task checkpoints are evaluation by-products.

`vexmlm-expanded` (pre-Stage-1) is optional; it would only aid ablation
replication and is not required to reproduce the paper's results.
