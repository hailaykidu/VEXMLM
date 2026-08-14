---
language:
  - ti
base_model:
  - FacebookAI/xlm-roberta-base
tags:
  - xlm-roberta
  - tigrinya
  - geez-script
  - text-classification
  - offensive-language
  - hate-speech-detection
  - low-resource
pipeline_tag: text-classification
widget:
  - text: "እንቋዕ ብደሓን መጻእካ፣ ጽቡቕ ስራሕ ኢዩ"
  - text: "ደንቆሮ ኢኻ ትም በል"
---

# EXLMR: Vocabulary-Extended XLM-R for Tigrinya Text Classification

[![Model](https://img.shields.io/badge/HuggingFace-EXLMR-yellow)](https://huggingface.co/Hailay/EXLMR)
[![Language](https://img.shields.io/badge/Language-Tigrinya-orange)]()
[![Base Model](https://img.shields.io/badge/Base-XLM--RoBERTa--base-blue)](https://huggingface.co/FacebookAI/xlm-roberta-base)

---

## Overview

Multilingual pretrained language models such as XLM-R suffer from high
out-of-vocabulary rates and heavy subword fragmentation on Ge'ez-script
languages like Tigrinya, which are underrepresented in their training data.

**EXLMR** ("Extended XLM-R") addresses this for Tigrinya by extending
`xlm-roberta-base`'s subword vocabulary with Tigrinya-specific tokens, then
fine-tuning the resulting model as a binary text classifier over Tigrinya
social-media-style text.

**A note on the label names:** the model's output labels are `negative` /
`positive`, inherited from the original training script. Based on manual
inspection of the training data, the `negative` class is dominated by
offensive language, slurs, and hostile/hateful comments — not merely
critical or unfavorable opinions — while `positive` covers compliments,
well-wishes, and other non-offensive content. In practice this model
behaves as an **offensive-language / hate-speech-leaning classifier** for
Tigrinya rather than a general-purpose sentiment analyzer. See
[Model Behavior](#model-behavior) for a concrete illustration.

---

## Results

Evaluated on a held-out Tigrinya test set of 4,000 examples. Because
`load_best_model_at_end=True` with `metric_for_best_model="f1"`, the
checkpoint shipped in this repository is the **best of 3 epochs by weighted
F1** (reached at epoch 1), not the final epoch:

| Metric | Value |
|---|---|
| Accuracy | 80.4% |
| F1 (weighted) | 0.8024 |
| Precision (weighted) | 0.8160 |
| Recall (weighted) | 0.8043 |
| Eval loss | 0.6558 |

Per-epoch weighted F1 across the full run: epoch 1 — 0.8024 (best, shipped),
epoch 2 — 0.7968, epoch 3 — 0.8020.

---

## Supported Languages

| Language | Script | ISO Code |
|---|---|---|
| Tigrinya | Ge'ez (Ethiopic) | `tir` |

Tigrinya-distinctive grammatical markers (e.g. `ካብ`, `እንታይ`, `ኣሎ`, `ድዩ`,
`ኣይኮነን`) appear thousands of times in the training data, while
Amharic-distinctive equivalents are effectively absent. **This model has
not been evaluated on Amharic or other Ge'ez-script languages** and should
not be assumed to generalize to them despite the shared script and
multilingual base model.

---

## Model Architecture

```
XLM-R (xlm-roberta-base)
    └── Vocabulary Expansion (+30,145 Tigrinya subword tokens)
            └── Mixed embedding initialization (random + mean of source embeddings)
                    └── Fine-tuning: binary sequence classification
```

- **Base model:** `xlm-roberta-base`
- **Tokenizer:** original XLM-R SentencePiece vocabulary, extended with
  Tigrinya-specific tokens
- **Vocabulary extension:** 30,145 new tokens actually added (of 30,522
  candidates; the rest already existed in XLM-R's vocabulary) → final
  vocabulary size **280,147** (verified to match the model's embedding
  matrix)
- **Embedding initialization:** each new token's embedding is the average
  of a random-normal vector and the mean of all pretrained token embeddings
- **Training framework:** Hugging Face Transformers, PyTorch

---

## Installation

```bash
pip install transformers torch sentencepiece
```

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Hailay/EXLMR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "እንቋዕ ብደሓን መጻእካ፣ ጽቡቕ ስራሕ ኢዩ"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

prediction = outputs.logits.argmax(-1)
print(model.config.id2label[prediction.item()])
```

---

## Model Behavior

**Input:** Tigrinya text written in Ethiopic script.
**Output:** one of `negative` or `positive`.

Verified examples (actual model output):

```
Input:  እንቋዕ ብደሓን መጻእካ፣ ጽቡቕ ስራሕ ኢዩ        (a well-wish / compliment)
Output: positive   (confidence 0.999)

Input:  ደንቆሮ ኢኻ ትም በል                        (a mild insult)
Output: negative   (confidence 0.994)
```

This next example illustrates the labeling nuance above — a
critical-but-not-offensive opinion is classified `positive` (i.e.
non-offensive), not `negative`:

```
Input:  እዚ ኣገልግሎት ኣዝዩ ደስ ዘይብል ነበረ           ("This service was very unpleasant")
Output: positive   (confidence 0.995)
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Training examples | 49,374 (label 0: 25,031 · label 1: 24,343) |
| Test examples | 4,000 |
| Added vocabulary tokens | 30,145 |
| Final vocabulary size | 280,147 |
| Epochs | 3 |
| Train batch size | 16 |
| Eval batch size | 8 |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Max sequence length | 128 |
| Model selection | best checkpoint by weighted F1 |

Training data source and annotation methodology are not documented
elsewhere in this project — treat the offensive vs. non-offensive framing
above as an empirical observation from inspecting label samples, not a
confirmed annotation guideline.

---

## Limitations

- Binary classification only — no neutral class or intensity levels.
- Functions as an offensive/hate-speech-leaning detector rather than a
  general sentiment analyzer: plain negative opinions without hostile or
  offensive language may be misclassified as `positive` (see the third
  example above).
- Trained and evaluated on Tigrinya only; no verified Amharic or other
  Ethiopic-script language coverage.
- Training data appears to be informal, social-media-style text;
  performance on formal, domain-specific, code-mixed,
  Romanized/Latin-transliterated, or dialectal Tigrinya has not been
  evaluated.
- Training data source and annotation guidelines are undocumented in the
  source project — verify the model's behavior against your own
  definitions before deployment in hate-speech-sensitive applications.

---

## Related Work & Acknowledgements

- Base model: [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base)
  (Conneau et al., 2020)

---

## License

Not specified in the source project. Contact the repository owner for
licensing terms before reuse.
