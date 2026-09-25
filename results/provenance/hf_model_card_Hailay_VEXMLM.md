---
language:
- am
- ti
license: apache-2.0
library_name: transformers
pipeline_tag: fill-mask
tags:
- xlm-roberta
- vocabulary-expansion
- geez
- amharic
- tigrinya
- low-resource
base_model: FacebookAI/xlm-roberta-base
---

# VEXMLM

**Vocabulary Expansion for Low-Resource Multilingual Language Modeling.**

VEXMLM extends `xlm-roberta-base` with 30,000 Ge'ez-script subword tokens merged
natively into its SentencePiece model, then adapts the expanded model with
continued masked-language-model pretraining on Amharic and Tigrinya.

Official implementation: **https://github.com/hailaykidu/VEXMLM**

## Languages

Amharic (`am`) and Tigrinya (`ti`) — the two highest-resource Ge'ez-script
languages. Both are covered by the pretraining corpus and by every reported
evaluation.

## Model details

| | |
|---|---|
| Base model | `FacebookAI/xlm-roberta-base` |
| Architecture | `XLMRobertaForMaskedLM`, 12 layers, hidden 768, 12 heads |
| Parameters | 301,365,186 |
| Vocabulary | **280,002** (250,002 base + 30,000 added) |
| Max position embeddings | 514 |
| Tokenizer | `XLMRobertaTokenizerFast` (SentencePiece) |

### Tokenizer

The 30,000 new tokens are merged directly into the SentencePiece model rather
than appended as Hugging Face `added_tokens`. Appending them causes the
added-token matcher to run before SentencePiece segmentation, which emits the
`▁` word-boundary marker mid-word and corrupts decoding; the merged
construction avoids this.

New embedding rows are initialized to the mean of the existing embedding matrix
(`global_mean`), then trained during continued pretraining.

### Tokenizer quality

Measured on Amharic and Tigrinya development corpora (2,588 / 2,811 sentences):

| Metric | Language | XLM-R | VEXMLM |
|---|---|---|---|
| Fertility (subwords/word) ↓ | Amharic | 2.0692 | **1.4888** |
| | Tigrinya | 3.1300 | **1.6928** |
| Compression (chars/token) ↑ | Amharic | 2.2950 | **3.1896** |
| | Tigrinya | 1.4591 | **2.6979** |
| OOV word round-trip ↑ | Amharic | 1.0000 | 1.0000 |
| | Tigrinya | 0.9954 | **0.9987** |

Tigrinya fertility falls 45.9% and compression rises 84.9%. Added tokens carry
24.2% (Amharic) and 45.5% (Tigrinya) of token mass, so they are actively used.

## Training

Continued MLM pretraining on Amharic and Tigrinya monolingual corpora
(200,001 and 200,000 non-empty lines; 9.09M and 6.98M characters).

| Hyperparameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 32 |
| Epochs | 56 completed of 60 configured |
| Learning rate | 5e-5 |
| LR schedule | Linear decay, 6% warmup |
| Weight decay | 0.01 |
| MLM probability | 0.15 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8) |
| Precision | bf16 |
| Hardware | 1× NVIDIA A100 |

Released checkpoint is the best-by-validation-loss model at epoch 56
(24,808 of 26,580 steps): **eval loss 3.7120, perplexity 41.67**.

## Evaluation

Fine-tuned downstream, seeds 42–46, one configuration (hash `ce27cc194946`) on
A100-PCIE-40GB with deterministic kernels. Mean ± standard deviation over 5 seeds.

| Task | Dataset | Metric | VEXMLM |
|---|---|---|---|
| NER | MasakhaNER Amharic | Accuracy | 0.9413 ± 0.0026 |
| | | Macro-F1 | 0.7423 ± 0.0122 |
| | | Entity-F1 | 0.6347 ± 0.0148 |
| NER | Tigrinya NER | Accuracy | 0.9515 ± 0.0005 |
| | | Macro-F1 | 0.8219 ± 0.0069 |
| | | Entity-F1 | 0.7282 ± 0.0079 |
| QA | AmQA | EM | 32.57 ± 0.77 |
| | | F1 | 48.85 ± 0.96 |
| QA | TIGQA | EM | 2.39 ± 0.82 |
| | | F1 | 9.76 ± 0.97 |
| SA | AfriSenti (Amharic) | Accuracy | 0.4978 ± 0.0331 |
| | | Macro-F1 | 0.4971 ± 0.0193 |

**Supplementary** — TiQuAD is a diagnostic task, not a paper benchmark:
EM 50.24 ± 0.48, F1 58.90 ± 0.66 (926 questions).

TIGQA has only 67 test questions, too few to support a QA claim on its own;
TiQuAD is reported alongside it for that reason.

### Ablation — downstream NER OOV accuracy

Tigrinya NER, 4 configurations × 5 seeds. A word is out-of-vocabulary when the
baseline `xlm-roberta-base` tokenizer emits `<unk>`, fails to round-trip it, or
fragments it into more subwords than the expanded tokenizer. All arms are scored
on one identical set: 3,491 of 4,677 word types (74.6%).

| Configuration | OOV Acc. (%) | Δ |
|---|---|---|
| XLM-R baseline | 94.57 ± 0.16 | — |
| + VocabExp (Random Init) | 87.04 ± 0.20 | −7.52 |
| + VocabExp (Mean Init) | 87.63 ± 0.14 | +0.59 |
| + Continued Pretraining | **95.66 ± 0.09** | +8.02 |

Vocabulary expansion **alone degrades** OOV accuracy: the newly added embedding
rows are untrained, and the classifier must work around them. Continued
pretraining adapts the expanded vocabulary, recovers that loss, and finishes
**1.09 points above** the baseline.

Note that arms 2 and 3 receive no continued pretraining at all, so the +8.02
attributed to it also includes the effect of 56 additional epochs of training on
Amharic/Tigrinya text. Separating embedding adaptation from general continued
training would require an unexpanded arm given the same budget, which was not run.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Hailay/VEXMLM")
model = AutoModelForMaskedLM.from_pretrained("Hailay/VEXMLM")
model.eval()

text = "ትግርኛ <mask> ቋንቋ እዩ።"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

mask_pos = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero()[0, 0]
top = logits[0, mask_pos].topk(3).indices.tolist()
print([tokenizer.decode([t]).strip() for t in top])
```

This is a masked-language model. For token classification, question answering, or
sequence classification, fine-tune it with the corresponding
`AutoModelFor...` class — see the
[GitHub repository](https://github.com/hailaykidu/VEXMLM) for the fine-tuning
scripts and configurations used to produce the results above.

## Intended use

Intended for research on Amharic and Tigrinya NLP: as a starting point for
fine-tuning on token classification, extractive QA, and sequence classification,
and for studying vocabulary expansion in low-resource multilingual models.

## Limitations

- **Two languages only.** Amharic and Tigrinya. Other Ge'ez-script languages were
  not part of pretraining and are not evaluated here.
- **Extractive QA remains weak in absolute terms.** TIGQA EM of 2.39 reflects a
  very small dataset (67 test questions) and a hard task, not a usable QA system.
- **Sentiment results are near chance** on AfriSenti Amharic (accuracy 0.4978 on
  a 3-class task).
- **Tokenizer parity is not reported.** It requires a sentence-aligned parallel
  corpus, which was unavailable; parity computed on non-parallel text reflects
  content differences rather than tokenizer fairness.
- **Baseline comparisons are single-seed.** XLM-R and Glot500 comparison runs
  exist for seed 42 only, so no multi-seed head-to-head claim is made.
- Continued pretraining conflates embedding adaptation with additional training
  budget, as noted in the ablation section.
- The corpora are drawn largely from religious and news domains; the model may
  reflect those distributions and any biases present in them.

## Citation

```bibtex
@inproceedings{teklehaymanot2026vexmlm,
  title     = {Expanding the Lexicon of Ge'ez Based African Languages:
               A Comparative Study of Amharic and Tigrinya},
  author    = {Teklehaymanot, Hailay Kidu and Yadeta, Gebregziabihier and
               Nejdl, Wolfgang},
  booktitle = {Proceedings of the Workshop on Language Models for
               Underserved Communities (LM4UC) at IJCAI},
  year      = {2026}
}
```

Accepted at the LM4UC Workshop, IJCAI 2026.

## License

Apache 2.0, following `xlm-roberta-base`.
