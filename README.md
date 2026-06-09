# VEXMLM: Vocabulary-Extended XLM-R for Ge'ez-Script African Languages

[![Paper](https://img.shields.io/badge/Paper-LM4UC@IJCAI2026-blue)](https://ijcai.org)
[![Model](https://img.shields.io/badge/HuggingFace-VEXMLM-yellow)](https://huggingface.co/Hailay)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Languages](https://img.shields.io/badge/Languages-19%20African-orange)]()

> **Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya**  
> Hailay Kidu Teklehaymanot, Wolfgang Nejdl 
> *Third Workshop on Language Models for Underserved Communities (LM4UC@IJCAI 2026)*

---

## Overview

Multilingual pre-trained language models (PLMs) face persistent challenges with low-resource languages that use non-Latin scripts, due to high out-of-vocabulary (OOV) rates and subword fragmentation stemming from Latin-centric tokenization.

**VEXMLM** is a vocabulary-extended variant of [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base) specifically optimized for the Ge'ez-script languages **Amharic** and **Tigrinya**, and extended via task-specific fine-tuning to **19 low-resource African languages**.

### Key Contributions

- **Principled vocabulary expansion**: A language-specific SentencePiece tokenizer trained on curated monolingual corpora augments XLM-R's vocabulary with **30,000 Ge'ez-derived subword tokens**, initialized via mean initialization over the source embedding space.
- **Two-stage training strategy**: Continued masked language modeling (MLM) pretraining followed by supervised fine-tuning on QA, NER, and sentiment analysis — enabling cross-lingual transfer to 17 additional low-resource languages.
- **Comprehensive evaluation**: Intrinsic metrics (parity, fertility, compression, OOV rate) and extrinsic benchmarks (NER, SA, QA) across 19 African languages.

---

## Results

### Question Answering (QA)

| Model    | Exact Match | F1   |
|----------|-------------|------|
| XLM-R    | 0.66        | 0.78 |
| Glot500  | 0.74        | 0.78 |
| **VEXMLM**   | **0.87**    | **0.90** |

### Sentiment Analysis (SA)

| Model    | Accuracy |
|----------|----------|
| XLM-R    | 0.77     |
| Glot500  | 0.46     |
| **VEXMLM**   | **0.80** |

### NER — OOV Word Accuracy (11 African Languages)

| Model    | OOV Word Accuracy |
|----------|--------------------|
| XLM-R    | 81.4%              |
| **VEXMLM**   | **94.3%**      |

---

## Supported Languages

VEXMLM covers **19 low-resource African languages**, with core optimization for:

| Language  | Script | ISO Code |
|-----------|--------|----------|
| Tigrinya  | Ge'ez  | `tir`    |
| Amharic   | Ge'ez  | `amh`    |

Extended cross-lingual transfer to 17 additional African languages via fine-tuning.

---

## Model Architecture

```
XLM-R (base)
    └── Vocabulary Expansion (+30,000 Ge'ez subword tokens)
            └── Mean embedding initialization
                    └── Stage 1: Continued MLM Pretraining
                            └── Stage 2: Task-Specific Fine-Tuning
                                    ├── Question Answering (QA)
                                    ├── Named Entity Recognition (NER)
                                    └── Sentiment Analysis (SA)
```

- **Base model**: `FacebookAI/xlm-roberta-base`
- **Tokenizer**: SentencePiece, trained on curated Amharic + Tigrinya monolingual corpora
- **Vocabulary extension**: 30,000 new Ge'ez-derived subword tokens
- **Embedding initialization**: Mean initialization over XLM-R source embedding space
- **Training framework**: Hugging Face Transformers, PyTorch

---

## Installation

```bash
pip install transformers torch sentencepiece
```

---

## Usage

### Load the Model

```python
from transformers import AutoTokenizer, AutoModel

model_name = "Hailay/VEXMLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### Question Answering (Tigrinya)

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "Hailay/VEXMLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = "መን እዩ ፈጣሪ?"          # "Who is the creator?"
context  = "ኣምላኽ ፈጣሪ ኩሉ ዓለም እዩ።"   # "God is the creator of the whole world."

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

start = torch.argmax(outputs.start_logits)
end   = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs["input_ids"][0][start:end])
print(f"Answer: {answer}")
```

### Named Entity Recognition (Amharic)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "Hailay/VEXMLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
text = "አቶ አብይ አህመድ በአዲስ አበባ ይኖራሉ።"   # "Mr. Abiy Ahmed lives in Addis Ababa."
print(ner(text))
```

### Sentiment Analysis

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "Hailay/VEXMLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "እዚ መጽሓፍ ብጣዕሚ ጽቡቕ እዩ።"   # "This book is very good."
print(classifier(text))
```

---

## Repository Structure

```
VEXMLM/
├── tokenizer/
│   ├── train_sentencepiece.py       # SentencePiece tokenizer training
│   └── geez_vocab_extension.py      # Vocabulary expansion & embedding init
├── pretraining/
│   └── run_mlm.py                   # Continued MLM pretraining script
├── finetuning/
│   ├── run_qa.py                    # QA fine-tuning
│   ├── run_ner.py                   # NER fine-tuning
│   └── run_sa.py                    # Sentiment analysis fine-tuning
├── evaluation/
│   ├── intrinsic_eval.py            # Parity, fertility, compression, OOV metrics
│   └── extrinsic_eval.py            # NER, SA, QA benchmark evaluation
├── data/
│   └── README.md                    # Dataset sources and preprocessing notes
├── configs/
│   └── training_config.yaml         # Hyperparameters and training settings
├── requirements.txt
└── README.md
```

---

## Training Details

| Parameter              | Value                        |
|------------------------|------------------------------|
| Base model             | `xlm-roberta-base`           |
| Vocabulary extension   | +30,000 Ge'ez subword tokens |
| Embedding init         | Mean initialization          |
| Stage 1                | Continued MLM pretraining    |
| Stage 2                | Task-specific fine-tuning    |
| Tasks                  | QA, NER, Sentiment Analysis  |
| Languages (core)       | Amharic, Tigrinya            |
| Languages (total)      | 19 African languages         |
| Framework              | Hugging Face Transformers    |

---

## Citation

If you use VEXMLM in your research, please cite:

```bibtex
@inproceedings{teklehaymanot2026vexmlm,
  title     = {Expanding the Lexicon of Ge'ez Based African Languages:
               A Comparative Study of Amharic and Tigrinya},
  author    = {Teklehaymanot, Hailay Kidu},
  booktitle = {Proceedings of the Third Workshop on Language Models
               for Underserved Communities (LM4UC@IJCAI 2026)},
  year      = {2026}
}
```

---

## Related Work & Acknowledgements

- Base model: [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base) (Conneau et al., 2020)
- Tokenization baseline: [Glot500](https://huggingface.co/cis-lmu/glot500-base) (ImaniGooghari et al., 2023)
- This work was carried out at the [L3S Research Center](https://www.l3s.de), Leibniz Universität Hannover, and the University of Zurich.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
