#!/usr/bin/env python3
"""Model-card generator for the VEXMLM downstream Hugging Face repositories.

Cards carry only values that come from the verified five-seed evaluation, and they
separate the benchmark measurement from interactive inference so a reader cannot
mistake a demo prediction for a reported score.

No scheduler metadata, private path, credential, or AI attribution is emitted.
Personal names appear only inside the formal BibTeX citation.
"""

from __future__ import annotations

CITATION = """```bibtex
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

Accepted at the LM4UC Workshop, IJCAI 2026."""

GITHUB = "https://github.com/hailaykidu/VEXMLM"

# Verified five-seed results. Percent-scaled for readability; the underlying
# records are in results/spm_stage2/ in the official repository.
BENCH = {
    "tigrinya_ner": [("Entity-F1", "**72.82 ± 0.79**"), ("Macro-F1", "82.19 ± 0.69"),
                     ("Accuracy", "95.15 ± 0.05")],
    "masakhaner_amh": [("Entity-F1", "**63.47 ± 1.48**"), ("Macro-F1", "74.23 ± 1.22"),
                       ("Accuracy", "94.13 ± 0.26")],
    "amqa": [("Exact Match", "**32.57 ± 0.77**"), ("F1", "**48.85 ± 0.96**")],
    "tigqa": [("Exact Match", "**2.39 ± 0.82**"), ("F1", "**9.76 ± 0.97**")],
    "tiquad": [("Exact Match", "**50.24 ± 0.48**"), ("F1", "**58.90 ± 0.66**")],
    "afrisenti": [("Accuracy", "**49.78 ± 3.31**"), ("Macro-F1", "**49.71 ± 1.93**")],
}

PIPELINE = {"ner": "token-classification", "qa": "question-answering",
            "sa": "text-classification"}
ARCH = {"ner": "XLMRobertaForTokenClassification",
        "qa": "XLMRobertaForQuestionAnswering",
        "sa": "XLMRobertaForSequenceClassification"}
AUTOCLASS = {"ner": "AutoModelForTokenClassification",
             "qa": "AutoModelForQuestionAnswering",
             "sa": "AutoModelForSequenceClassification"}
LANGCODE = {"Amharic": "am", "Tigrinya": "ti"}

DEMO = {
    "ner": "enter arbitrary {lang} text and inspect the predicted entity spans",
    "qa": "supply a {lang} context and question, and read back the extracted span",
    "sa": "enter arbitrary {lang} text and inspect the predicted sentiment label",
}

NOTES = {
    "tigrinya_ner": "Labels cover PER, ORG, LOC, DATE and MISC in BIO format (11 classes).",
    "masakhaner_amh": "Labels cover PER, ORG, LOC and DATE in BIO format (9 classes).",
    "amqa": "Extractive QA: the answer is always a span copied from the supplied context.",
    "tigqa": """**This checkpoint is weak in absolute terms.** Exact Match of 2.39 is
near zero. The TIGQA test split contains only 67 questions, making the metric both
hard and high-variance. It is published for completeness and for reproducibility of
the paper's evaluation, not as a usable Tigrinya QA system. For Tigrinya extractive
QA prefer [`Hailay/VEXMLM-TiQuAD`](https://huggingface.co/Hailay/VEXMLM-TiQuAD).""",
    "tiquad": """**TiQuAD is a supplementary task** in the VEXMLM paper -- a diagnostic
evaluation rather than a reported benchmark. Its 926 test questions make it a more
stable measurement than TIGQA's 67, and it is the stronger of the two Tigrinya QA
checkpoints.""",
    "afrisenti": """Three classes: `negative`, `neutral`, `positive`.

**Accuracy of 49.78 on a three-class task is weak.** The AfriSenti Amharic split has
a majority class that differs between the training and test portions, so a model that
learns the training prior is systematically miscalibrated on test. Treat this as a
reproduction of the paper's evaluation rather than a production classifier.""",
}

EXAMPLES = {
    "ner": '''```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

repo = "Hailay/{repo}"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="seed-42")
model = AutoModelForTokenClassification.from_pretrained(repo, subfolder="seed-42")
model.eval()

words = "{text}".split()
enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)

with torch.no_grad():
    pred = model(**enc).logits.argmax(-1)[0].tolist()

seen = set()
for p, w in zip(pred, enc.word_ids(0)):
    if w is None or w in seen:
        continue
    seen.add(w)
    print(words[w], "->", model.config.id2label[p])
```''',
    "qa": '''```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

repo = "Hailay/{repo}"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="seed-42")
model = AutoModelForQuestionAnswering.from_pretrained(repo, subfolder="seed-42")
model.eval()

question = "{question}"
context = "{context}"
enc = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    out = model(**enc)

start = out.start_logits.argmax()
end = out.end_logits.argmax()
print(tokenizer.decode(enc.input_ids[0][start:end + 1], skip_special_tokens=True))
```''',
    "sa": '''```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "Hailay/{repo}"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="seed-42")
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="seed-42")
model.eval()

enc = tokenizer("{text}", return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    logits = model(**enc).logits

print(model.config.id2label[logits.argmax(-1).item()])
```''',
}

SAMPLE_TEXT = {"Tigrinya": "ኤርትራ ኣብ ቀርኒ አፍሪቃ እትርከብ ሃገር እያ።",
               "Amharic": "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት።"}
SAMPLE_QA = {"Amharic": ("ኢትዮጵያ የምትገኘው በየትኛው አህጉር ነው?", "ኢትዮጵያ በአፍሪካ አህጉር የምትገኝ ሀገር ናት።"),
             "Tigrinya": ("ኤርትራ ኣብ ኣየናይ ክፍለ ዓለም ትርከብ?", "ኤርትራ ኣብ አፍሪቃ ክፍለ ዓለም እትርከብ ሃገር እያ።")}
SA_TEXT = "በጣም ጥሩ ነው!"


def build_card(dataset_key: str, meta: dict) -> str:
    """Render the model card for one downstream task repository."""
    task, lang, repo = meta["task"], meta["lang"], meta["repo"]
    bench = "\n".join(f"| {k} | {v} |" for k, v in BENCH[dataset_key])
    demo = DEMO[task].format(lang=lang)

    if task == "ner":
        example = EXAMPLES["ner"].format(repo=repo, text=SAMPLE_TEXT[lang])
    elif task == "qa":
        q, c = SAMPLE_QA[lang]
        example = EXAMPLES["qa"].format(repo=repo, question=q, context=c)
    else:
        example = EXAMPLES["sa"].format(repo=repo, text=SA_TEXT)

    title = repo.replace("VEXMLM-", "VEXMLM — ").replace("-", " ")

    return f"""---
language:
- {LANGCODE[lang]}
license: apache-2.0
library_name: transformers
pipeline_tag: {PIPELINE[task]}
tags:
- {PIPELINE[task]}
- xlm-roberta
- vexmlm
- geez
- low-resource
base_model: Hailay/VEXMLM
---

# {title}

{lang} **{PIPELINE[task].replace('-', ' ')}** fine-tuned from
[`Hailay/VEXMLM`](https://huggingface.co/Hailay/VEXMLM), the vocabulary-extended
XLM-R for Ge'ez-script languages.

Official implementation: **{GITHUB}**

| | |
|---|---|
| Task | {PIPELINE[task]} |
| Dataset | {meta['dataset']} |
| Language | {lang} |
| Architecture | `{ARCH[task]}` |
| Base model | `Hailay/VEXMLM` |
| Vocabulary | 280,002 |
| Seeds published | 42, 43, 44, 45, 46 |

{NOTES[dataset_key]}

## Five-seed benchmark evaluation

Fine-tuned independently under seeds 42–46 with a single configuration on one
A100-class GPU, then evaluated on the dataset's held-out **test** split. Values are
mean ± standard deviation over the five runs.

{bench}

These are the paper's verified results, produced by the five-seed evaluation
described above.

### Benchmark evaluation vs. interactive inference

**Benchmark evaluation** is the five-seed measurement on the held-out test split,
reported in the table above. It is the only source of the numbers quoted here.

**Interactive inference** is what the example below does: {demo}. Predictions on
arbitrary input are demonstrations. They neither produce nor reproduce the benchmark
score, and no metric should be inferred from them.

## Repository layout

Five independently fine-tuned checkpoints, one per seed. The benchmark score is the
mean over all five; **no single seed is "the five-seed model."**

```
seed-42/  seed-43/  seed-44/  seed-45/  seed-46/
```

Select one with the `subfolder` argument, as below.

## Usage

{example}

## Fine-tuning

| Hyperparameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 32 |
| Epochs | 4 |
| Learning rate | 2e-5 |
| LR schedule | Linear decay, 10% warmup |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8) |
| Precision | bf16 |
| Trainable parameters | All |

Runs are bit-reproducible: `enable_full_determinism`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `dataloader_num_workers=0`.

## Limitations

- Fine-tuned for {lang} on {meta['dataset']} only; behaviour on other languages,
  domains, or label schemes is not characterised.
- The base model covers Amharic and Tigrinya; other Ge'ez-script languages were not
  part of pretraining.
- Pretraining corpora are drawn largely from religious and news domains, and the
  model may reflect those distributions and any biases within them.
- Single-configuration study: no hyperparameter search was performed, and the
  paper's baseline comparisons are single-seed.

## Reproducibility

Fine-tuning launcher, evaluation code, and per-run result records are in the
official repository: **{GITHUB}**

```bash
sbatch scripts/slurm_stage2_spm_seeds.sh    # 6 tasks x 5 seeds
python3 evaluation/export_spm_results.py    # regenerates the metrics table
```

## Citation

{CITATION}

## License

Apache 2.0, following `xlm-roberta-base`.
"""
