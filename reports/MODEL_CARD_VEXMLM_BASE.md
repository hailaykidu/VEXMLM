# Model Card: VEXMLM Base

The released model is on Hugging Face: https://huggingface.co/Hailay/VEXMLM
(checkpoint `checkpoints/vexmlm-stage1-spm`). Every value below is taken from the stored run
records named in the Source column.

## Model details

| | Value | Source |
|---|---|---|
| Base model | `xlm-roberta-base` | `results/provenance/expansion_manifest_spm.json` |
| Architecture | 12 layers, hidden size 768, 12 attention heads, GELU | `configs/base.yaml`; checkpoint `config.json` |
| Parameters | 301,365,186 | Hugging Face model card |
| Vocabulary | 280,002 = 250,002 (XLM-R) + 30,000 added | `results/provenance/expansion_manifest_spm.json` |
| Added pieces | 14,189 Amharic-only, 14,279 Tigrinya-only, 1,532 shared | `results/provenance/new_tokens*.json` |
| Integration | merged into XLM-R's SentencePiece model (`sentencepiece_merge`) | `results/provenance/expansion_manifest_spm.json` |
| Initialization of new embeddings | mean of all pretrained embeddings (`global_mean`) | `results/provenance/expansion_manifest_spm.json` |
| Languages | Amharic (am), Tigrinya (ti) | — |

Candidate pieces come from language-specific SentencePiece unigram models (Amharic 32,000,
Tigrinya 50,000 pieces, character coverage 0.9995, NFC; `results/provenance/tokenizer_manifest_*.json`):
per language, the 20,000 most frequent pieces absent from XLM-R, interleaved by rank and
deduplicated to 30,000.

## Continued pretraining (Stage 1)

| Setting | Value | Source |
|---|---|---|
| Objective | masked language modelling, masking probability 0.15 | `configs/base.yaml` |
| Epochs | 60 configured; checkpoint with the lowest validation loss released (epoch 56, step 24,808) | `results/provenance/training_args.json`, `stage1_spm_trainer_state_best.json` |
| Best validation loss | 3.7120 | `results/provenance/stage1_spm_trainer_state_best.json` |
| Final re-evaluation of the released checkpoint | loss 3.7299, perplexity 41.67 | `results/provenance/stage1_summary_spm.json` |
| Optimizer | AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8), learning rate 5e-5, linear decay after 6% warmup, weight decay 0.01 | `results/provenance/training_args.json` |
| Batch size / sequence length | 32 / 256 (block-chunked) | `results/provenance/training_args.json`, `configs/base.yaml` |
| Gradient clipping / precision | 1.0 / bf16 | `results/provenance/training_args.json` |
| Hardware | 1× NVIDIA A100 80GB PCIe, CUDA 11.8, PyTorch 2.5.1+cu118 | `results/provenance/stage1_summary_spm.json` |

## Training data

The pretraining text is one Amharic and one Tigrinya monolingual file (200,002 and 200,000
lines) taken from a prepared local collection whose original source is not documented. The
files match no named corpus available to the authors, and manual samples contain religious
translations alongside general web prose; the licence is therefore unknown and the text is not
redistributed. An earlier attribution to HornMT is ruled out (HornMT has about 2,030 sentence
pairs, and no line overlaps). See `docs/DATASET_PROVENANCE.md` and `docs/MLM_CORPUS_ANALYSIS.md`.

| | Amharic | Tigrinya | Source |
|---|---|---|---|
| Lines kept after NFC normalization, length / script filtering and deduplication | 129,402 | 140,583 | `results/provenance/preparation_report.json` |
| Training / development lines (2% held out) | 126,814 / 2,588 | 137,772 / 2,811 | `results/provenance/preparation_report.json` |
| Lines after language resampling (α = 0.5) | 129,552 | 135,034 | `results/provenance/stage1_spm_data.log.txt` |

Stage 1 holds out a further 1% (2,645 lines) for validation and trains on 14,161 blocks of 256
tokens (`results/provenance/stage1_spm_data.log.txt`).

## Evaluation

Tokenizer metrics on the held-out development text (`results/spmerge_tokenizer_metrics.json`):

| Metric | Language | XLM-R | VEXMLM |
|---|---|---|---|
| Fertility (tokens per word) ↓ | Amharic | 2.0692 | 1.4888 |
| | Tigrinya | 3.1300 | 1.6928 |
| Compression (characters per token) ↑ | Amharic | 2.2950 | 3.1896 |
| | Tigrinya | 1.4591 | 2.6979 |
| OOV word round-trip ↑ | Amharic | 1.0000 | 1.0000 |
| | Tigrinya | 0.9954 | 0.9987 |

Parity is not reported: it needs a sentence-aligned parallel corpus, which is not available.

Downstream results, mean ± standard deviation over seeds 42–46 (`results/downstream_task_metrics.csv`).
NER and SA are scored on the test splits, QA on the development splits:

| Task | Dataset | Metric | VEXMLM |
|---|---|---|---|
| NER | MasakhaNER Amharic | Accuracy / Macro-F1 / Entity-F1 | 0.9413 ± 0.0026 / 0.7423 ± 0.0122 / 0.6347 ± 0.0148 |
| NER | Tigrinya NER | Accuracy / Macro-F1 / Entity-F1 | 0.9515 ± 0.0005 / 0.8219 ± 0.0069 / 0.7282 ± 0.0079 |
| QA | AmQA | EM / F1 | 32.57 ± 0.77 / 48.85 ± 0.96 |
| QA | TIGQA | EM / F1 | 2.39 ± 0.82 / 9.76 ± 0.97 |
| SA | AfriSenti (Amharic) | Accuracy / Macro-F1 | 0.4978 ± 0.0331 / 0.4971 ± 0.0193 |

A single-seed XLM-R baseline is in `results/baselines/xlmr_seed42/`. Ablation on Tigrinya NER OOV
accuracy (`results/table5_ablation.csv`): XLM-R 94.57 ± 0.16; vocabulary expansion with random
initialization 87.04 ± 0.20; with mean initialization 87.63 ± 0.14; plus continued pretraining
95.66 ± 0.09.

## Limitations

- Downstream benefit is task-dependent: NER improves over the single-seed XLM-R baseline, QA is
  lower and sentiment is comparable.
- Vocabulary expansion alone lowers OOV accuracy; continued pretraining is required.
- The pretraining text has undocumented origin and an unknown licence.
- The XLM-R downstream baseline is a single seed.

## Citation

```bibtex
@inproceedings{teklehaymanot2026vexmlm,
  title     = {Vocabulary Expansion for Low-Resource African Languages:
               A Case Study in Amharic and Tigrinya},
  author    = {Teklehaymanot, Hailay Kidu and Yadeta, Debela Desalegn and Nejdl, Wolfgang},
  booktitle = {Proceedings of the Workshop on Language Models for Underserved Communities (LM4UC), IJCAI 2026},
  year      = {2026}
}
```

Corrected version of the LM4UC 2026 paper published as *Expanding the Lexicon of Ge'ez Based
African Languages: A Comparative Study of Amharic and Tigrinya*.

## Changelog

| Date | Change |
|---|---|
| 2026-08-15 | First model card |
| 2026-09-25 | Rewritten for the released checkpoint `vexmlm-stage1-spm` (the previous card described an earlier 10-epoch run and a 576,520-entry vocabulary); data statement corrected; title updated |
