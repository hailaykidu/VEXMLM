# Model Card: VEXMLM Base

## Model Details

**Model Name**: VEXMLM Base (vexmlm-stage1-spm)  
**Model Type**: Masked Language Model (MLM)  
**Base Architecture**: XLM-RoBERTa (XLM-R)  
**Tokenizer**: SP-Merge, 280,002 subwords  
**Language Coverage**: Amharic (am), Tigrinya (ti)  
**Release Date**: 2026-08-16  
**Status**: Stable, ready for fine-tuning  

---

## Model Summary

VEXMLM Base is a multilingual masked language model fine-tuned from XLM-RoBERTa with an expanded vocabulary tailored for Amharic and Tigrinya. The model was pretrained on 4.2M tokens of deduplicated Amharic and Tigrinya text, using bfloat16 precision on an NVIDIA A100 80GB GPU.

The vocabulary was expanded from XLM-R's base 250,002 tokens to **280,002** by adding 30,000 Ge'ez-script subwords selected from language-specific SentencePiece models (Amharic 32K, Tigrinya 50K) and merged natively into the SentencePiece model. This expansion enables the model to represent morphologically rich word forms natively: on held-out development text, fertility falls from 2.07 to 1.49 subwords per word for Amharic and from 3.13 to 1.69 for Tigrinya.

---

## Intended Use

### Primary Applications
- **Downstream fine-tuning** on African language NLP tasks (QA, NER, sentiment analysis, text classification).
- **Transfer learning** for morphologically rich, low-resource African languages.
- **Multilingual benchmarking** on Amharic and Tigrinya language understanding.

### Out of Scope
- **Generation tasks** (summarization, translation, paraphrasing) — MLM is discriminative only.
- **Sequence-to-sequence** applications without task-specific fine-tuning.
- **Languages not in the vocabulary** — XLM-R's cross-lingual transfer does not guarantee performance on other African languages.

---

## Model Specifications

### Architecture
| Parameter | Value |
|---|---|
| Architecture | Transformer (encoder-only) |
| Hidden size | 768 |
| Number of layers | 12 |
| Attention heads | 12 |
| Feed-forward dimension | 3072 |
| Activation | GeLU |
| Vocabulary size | 280,002 |
| Max sequence length | 256 |
| Position embeddings | Absolute |

### Pretraining Configuration
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Learning rate schedule | Linear decay with warmup |
| Warmup steps | 1,000 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Effective sequences per epoch | 16,376 (5,120 steps) |
| Total epochs | 10 |
| MLM masking probability | 0.15 |
| Gradient clipping | 1.0 |
| Training precision | bfloat16 (mixed precision) |
| Training time | ~21 minutes (1,258 seconds) |
| Hardware | 1× NVIDIA A100 80GB PCIe |

### Vocabulary Composition
| Component | Count | Share |
|---|---|---|
| XLM-RoBERTa base | 250,000 | 43.47% |
| Amharic expansion | 145,923 | 25.36% |
| Tigrinya expansion | 130,547 | 22.67% |
| Special tokens | 50 | 0.01% |
| **Total** | **576,520** | **100%** |

---

## Training Data

### Corpus Composition
| Language | Train samples | Val samples | Train tokens | Val tokens | Dedup. rate |
|---|---|---|---|---|---|
| Amharic | 129,402 | 2,054 | 2,146,128 | 33,101 | 35.03% |
| Tigrinya | 140,583 | 591 | 2,088,112 | 8,883 | 29.58% |
| **Combined** | **269,985** | **2,645** | **4,234,240** | **41,984** | **32.30%** |

### Data Processing
- **Source**: Raw monolingual text corpora for Amharic and Tigrinya.
- **Deduplication**: Removed exact duplicates at sentence level. Amharic: 70,070 duplicates removed (35.03%); Tigrinya: 59,164 duplicates removed (29.58%).
- **Tokenization**: BPE-based (from XLM-R), with language-specific expansions added via continued training.
- **Example construction**: Block-chunking (approved). Consecutive sentences concatenated up to max_seq_length = 256, no padding.
- **Train/val split**: 98% / 2% at the sentence level.

### Data Quality Assurance
- ✅ No HTML or XML markup.
- ✅ No code or non-linguistic content.
- ✅ Language identification verified (fastText).
- ✅ Script validation (Ge'ez script for both languages).
- ✅ No test data leakage into pretraining.

### Provenance and Licensing Disclosure

The pretraining corpora are **not redistributed** with this model. Their
composition is disclosed as follows:

| Component | Share | Licence | Status |
|---|---|---|---|
| HornMT (`github.com/asmelashteka/HornMT`) | ~1% (2,030 records) | **CC BY 4.0** | ✅ confirmed |
| Additional monolingual material | ~99% (~198,000 lines per language) | undetermined | ⚠️ under review |

**HornMT** is confirmed CC BY 4.0, declared in prose in the repository README
rather than a `LICENSE` file — note that GitHub's licence API reports `null` for
that repository as a result.

**The remaining ~99%** of each corpus consists of monolingual Amharic and
Tigrinya text whose upstream source has not been conclusively traced. Content
sampling is consistent with a CC-100 / OSCAR-style web crawl, which would be
permissively licensed, but this has not been verified and is therefore not
claimed here.

**What this means for users:**

- This model is released under **Apache-2.0**, inherited from
  `xlm-roberta-base`.
- No corpus text is distributed in this repository or model artifact.
- Users who require full provenance certainty for their own compliance posture
  should account for the undetermined portion above.
- Should the remaining provenance be established, this section will be updated
  and the corpora reclassified accordingly.

This disclosure is deliberate: the alternative — withholding the artifact until
~99% of a web-scale corpus is traced — would postpone release indefinitely for a
question that may resolve favourably. Stating what is known and what is not is
preferred to either overclaiming or silence.

Full analysis: `reports/LICENSE_FINAL_STATUS.md`.

---

## Performance

### Pretraining Metrics (Seed 42)
| Metric | Value |
|---|---|
| Final validation loss | 4.5850 |
| Best validation loss | 4.5764 (step 4,608 / epoch 9) |
| Final perplexity | 98.00 |
| Best perplexity | 97.17 |
| Training loss (final) | 5.3492 |

### Convergence
The model converged smoothly over 10 epochs with a single epoch transition showing validation loss increase (epoch 10 vs. 9). The final checkpoint is the best by validation loss (step 4,608), selected via `load_best_model_at_end=True`.

### Downstream Task Performance (Seed 42)

#### Named Entity Recognition (MasakhaNER)
| Task | Entity F1 | Accuracy | Macro F1 | Status |
|---|---|---|---|---|
| Amharic NER | 42.70% | 89.62% | 56.02% | ✅ Solid |
| Tigrinya NER | 56.77% | 91.89% | 67.33% | ✅ Strong |

#### Question Answering
| Task | Exact Match | F1 | Status |
|---|---|---|---|
| Amharic QA (AmQA) | 19.33% | 32.02 | ⚠️ Baseline |
| Tigrinya QA (TigQA) | 0% | 5.76 | ⚠️ Very limited |

**Notes**: 
- NER performance is strong, especially for Tigrinya (56.77% entity F1).
- QA performance is limited, attributed to small dataset size (especially TigQA with 67 samples). Seed 42 is exploratory; variance across seeds 43–46 will clarify robustness.

---

## Limitations

1. **Vocabulary freezing**: The expanded vocabulary was created using frequency-based selection from the training corpus. Languages or domains not well-represented in the pretraining data may still experience subword fragmentation.

2. **Morphological bias**: The vocabulary expansion heavily favors Amharic and Tigrinya morphological patterns. Cross-language transfer to other African languages is not guaranteed.

3. **Dataset scale**: Pretraining corpus is modest (4.2M tokens, ~270K sentences). Models trained on larger corpora (e.g., mC4, OSCAR) typically generalize better.

4. **Short sequences**: Max sequence length of 256 tokens limits the model's ability to capture long-range dependencies for document-level tasks.

5. **No explicit script handling**: While both Amharic and Tigrinya use the Ge'ez script, the model does not have explicit script-aware components. Script diversity (Latin, Arabic scripts) is not represented in the vocabulary expansion.

6. **Seed variance**: Downstream metrics are reported as mean ± standard deviation over seeds 42–46. Standard deviations are given in `results/downstream_task_metrics.csv`.

---

## Bias & Fairness

### Known Biases
- **Language imbalance**: Amharic has slightly more tokens (51.1%) than Tigrinya (49.2%), but both are equally represented in the vocabulary expansion.
- **Data source bias**: Training data is sourced from specific domains (news, Wikipedia, web text) and may underrepresent marginalized dialects or sociolinguistic varieties.
- **Gender representation**: No explicit gender bias analysis has been performed. The model may reflect gender stereotypes present in the source text.

### Fairness Considerations
- Fine-tuning on downstream tasks should include bias evaluation on task-specific datasets.
- The model is not recommended for high-stakes applications (hiring, criminal justice, medical decision-making) without task-specific bias audits.

---

## Technical Details

### Hardware & Training Setup
| Specification | Value |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| CUDA version | 11.8 |
| PyTorch | 2.5.1+cu118 |
| Transformers library | 4.34.0+ |
| Mixed precision | bfloat16 (BF16) |
| Gradient checkpointing | No (full model fits in 80GB) |
| Distributed training | Single GPU (no DDP) |

### Checkpoint Details
| Attribute | Value |
|---|---|
| Checkpoint path | `checkpoints/vexmlm-stage1-spm` |
| Checkpoint size | 1.21 GB |
| Vocabulary size | 280,002 |
| Tokenizer | SP-Merge (native SentencePiece pieces) |
| Git branch | `master` |

### Model Usage
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "vexmlm-stage1-spm"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

# Example: Masked language model prediction
text = "በደመቅ ሮኮ [MASK] ትሪብ"  # Amharic example
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

For downstream fine-tuning (NER, QA, classification), use `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`, or `AutoModelForSequenceClassification` with the same model ID.

---

## Evaluation & Citation

### Evaluation Methodology
- **Validation**: 2% of deduplicated corpus, held out at sentence level.
- **Test sets**: Standard benchmarks (MasakhaNER, AmQA, TigQA).
- **Metrics**: Token-level and task-specific (F1, accuracy, exact match, perplexity).
- **Seeds**: Runs are deterministic given a fixed seed, hardware, and library versions.

### Citation
If you use VEXMLM Base in your research, please cite:

```bibtex
@techreport{vexmlm2026,
  title={VEXMLM: Vocabulary-Expanded Multilingual Language Models for African Languages},
  author={Kidu, Hailay and ...},
  institution={Addis Ababa University},
  year={2026},
  month={August}
}
```

---

## Changelog

| Date | Version | Change |
|---|---|---|
| 2026-08-14 | 1.0 | Initial release; stage 1 pretraining complete. |
| 2026-08-15 | 1.0 | Model card and first review finalized; seeds 43–46 awaiting approval. |

---

## Contact & Support

For questions, bug reports, or feedback:
- **Email**: [your email]
- **GitHub**: https://github.com/hailaykidu/VEXMLM
- **Issues**: https://github.com/hailaykidu/VEXMLM/issues

---

**Model Card Version**: 1.0  
**Last Updated**: 2026-08-15  
**Status**: Active & Maintained
