# EXLMR — Vocabulary-Extended XLM-R for Tigrinya Sentiment Classification

EXLMR ("Extended XLM-R") adapts `xlm-roberta-base` to Tigrinya by extending its
subword vocabulary with Tigrinya-specific tokens before fine-tuning it as a
binary sentiment classifier (`negative` / `positive`) on Tigrinya text.

## Pipeline

The whole pipeline runs from a single script, [EXLMR.py](EXLMR.py), in two stages:

### 1. Vocabulary expansion

- Base model/tokenizer: `xlm-roberta-base`
- New tokens are loaded from [vocab.json](vocab.json) (30,522 candidate Tigrinya tokens)
  and added to the tokenizer with `tokenizer.add_tokens()`, which silently
  skips any token already present in the base vocabulary — **30,145** were
  actually new.
- The model's token embedding matrix is resized to match
  (final vocab size: **280,147**).
- Each new token's embedding is initialized with a mixed strategy: the
  average of a random-normal vector and the mean of all existing (pretrained)
  token embeddings — this gives new tokens a reasonable starting point rather
  than pure noise.
- The vocab-extended (not yet fine-tuned) tokenizer + model are saved to
  `/homes/neumann/teklehaymanot/EXLMR_Model`.

### 2. Fine-tuning

- Task: binary sequence classification, `{0: "negative", 1: "positive"}`.
- Data: [cleaned_train.csv](cleaned_train.csv) / [cleaned_test.csv](cleaned_test.csv)
  — semicolon-delimited `label;text` rows, Tigrinya (Ge'ez script) text.
  49,373 train rows / 3,999 test rows.
- Hyperparameters: 3 epochs, max sequence length 128, train batch size 16,
  eval batch size 8, learning rate 1e-5, weight decay 0.01.
- Best checkpoint (by weighted F1) is kept via `load_best_model_at_end`.
- If a previous run was preempted, model weights (not optimizer state — see
  note in `EXLMR.py` re: `CVE-2025-32434` on torch < 2.6) are resumed from the
  last checkpoint in `finetuned_model/`.
- Final fine-tuned model + tokenizer are saved to
  `/homes/neumann/teklehaymanot/finetuned_model`.

## Running it

```bash
sbatch EXLMR.sh
```

Requests 1× A100-80GB on the `standby` partition, 32 CPUs, 120G RAM. Logs go
to `Ex.out` / `Ex.err`.

## Latest run (job 50530, 2026-07-15)

Completed successfully in 30m19s, no errors. Final epoch-3 eval metrics:

| Metric | Value |
|---|---|
| eval_loss | 0.7075 |
| accuracy | 80.3% |
| F1 (weighted) | 0.802 |
| precision (weighted) | 0.811 |
| recall (weighted) | 0.803 |

## Files

| Path | Purpose |
|---|---|
| `EXLMR.py` | Main script: vocab expansion + fine-tuning |
| `EXLMR.sh` | SLURM submit script |
| `vocab.json` | Candidate Tigrinya tokens to add to XLM-R's vocabulary |
| `train.csv` / `test.csv` | Raw labeled data |
| `cleaned_train.csv` / `cleaned_test.csv` | Cleaned data actually used for training |
| `Ex.out` / `Ex.err` | Latest run's stdout/stderr logs |

Outputs (not in this directory):
- `/homes/neumann/teklehaymanot/EXLMR_Model` — vocab-extended base model, pre-fine-tuning
- `/homes/neumann/teklehaymanot/finetuned_model` — final fine-tuned classifier
