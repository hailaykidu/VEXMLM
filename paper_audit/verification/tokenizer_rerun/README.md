# Tokenizer metrics re-run — verification only

Re-run during the correction (2026-09-24) to confirm the stored results. **Not a source of the
paper's numbers**: Table 2 and Figure 1 read the stored results `results/spmerge_tokenizer_metrics.json`
(GitHub) and `results/provenance/tokenizer_metrics_2026-08-15.json` (earlier run), which this re-run
reproduces exactly.

`tokenizer_metrics.json` was produced by:

```bash
python3 evaluation/run_intrinsic.py \
  --tokenizer xlm-roberta-base cis-lmu/glot500-base checkpoints/vexmlm-stage1-spm \
  --corpus amh=datasets/processed/amharic.dev.txt tir=datasets/processed/tigrinya.dev.txt \
  --oov-words amh=datasets/processed/amharic.oov.txt tir=datasets/processed/tigrinya.oov.txt \
  --base-vocab-size 250002 \
  --out paper_audit/verification/tokenizer_rerun/tokenizer_metrics.json
```

- Repository commit: `1e56e55` (`fix/deterministic-evaluation`).
- Tokenizers: `xlm-roberta-base`, `cis-lmu/glot500-base` (Hugging Face Hub), and the
  released VEXMLM SP-Merge tokenizer (`checkpoints/vexmlm-stage1-spm`, vocabulary 280,002).
- Evaluation text: `datasets/processed/{amharic,tigrinya}.dev.txt` (2,588 / 2,811 sentences)
  and `datasets/processed/{amharic,tigrinya}.oov.txt` (2,414 / 3,021 words), written by
  `datasets/prepare.py` (`datasets/processed/preparation_report.json`).
  They are held out from the MLM pretraining corpus; the SentencePiece tokenizers were
  trained on the train split only (`tokenizer/artifacts/*/manifest.json`), and no dev
  line occurs in the train split. The pretraining corpus has unresolved provenance and
  an unknown licence; see `docs/DATASET_PROVENANCE.md`.
- `--parallel` was omitted because the two dev sets are not sentence-aligned. The
  `parity` fields in the JSON are therefore marked `"valid": false` and are not reported.

Consistency: every XLM-R and SP-Merge field equals `results/spmerge_tokenizer_metrics.json`
(commit `d825163`), and every XLM-R and Glot500 field equals an earlier run of the same
command on the same files (52/52 fields identical in each comparison).
