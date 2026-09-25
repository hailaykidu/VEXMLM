# Composition of the Table 3 / Table 5 word set — verification only

Run during the correction (2026-09-24) to answer a review question. **Not a source of any number
in the paper.**

`word_set_composition.json` is produced by

```bash
HF_HUB_OFFLINE=1 python3 paper_audit/verification/oov_set_analysis/oov_set_analysis.py
```

It applies `evaluation/run_ner_oov.py::oov_word_set` (baseline `xlm-roberta-base`, reference
`checkpoints/vexmlm-expanded-spm`) to the Tigrinya NER test split and counts gold tags over the
same first-subword positions the metric scores. No model is run. The script asserts that the
word-type count (3,491) and token totals (7,194 / 3,624) equal
`results/ablation/full_vexmlm-seed42.json`.

Finding: every one of the 3,491 word types enters the set through the third clause (XLM-R needs
more pieces than the expanded tokenizer); none is `<unk>` or fails to round-trip. The paper
therefore calls these words "fragmented by XLM-R" rather than out-of-vocabulary. Entity tokens
make up 13.4% of the fragmented subset and 19.6% of the rest.
