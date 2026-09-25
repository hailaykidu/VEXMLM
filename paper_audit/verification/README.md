# Verification runs made during the correction

Nothing in this folder is a source of a number in the paper. Each item was run once to check a
stored result or answer a review question:

| Folder | What it checked | Outcome |
|---|---|---|
| `tokenizer_rerun/` | re-ran `evaluation/run_intrinsic.py` on the stored dev/OOV files | identical to `results/spmerge_tokenizer_metrics.json` and the earlier Glot500 run |
| `param_counts/` | counted parameters of the released checkpoint | identical to the model card (301,365,186) |
| `dataset_splits_recount.json` | re-counted split sizes with the fine-tuning loader | identical to `datasets/metadata/*.json` |
| `qa_span_check/` | QA answer-span labels with both tokenizers | span labels reconstruct the gold answers; not reported in the paper |
| `oov_set_analysis/`, `oov_set_composition/` | which clause of the OOV rule selects each word | all 3,491 words via the "more pieces" clause; not reported in the paper |
