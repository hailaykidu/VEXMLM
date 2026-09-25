# QA answer-span mapping check — verification only

Run during the correction (2026-09-24). **Not a source of any number in the paper**; its result
is not reported there.

Question: does the SP-Merge tokenizer corrupt the answer-span labels that
`finetuning/qa/run_qa.py::prepare_train` derives from `offset_mapping`?

```bash
HF_HUB_OFFLINE=1 python3 paper_audit/verification/qa_span_check/qa_span_check.py
```

For the first 50 answered examples of each split (AmQA and TigQA, train and validation;
200 per tokenizer), the script runs the repository's own `prepare_train`, rebuilds the
labelled answer as `context[offset[start][0]:offset[end][1]]`, and compares it to the gold
answer. Settings match fine-tuning: max length 256, stride 128. Output:
`qa_span_check_report.json`.

| Tokenizer | Exact | Exact except leading space | Span runs past answer | Wrong / missing |
|---|---|---|---|---|
| VEXMLM SP-Merge (`checkpoints/vexmlm-stage1-spm`) | 44 | 148 | 8 | 0 |
| XLM-R (`xlm-roberta-base`) | 192 | 0 | 8 | 0 |

- Leading space: SP-Merge pieces carry the preceding space in their offsets.
  `normalize_answer` strips whitespace before EM/F1, so scoring is unaffected.
- Span runs past answer (8/200 for each tokenizer), two causes:
  - 4/200: the gold answer text ends in a space (e.g. `'ፕሬዚዳንት '`), so `end_char` falls
    on the next token and the label includes the next word. Same 4 examples for both
    tokenizers.
  - 4/200 per tokenizer: the last subword extends past the gold answer (e.g. `23ኛ` → `23ኛው`,
    `ናእዳን ሞገስን` → `ናእዳን ሞገስን፣`). Different examples per tokenizer (SP-Merge adds
    `ብሪታንያ` → `የብሪታንያ` and `፴፪ኛ` → `፴፪ኛዋ`; XLM-R adds two TigQA cases).
- Every answer lies inside a window, and every gold answer occurs at its `answer_start`.

Conclusion: span labelling is not the cause of the lower VEXMLM QA scores.
