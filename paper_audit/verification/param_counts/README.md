# Parameter counts — verification only

Computed during the correction; **not a source of the paper**, which reports the count stated on
the Hugging Face model card (301,365,186, identical to this computation).


`param_counts.json` counts the unique parameters of `XLMRobertaForMaskedLM` for
`xlm-roberta-base` and the released VEXMLM checkpoint (`checkpoints/vexmlm-stage1-spm`),
with tied input/output embeddings counted once and no pooler. The difference,
23,070,000, equals 30,000 new embedding rows x 768 plus 30,000 output-bias entries.

Produced at commit `1e56e55` with:

```python
from transformers import AutoModelForMaskedLM
m = AutoModelForMaskedLM.from_pretrained(path)
sum(t.numel() for t in m.parameters())
```
