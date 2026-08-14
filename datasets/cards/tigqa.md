# Dataset Card — TIGQA — Tigrinya Question Answering

| Field | Value |
|---|---|
| Key | `tigqa` |
| Task | qa |
| Languages | tir |
| Version | TIGQA-1.0 |
| Licence | CC BY 4.0 |
| Source | https://zenodo.org/records/11423987 |
| Hub ID | **none — local copy** |
| Imported from | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/tigqa_squad` |
| Imported on | 2026-08-14T20:44:48 |

## Checksums (source files)

| Split | Bytes | SHA-256 | Verified |
|---|---|---|---|
| train | 466,419 | `920aee6eb379af97d17cd51d830a2f69` | ✅ |
| validation | 50,021 | `4d79d9bbf7fbb8e2f5cb8e93cf137401` | ✅ |
| test | 58,614 | `999971f826d7521280888e2be4027e78` | ✅ |

## Notes

SQuAD-format extractive QA, stored flat (title/context/qas, with no 'paragraphs' level). 797 question-answer pairs over 365 contexts, split 80/10/10 by context with seed 42: 644/67/86 questions over 292/36/37 contexts. 1,039 abstractive and 272 unanswerable items were excluded upstream, since answer_start == -1 cannot be scored by span extraction; no negative offsets remain. See docs/DATASET_PROVENANCE.md.

## Split statistics

| Split | Rows | SHA-256 (32) | Mean chars |
|---|---|---|---|
| train | 644 | `54e28676ae94c307cb2a2d06b12321a1` | 29.21 |
| validation | 67 | `7525f4eeccb645f77bdc1b2cd623f79f` | 28.84 |
| test | 86 | `a13983f5e58b4b70e93a72d71036a3b1` | 32.38 |

## Citation

```bibtex
@inproceedings{teklehaymanot2024tigqa,
  title={TIGQA: An Expert-Annotated Question-Answering Dataset in Tigrinya},
  author={Teklehaymanot, Hailay and others},
  year={2024},
  note={Zenodo: https://zenodo.org/records/11423987}
}
```
