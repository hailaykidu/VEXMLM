# Dataset Card — Tigrinya NER

| Field | Value |
|---|---|
| Key | `tigrinya_ner` |
| Task | ner |
| Languages | tir |
| Version | Yohannes and Amagasa (2022) |
| Licence | see source repository |
| Source | https://github.com/mehari-eng/Tigrinya-NER |
| Hub ID | **none — local copy** |
| Imported from | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/tigrinya` |
| Imported on | 2026-08-14T20:44:48 |

## Checksums (source files)

| Split | Bytes | SHA-256 | Verified |
|---|---|---|---|
| train | 1,251,673 | `01a9968d78ce30c9c3ca84ab290ea192` | ✅ |
| validation | 156,134 | `e3f86c999f177e1fe7d3233f9b55e72e` | ✅ |
| test | 152,957 | `19fa30537db6b88addf254b1fbaf5cb1` | ✅ |

## Notes

Tigrinya is NOT covered by MasakhaNER v1 or v2; this is a separate resource. Splits: 4,562/570/571 sentences (88,102/11,003/10,818 tokens). Tags: PER, ORG, LOC, DATE, MISC. NOTE: the train split contains one malformed tag ('B-LO' at line 1350, evidently a truncated 'B-LOC'); see docs/DATASET_PROVENANCE.md for how it is handled.

## Split statistics

| Split | Rows | SHA-256 (32) | Mean chars |
|---|---|---|---|
| train | 4,562 | `cad6540e780230010482b65d1fc333da` | 86.35 |
| validation | 570 | `5db216697d0deb535c1127738c0c72c5` | 86.38 |
| test | 571 | `5eac7a735145525faa78ed3453ffe637` | 84.34 |

## Citation

```bibtex
@article{yohannes2022tigrinya,
  title={Named Entity Recognition for Tigrinya},
  author={Yohannes, Hailemariam Mehari and Amagasa, Toshiyuki},
  year={2022},
  note={https://github.com/mehari-eng/Tigrinya-NER}
}
```
