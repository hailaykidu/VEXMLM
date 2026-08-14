# Dataset Card — MasakhaNER — Amharic

| Field | Value |
|---|---|
| Key | `masakhaner_amh` |
| Task | ner |
| Languages | amh |
| Version | MasakhaNER v1 (amh) |
| Licence | CC BY-NC 4.0 |
| Source | https://github.com/masakhane-io/masakhane-ner |
| Hub ID | masakhane/masakhaner2 |
| Imported from | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/amharic` |
| Imported on | 2026-08-14T20:44:48 |

## Checksums (source files)

| Split | Bytes | SHA-256 | Verified |
|---|---|---|---|
| train | 398,840 | `bc82370a67d38ddfc97d042bfeabe874` | ✅ |
| validation | 57,855 | `1a5a5d54e87b73b7e41eba0dcad13e07` | ✅ |
| test | 114,626 | `af20aa41753c19fc84b9e938afcaae1d` | ✅ |

## Notes

Official splits used as released: 1,750/250/500 sentences (25,819/3,749/7,449 tokens). CoNLL BIO tags: PER, ORG, LOC, DATE. LICENCE IS NON-COMMERCIAL (CC BY-NC 4.0), unlike most of the other datasets here; underlying news text carries per-site licences. Any model fine-tuned on it inherits a non-commercial constraint. See docs/DATASET_REDISTRIBUTION.md.

## Split statistics

| Split | Rows | SHA-256 (32) | Mean chars |
|---|---|---|---|
| train | 1,750 | `074b65e5c520b40489cf18a8aa190d41` | 71.41 |
| validation | 250 | `a43df9a483e5cb05751c2c8ed2033a94` | 72.46 |
| test | 500 | `06a4fda1031400da18f60eeddff145b8` | 72.19 |

## Citation

```bibtex
@inproceedings{adelani-etal-2021-masakhaner,
  title={MasakhaNER: Named Entity Recognition for African Languages},
  author={Adelani, David Ifeoluwa and others},
  journal={TACL},
  year={2021}
}
```
