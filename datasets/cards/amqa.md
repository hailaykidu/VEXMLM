# Dataset Card — AmQA — Amharic Question Answering

| Field | Value |
|---|---|
| Key | `amqa` |
| Task | qa |
| Languages | amh |
| Version | SQuAD v2.0 format |
| Licence | MIT |
| Source | https://github.com/semantic-systems/amharic-qa |
| Hub ID | **none — local copy** |
| Imported from | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/amqa` |
| Imported on | 2026-08-14T20:44:48 |

## Checksums (source files)

| Split | Bytes | SHA-256 | Verified |
|---|---|---|---|
| train | 1,747,839 | `c2a76dee46332964cfa9b2b1d7f0a21a` | ✅ |
| validation | 497,479 | `af8497b3753aa3386c4e515712995dfb` | ✅ |
| test | 254,291 | `2196b4bb88d89db4d952832277825101` | ✅ |

## Notes

Official train/dev/test splits used as released: 1,723/600/299 questions. 20 answer offsets were corrected upstream and one dict-wrapped paragraph list normalized; no questions dropped.

## Split statistics

| Split | Rows | SHA-256 (32) | Mean chars |
|---|---|---|---|
| train | 1,723 | `745ff98118b537548ea17f5103a5eabd` | 45.16 |
| validation | 600 | `dca8d8ea93290ea194ab9a15ce6132fe` | 43.81 |
| test | 299 | `712edee82c947f68b46f06759db73ed1` | 39.94 |

## Citation

```bibtex
@inproceedings{taffa2024amqa,
  title={Low-Resource Question Answering: An Amharic Benchmarking Dataset},
  author={Taffa, Tilahun Abedissa and others},
  year={2024},
  note={https://github.com/semantic-systems/amharic-qa}
}
```
