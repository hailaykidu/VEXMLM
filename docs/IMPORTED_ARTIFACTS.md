# Imported Artifacts

Every file imported into this repository from elsewhere on this machine.
Sources are **read-only**: each artifact was copied (never moved, never
symlinked) and no source file or repository was modified.

Machine-readable records:
[`datasets/metadata/acquisition.json`](../datasets/metadata/acquisition.json),
[`legacy/PROVENANCE.json`](../legacy/PROVENANCE.json). Regenerate with:

```bash
python scripts/acquire_datasets.py --copy
python scripts/forensic_import.py --copy
```

## Datasets

| Dataset | Origin (read-only) | Imported | Split | Bytes | SHA-256 | Verified |
|---|---|---|---|---|---|---|
| `tigqa` | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/tigqa_squad` | 2026-08-14 | train | 466,419 | `920aee6eb379af97d17c` | ✅ |
| `tigqa` | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/tigqa_squad` | 2026-08-14 | validation | 50,021 | `4d79d9bbf7fbb8e2f5cb` | ✅ |
| `tigqa` | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/tigqa_squad` | 2026-08-14 | test | 58,614 | `999971f826d752128088` | ✅ |
| `amqa` | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/amqa` | 2026-08-14 | train | 1,747,839 | `c2a76dee46332964cfa9` | ✅ |
| `amqa` | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/amqa` | 2026-08-14 | validation | 497,479 | `af8497b3753aa3386c4e` | ✅ |
| `amqa` | `/homes/neumann/teklehaymanot/lgse-repro/data/qa/amqa` | 2026-08-14 | test | 254,291 | `2196b4bb88d89db4d952` | ✅ |
| `masakhaner_amh` | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/amharic` | 2026-08-14 | train | 398,840 | `bc82370a67d38ddfc97d` | ✅ |
| `masakhaner_amh` | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/amharic` | 2026-08-14 | validation | 57,855 | `1a5a5d54e87b73b7e41e` | ✅ |
| `masakhaner_amh` | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/amharic` | 2026-08-14 | test | 114,626 | `af20aa41753c19fc84b9` | ✅ |
| `tigrinya_ner` | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/tigrinya` | 2026-08-14 | train | 1,251,673 | `01a9968d78ce30c9c3ca` | ✅ |
| `tigrinya_ner` | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/tigrinya` | 2026-08-14 | validation | 156,134 | `e3f86c999f177e1fe7d3` | ✅ |
| `tigrinya_ner` | `/homes/neumann/teklehaymanot/lgse-repro/data/ner/tigrinya` | 2026-08-14 | test | 152,957 | `19fa30537db6b88addf2` | ✅ |
| `amharic_mlm` | `/homes/neumann/teklehaymanot/lgse-repro/data/lapt` | 2026-08-14 | corpus | 23,338,664 | `b5d5b1da90d3fff680d0` | ✅ |
| `tigrinya_mlm` | `/homes/neumann/teklehaymanot/lgse-repro/data/lapt` | 2026-08-14 | corpus | 18,026,579 | `e006c5ff19e3ec63f29b` | ✅ |

All 14 copies were byte-verified against their sources at import.

## Reference artifacts

Imported for analysis only; none is used by the VEXMLM pipeline. See
[REFERENCE_ARTIFACTS.md](REFERENCE_ARTIFACTS.md).

| Artifact | Origin (read-only) | Bytes | SHA-256 |
|---|---|---|---|
| `EXLMR.py` | `/homes/neumann/teklehaymanot/EXLMR/EXLMR.py` | 7,415 | `086c67285779a863a08e` |
| `EXLMR.py.save` | `/homes/neumann/teklehaymanot/EXLMR/EXLMR.py.save` | 5,480 | `820599614ca8c319f780` |
| `EXLMR.sh` | `/homes/neumann/teklehaymanot/EXLMR/EXLMR.sh` | 928 | `d20e43619c787c64eec3` |
| `vocab.json` | `/homes/neumann/teklehaymanot/EXLMR/vocab.json` | 918,848 | `93bf5cf0565d78e66285` |
| `README.md` | `/homes/neumann/teklehaymanot/EXLMR/README.md` | 3,026 | `c8b344c1ccee43014cf0` |
| `Ex.out` | `/homes/neumann/teklehaymanot/EXLMR/Ex.out` | 99,180 | `0da36108f8642cea7ddb` |
| `config.json` | `/homes/neumann/teklehaymanot/VEXMLM_Model/config.json` | 721 | `53438d14712ce7422f24` |
| `added_tokens.json` | `/homes/neumann/teklehaymanot/VEXMLM_Model/added_tokens.json` | 1,076,569 | `7f40fb638258dba2b995` |
| `special_tokens_map.json` | `/homes/neumann/teklehaymanot/VEXMLM_Model/special_tokens_map.json` | 280 | `06e405a36dfe4b9604f4` |
| `EXLMR_Model.config.json` | `/homes/neumann/teklehaymanot/EXLMR_Model/config.json` | 802 | `449d72736a4ece43d578` |
| `EXLMR_release.config.json` | `/homes/neumann/teklehaymanot/EXLMR_release/config.json` | 851 | `55ab8a57cd6159a67c82` |
| `EXLMR_release.README.md` | `/homes/neumann/teklehaymanot/EXLMR_release/README.md` | 7,023 | `66cefde410725c4c48e1` |

Recorded by reference only — too large to vendor, never copied:

| Artifact | Origin | Bytes |
|---|---|---|
| `cleaned_train.csv` | `/homes/neumann/teklehaymanot/EXLMR/cleaned_train.csv` | 11,611,745 |
| `cleaned_test.csv` | `/homes/neumann/teklehaymanot/EXLMR/cleaned_test.csv` | 742,783 |
| `sentencepiece.bpe.model` | `/homes/neumann/teklehaymanot/VEXMLM_Model/sentencepiece.bpe.model` | 5,069,051 |
| `tokenizer_config.json` | `/homes/neumann/teklehaymanot/VEXMLM_Model/tokenizer_config.json` | 5,752,083 |
| `model.safetensors` | `/homes/neumann/teklehaymanot/VEXMLM_Model/model.safetensors` | 1,204,810,552 |

## Isolation guarantee

Verified at import and again before each commit:

- no file outside `~/VEXMLM_Official/` was created, modified, or deleted;
- no git command was run in any other repository;
- source data files retain their original mtimes (Aug 5, predating this work);
- pre-existing uncommitted changes in other repositories are theirs, not this
  session's, and were left untouched;
- the one known upstream data defect (Tigrinya NER `B-LO`) is repaired **in
  memory at read time**, never in the source file.
