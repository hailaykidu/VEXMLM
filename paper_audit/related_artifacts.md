# Related Hugging Face artifacts that are *not* paper results

This note records public models of the same author that are sometimes associated with VEXMLM but
make no claim from this paper. It exists so that the boundary of the paper's evidence is explicit:
none of the models below is used, evaluated, cited or reported in *Vocabulary Expansion for
Low-Resource African Languages: A Case Study in Amharic and Tigrinya*, and none contributed to any
number in it.

Compiled 2026-10-10 from the Hugging Face API (`/api/models/<id>` and `/api/models/<id>/commits/main`).

## The models

| Model | Created | Current revision | Last modified | Task |
|---|---|---|---|---|
| [`Hailay/TigRoberta`](https://huggingface.co/Hailay/TigRoberta) | 2024-08-09 | `d2754565bf14c60feeba8f567bcdd76ec929bcff` | 2024-08-20 | Tigrinya RoBERTa (no pipeline tag) |
| [`Hailay/EXLMR`](https://huggingface.co/Hailay/EXLMR) | 2024-08-21 | `20e9b73a3c2a24dd091f2f814214a938ebb7fefa` | 2026-07-16 | text-classification (Tigrinya offensive-language, binary) |
| [`Hailay/FT_EXLMR`](https://huggingface.co/Hailay/FT_EXLMR) | 2024-08-22 | `cabb2bb7285a4790074da313cf0d5e61015e32f1` | 2024-10-11 | text-classification |

## Revision history

```
Hailay/TigRoberta
  d2754565bf  2024-08-20  Update README.md
  f9d38ad94e  2024-08-20  Update README.md
  7aa032e417  2024-08-09  Upload 8 files
  581a816e18  2024-08-09  initial commit

Hailay/EXLMR
  20e9b73a3c  2026-07-16  Correct model card: EXLMR is a Tigrinya-only binary offensive-…
  4bf1fd1d64  2026-06-09  Update README.md
  028d7eb794  2026-06-09  Update README.md
  87d8aea438  2024-10-09  Update README.md
  a76c5080e2  2024-09-17  Update README.md
  2cec33f9c4  2024-09-17  Update README.md
  5d5b06ecb4  2024-09-17  Update README.md
  48ce78b7ee  2024-09-14  Update README.md (#2)
  924f5216a7  2024-08-31  Create README.md
  7f4dc88719  2024-08-21  Upload 6 files
  7f4814c017  2024-08-21  initial commit

Hailay/FT_EXLMR
  cabb2bb728  2024-10-11  Update README.md
  b21073e000  2024-10-11  Update README.md
  b0faab7d05  2024-10-09  Update README.md
  4f63b68b78  2024-10-09  Update README.md
  10b2275018  2024-08-25  Update README.md
  beb024ee22  2024-08-25  Update README.md
  1941505b70  2024-08-22  Update README.md
  922b9954ae  2024-08-22  Update README.md
  973ed98c04  2024-08-22  Update README.md
  5837aaf5cf  2024-08-22  Upload tokenizer
  44095cf153  2024-08-22  Upload XLMRobertaForSequenceClassification
  1d3d19cd43  2024-08-22  initial commit
```

## Why they are not paper results

1. **They predate VEXMLM by about two years.** All three were created in August 2024; VEXMLM's
   first commit is `4b6cc53`, 2026-08-14. Only one revision of any of them falls in 2026
   (`20e9b73a3c`, 2026-07-16), and it only corrects a model card.
2. **They are a different task and a different architecture.** `EXLMR` and `FT_EXLMR` are
   binary/multi-class text classifiers; `TigRoberta` is a Tigrinya RoBERTa. VEXMLM is
   XLM-RoBERTa-base with a SentencePiece-merged 280,002-entry vocabulary, evaluated on QA, NER
   and SA.
3. **They appear nowhere in this repository's pipeline.** The strings `TigRoberta`, `EXLMR` and
   `FT_EXLMR` occur in no config, runner, evaluation script or result file. The only mentions are
   this note, `paper_audit/public_artifacts_fixes.md` (which already lists them under "Not
   affected"), a comment in `tokenizer/extract_new_tokens.py` referring to an unrelated local
   `vocab.json`, and `docs/RELEASE_BRANCH.md`, which records a June 2026 `EXLMR` *prototype
   branch* that was superseded before the released architecture and contributed no result.

## On the dates 22 May, 11 August and 17 August

These dates have been raised in connection with "versions" of this work. For the record:

- **22 May** — no commit exists on this date in the VEXMLM repository or on any of the three
  models above, in any year.
- **11 August** — likewise none. The VEXMLM repository's earliest commit is 2026-08-14 20:06
  (`4b6cc53`); its lowest SLURM job id is 60577 at 2026-08-14 21:05; no log file predates
  2026-08-14. `TigRoberta` has commits on 2024-08-09 and 2024-08-20, neither on the 11th.
- **17 August** — this date is real **in VEXMLM only**: `bbaf6fc` (2026-08-17 16:42) added the
  completed Tigrinya NER OOV ablation, 4 arms × 5 seeds. None of the three models has a commit on
  any 17 August.

VEXMLM's entire experimental window is **2026-08-14 to 2026-08-18**, verifiable three independent
ways: git commit dates, result-file mtimes, and SLURM job ids and log timestamps. The correction
work took place 2026-09-24/25 and changed no result value.
