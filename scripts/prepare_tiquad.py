#!/usr/bin/env python3
"""Normalize the TiQuAD parquet release into the SQuAD schema this repo uses.

The upstream files (fgaim/tiquad train.parquet / dev.parquet) store `answers`
as a list of {answer_start, text} dicts. finetuning/qa/run_qa.py expects the
SQuAD dict-of-lists form -- `answers["text"][0]` and
`answers["answer_start"][0]` -- so the column is transposed here rather than in
the runner, which would perturb the already-completed TIGQA and AmQA results.

Answers whose offset does not reproduce the gold span in the context are
dropped, mirroring the upstream cleaning applied to TIGQA (see
docs/DATASET_PROVENANCE.md). Questions left with no valid answer are dropped
entirely, since span extraction cannot score them.

Writes train.json / validation.json next to the parquet files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "tiquad"

# dev.parquet holds the held-out split; this repo calls that 'validation'.
SPLITS = {"train": "train.parquet", "validation": "dev.parquet"}


def convert(split: str, filename: str) -> dict:
    df = pd.read_parquet(RAW / filename)
    rows, dropped_ans, dropped_q = [], 0, 0

    for _, r in df.iterrows():
        texts, starts = [], []
        for a in r["answers"]:
            start, text = int(a["answer_start"]), a["text"]
            # Keep only answers whose offset reproduces the gold span exactly.
            if r["context"][start:start + len(text)] == text:
                texts.append(text)
                starts.append(start)
            else:
                dropped_ans += 1

        if not texts:
            dropped_q += 1
            continue

        rows.append({
            "id": r["id"],
            "question": r["question"],
            "context": r["context"],
            "answers": {"text": texts, "answer_start": starts},
        })

    out = RAW / f"{split}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    return {
        "split": split,
        "questions_in": len(df),
        "questions_out": len(rows),
        "questions_dropped": dropped_q,
        "answers_dropped": dropped_ans,
        "path": str(out),
    }


def main() -> None:
    for split, filename in SPLITS.items():
        stats = convert(split, filename)
        print(f"{stats['split']:11s} {stats['questions_out']:5d} questions "
              f"(in {stats['questions_in']}, dropped {stats['questions_dropped']} "
              f"questions / {stats['answers_dropped']} answers) -> {stats['path']}")


if __name__ == "__main__":
    main()
