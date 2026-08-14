#!/usr/bin/env python3
"""Merge per-language token selections into one expansion vocabulary.

Amharic and Tigrinya share the Ge'ez script and therefore share many subwords.
Concatenating the two selections would waste embedding rows on duplicates and
overshoot the target, so this script deduplicates and then trims to
--target-total using interleaved (rank-balanced) selection, which keeps each
language's highest-value tokens rather than favouring whichever file came first.

    python vocabulary_expansion/merge_vocabularies.py \
        --inputs new_tokens_amh.json new_tokens_tir.json \
        --target-total 30000 --out new_tokens.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vexmlm.geez import has_ethiopic, looks_bytelevel_encoded  # noqa: E402

log = logging.getLogger(__name__)


def load(path: Path) -> tuple[list[str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tokens" in data:
        return list(data["tokens"]), data.get("metadata", {})
    if isinstance(data, list):
        return list(data), {}
    if isinstance(data, dict):
        return list(data), {}
    raise SystemExit(f"unrecognized token file: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--target-total", type=int, default=30000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-ethiopic", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sources = []
    for path in args.inputs:
        toks, meta = load(path)
        log.info("%s: %d tokens", path.name, len(toks))
        sources.append((path.name, toks, meta))

    # Interleave by rank so each language contributes its best tokens first.
    merged: list[str] = []
    seen: set[str] = set()
    overlap = 0
    for rank in range(max(len(t) for _, t, _ in sources)):
        for name, toks, _ in sources:
            if rank >= len(toks):
                continue
            tok = toks[rank]
            if tok in seen:
                overlap += 1
                continue
            if args.require_ethiopic and not has_ethiopic(tok.lstrip("▁")):
                continue
            if looks_bytelevel_encoded(tok):
                raise SystemExit(
                    f"ABORT: byte-level-BPE token {tok!r} from {name}. "
                    f"Regenerate with tokenizer/extract_new_tokens.py.")
            seen.add(tok)
            merged.append(tok)
        if len(merged) >= args.target_total:
            break

    selected = merged[: args.target_total]
    per_source = {}
    for name, toks, _ in sources:
        tset = set(toks)
        per_source[name] = sum(1 for t in selected if t in tset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tokens": selected,
        "metadata": {
            "sources": [str(p) for p in args.inputs],
            "target_total": args.target_total,
            "selected": len(selected),
            "cross_language_duplicates_skipped": overlap,
            "contribution_per_source": per_source,
            "encoding": "raw unicode (NFC)",
            "selection": "rank-interleaved across sources, deduplicated",
        },
    }, ensure_ascii=False, indent=2))

    print(f"merged -> {args.out}")
    print(f"  selected {len(selected):,} of {args.target_total:,} target")
    print(f"  shared tokens skipped: {overlap:,}")
    for name, count in per_source.items():
        print(f"  {name}: {count:,}")


if __name__ == "__main__":
    main()
