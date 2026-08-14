#!/usr/bin/env python3
"""Select which tokens from a trained tokenizer to add to XLM-R.

This script is the missing link in the original pipeline. `~/EXLMR/vocab.json`
existed with no generator, and the tokens in it were byte-level-BPE encoded --
so every one of them was unreachable once added to XLM-R's SentencePiece
tokenizer (0 firings measured over 84,101 tokens of real Tigrinya).

Guarantees enforced here:
  1. candidates come from a trained SentencePiece model, in raw Unicode;
  2. tokens already in XLM-R are dropped (they would be silently skipped anyway);
  3. every emitted token is validated to contain Ethiopic characters and to be
     free of byte-level-BPE mojibake -- the run ABORTS otherwise;
  4. optional frequency floor, so rare tokens don't waste embedding rows.

    python tokenizer/extract_new_tokens.py \
        --spm tokenizer/artifacts/tigrinya/tigrinya_sp.model \
        --corpus datasets/processed/tigrinya.txt \
        --max-new-tokens 30000 \
        --out vocabulary_expansion/artifacts/new_tokens.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vexmlm.geez import has_ethiopic, looks_bytelevel_encoded  # noqa: E402

log = logging.getLogger(__name__)


def load_base_vocab(base_model: str) -> set[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    return set(tok.get_vocab())


def candidate_tokens(spm_path: Path) -> list[str]:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))
    return [sp.id_to_piece(i) for i in range(sp.get_piece_size())]


def token_frequencies(spm_path: Path, corpus: Path) -> collections.Counter:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))
    counter: collections.Counter = collections.Counter()
    with open(corpus, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            counter.update(sp.encode(line.strip(), out_type=str))
    return counter


def validate(tokens: list[str]) -> None:
    """Abort if any token would be unreachable. This is the defect gate."""
    mojibake = [t for t in tokens if looks_bytelevel_encoded(t)]
    non_ethiopic = [t for t in tokens if not has_ethiopic(t.lstrip("▁"))]
    if mojibake:
        raise SystemExit(
            f"ABORT: {len(mojibake)} tokens are byte-level-BPE encoded, e.g. "
            f"{mojibake[:5]}. These can never match Ge'ez input -- this is exactly "
            f"the defect in the legacy vocabulary. See docs/TOKENIZER_DEFECT_REPORT.md."
        )
    if non_ethiopic:
        log.warning("%d selected tokens contain no Ethiopic characters, e.g. %s",
                    len(non_ethiopic), non_ethiopic[:5])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spm", required=True, type=Path, help="trained SentencePiece model")
    ap.add_argument("--corpus", type=Path, help="corpus for frequency ranking (optional)")
    ap.add_argument("--base-model", default="xlm-roberta-base")
    ap.add_argument("--max-new-tokens", type=int, default=30000)
    ap.add_argument("--min-frequency", type=int, default=0)
    ap.add_argument("--require-ethiopic", action="store_true", default=True)
    ap.add_argument("--out", type=Path,
                    default=Path("vocabulary_expansion/artifacts/new_tokens.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = load_base_vocab(args.base_model)
    cands = candidate_tokens(args.spm)
    log.info("base vocab %d, candidates %d", len(base), len(cands))

    specials = {"<pad>", "<unk>", "<s>", "</s>", "<mask>"}
    pool = [t for t in cands if t not in specials]
    already = [t for t in pool if t in base]
    pool = [t for t in pool if t not in base]
    log.info("dropped %d already present in base vocab", len(already))

    if args.require_ethiopic:
        before = len(pool)
        pool = [t for t in pool if has_ethiopic(t.lstrip("▁"))]
        log.info("dropped %d non-Ethiopic candidates", before - len(pool))

    freqs = None
    if args.corpus:
        freqs = token_frequencies(args.spm, args.corpus)
        if args.min_frequency:
            before = len(pool)
            pool = [t for t in pool if freqs[t] >= args.min_frequency]
            log.info("dropped %d below min-frequency=%d", before - len(pool),
                     args.min_frequency)
        pool.sort(key=lambda t: (-freqs[t], t))
    else:
        log.warning("no --corpus: selecting by SentencePiece order, not frequency")

    selected = pool[: args.max_new_tokens]
    validate(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": selected,
        "metadata": {
            "source_spm": str(args.spm),
            "base_model": args.base_model,
            "candidates_total": len(cands),
            "already_in_base": len(already),
            "selected": len(selected),
            "min_frequency": args.min_frequency,
            "ranked_by": "corpus frequency" if freqs else "sentencepiece order",
            "encoding": "raw unicode (NFC)",
            "validated_no_bytelevel_mojibake": True,
        },
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\nselected {len(selected):,} new tokens -> {args.out}")
    print(f"  examples: {selected[:8]}")


if __name__ == "__main__":
    main()
