#!/usr/bin/env python3
"""Train the Tigrinya SentencePiece tokenizer (target vocab: 50,000).

    python tokenizer/train_tigrinya_tokenizer.py \
        --input datasets/raw/tigrinya/*.txt \
        --output-dir tokenizer/artifacts/tigrinya

Supply the Tigrinya monolingual corpus with --input; see docs/DATA_SETUP.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spm_trainer import (TokenizerSpec, prepare_corpus, summarize_vocabulary,
                         train_sentencepiece, write_manifest)

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="raw Tigrinya text file(s)")
    ap.add_argument("--output-dir", default=str(REPO / "tokenizer/artifacts/tigrinya"))
    ap.add_argument("--vocab-size", type=int, default=50000)
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    spec = TokenizerSpec(
        language="tir",
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        seed=args.seed,
    )

    corpus = out / "corpus.normalized.txt"
    stats = prepare_corpus(args.input, corpus, spec)
    if stats.lines_kept == 0:
        raise SystemExit("no usable lines after filtering -- check --input")

    model = train_sentencepiece(corpus, out / "tigrinya_sp", spec)
    vocab = summarize_vocabulary(model)
    write_manifest(out / "manifest.json", spec, stats, vocab)

    print(f"\nTigrinya tokenizer -> {model}")
    print(f"  pieces          : {vocab['piece_size']:,}")
    print(f"  Ethiopic pieces : {vocab['ethiopic_pieces']:,} ({vocab['ethiopic_fraction']:.1%})")
    print(f"  corpus retained : {stats.lines_kept:,}/{stats.lines_read:,} lines")


if __name__ == "__main__":
    main()
