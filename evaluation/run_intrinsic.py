#!/usr/bin/env python3
"""Compute intrinsic metrics (Tables 2 and 3) for one or more tokenizers.

    python evaluation/run_intrinsic.py \
        --tokenizer xlm-roberta-base checkpoints/vexmlm-expanded \
        --corpus amh=datasets/processed/amh.dev.txt tir=datasets/processed/tir.dev.txt \
        --parallel \
        --oov-words amh=datasets/processed/amh.oov.txt \
        --out results/tokenizer_metrics.json

`--parallel` asserts the corpora are sentence-aligned; parity is only reported
as valid when it is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "metrics"))

from intrinsic import measure, oov_accuracy, parity  # noqa: E402

log = logging.getLogger(__name__)


def parse_pairs(items: list[str], what: str) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--{what} expects lang=path, got {item!r}")
        lang, path = item.split("=", 1)
        out[lang] = Path(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", nargs="+", required=True)
    ap.add_argument("--corpus", nargs="+", required=True, help="lang=path")
    ap.add_argument("--oov-words", nargs="*", default=[], help="lang=path")
    ap.add_argument("--parallel", action="store_true",
                    help="corpora are sentence-aligned (required for valid parity)")
    ap.add_argument("--base-vocab-size", type=int, default=250002)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/tokenizer_metrics.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from transformers import AutoTokenizer

    corpora = {}
    for lang, path in parse_pairs(args.corpus, "corpus").items():
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        corpora[lang] = lines[: args.limit] if args.limit else lines
        log.info("corpus %s: %d sentences", lang, len(corpora[lang]))

    if args.parallel and len({len(v) for v in corpora.values()}) != 1:
        raise SystemExit("--parallel requires equal line counts across corpora")

    oov_sets = {}
    for lang, path in parse_pairs(args.oov_words, "oov-words").items():
        oov_sets[lang] = [w for w in path.read_text(encoding="utf-8").split() if w.strip()]
        log.info("oov words %s: %d", lang, len(oov_sets[lang]))

    report: dict = {"parallel": args.parallel, "tokenizers": {}}
    for spec in args.tokenizer:
        tok = AutoTokenizer.from_pretrained(spec)
        entry: dict = {"vocab_size": len(tok), "per_language": {}, "oov": {}}
        for lang, lines in corpora.items():
            m = measure(tok, lines, language=lang, name=spec,
                        base_vocab_size=args.base_vocab_size)
            entry["per_language"][lang] = m.as_dict()
            log.info("%s/%s fertility=%.4f compression=%.4f new_share=%s",
                     spec, lang, m.fertility, m.compression, m.new_token_share)
        for lang, words in oov_sets.items():
            entry["oov"][lang] = oov_accuracy(tok, words)

        langs = list(corpora)
        if len(langs) == 2:
            a, b = langs
            entry["parity"] = parity(entry["per_language"][a]["fertility"],
                                     entry["per_language"][b]["fertility"],
                                     parallel=args.parallel)
            entry["parity"]["pair"] = f"{a}/{b}"
        report["tokenizers"][spec] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
