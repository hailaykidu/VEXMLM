#!/usr/bin/env python3
"""Intrinsic tokenizer metrics: fertility, compression, parity, OOV, coverage.

Definitions (stated explicitly because conventions differ across papers):

  fertility    = tokens / whitespace-word          (lower is better)
  compression  = characters / token                (higher is better)
  parity(A,B)  = fertility_A / fertility_B, same content in two languages.
                 1.0 = the tokenizer treats both languages equally. Following
                 Petrov et al. (2023), this requires PARALLEL text.
  oov_rate     = share of tokens that are <unk>
  continuation = share of tokens that are word-internal (no leading marker);
                 a proxy for how badly words are fragmented.

    python tokenizer/evaluate_tokenizer.py \
        --tokenizer xlm-roberta-base checkpoints/vexmlm-expanded \
        --corpus amh=datasets/processed/amh.txt tir=datasets/processed/tir.txt \
        --out results/tokenizer_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vexmlm.geez import has_ethiopic  # noqa: E402

log = logging.getLogger(__name__)


def load_tokenizer(spec: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec)


def evaluate(tok, lines: list[str], base_vocab_size: int | None = None) -> dict:
    n_tokens = n_words = n_chars = n_unk = n_cont = 0
    n_new_fired = 0
    unk_id = getattr(tok, "unk_token_id", None)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        ids = tok.encode(line, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        n_tokens += len(ids)
        n_words += len(line.split())
        n_chars += len(line)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
        n_cont += sum(1 for p in pieces if not p.startswith("▁"))
        if base_vocab_size is not None:
            n_new_fired += sum(1 for i in ids if i >= base_vocab_size)

    if not n_tokens:
        return {"error": "empty corpus"}

    out = {
        "sentences": len(lines),
        "tokens": n_tokens,
        "words": n_words,
        "chars": n_chars,
        "fertility": round(n_tokens / n_words, 4) if n_words else None,
        "compression": round(n_chars / n_tokens, 4),
        "oov_rate": round(n_unk / n_tokens, 6),
        "continuation_rate": round(n_cont / n_tokens, 4),
    }
    if base_vocab_size is not None:
        out["new_token_firings"] = n_new_fired
        out["new_token_share"] = round(n_new_fired / n_tokens, 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", nargs="+", required=True,
                    help="one or more tokenizer paths/names to compare")
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="lang=path pairs, e.g. amh=data/amh.txt tir=data/tir.txt")
    ap.add_argument("--parallel", action="store_true",
                    help="corpora are line-aligned parallel text (enables true parity)")
    ap.add_argument("--base-vocab-size", type=int, default=250002,
                    help="IDs >= this count as 'new' (for expanded tokenizers)")
    ap.add_argument("--limit", type=int, default=0, help="cap sentences per corpus")
    ap.add_argument("--out", type=Path, default=Path("results/tokenizer_metrics.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpora: dict[str, list[str]] = {}
    for item in args.corpus:
        if "=" not in item:
            raise SystemExit(f"--corpus expects lang=path, got {item!r}")
        lang, path = item.split("=", 1)
        lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        corpora[lang] = lines[: args.limit] if args.limit else lines
        log.info("%s: %d sentences", lang, len(corpora[lang]))

    if args.parallel:
        n = {len(v) for v in corpora.values()}
        if len(n) != 1:
            raise SystemExit(f"--parallel requires equal line counts, got {n}")

    results: dict = {"parallel": args.parallel, "tokenizers": {}}
    for spec in args.tokenizer:
        tok = load_tokenizer(spec)
        entry = {"vocab_size": len(tok), "per_language": {}}
        for lang, lines in corpora.items():
            entry["per_language"][lang] = evaluate(tok, lines, args.base_vocab_size)

        langs = list(corpora)
        if len(langs) == 2:
            a, b = langs
            fa = entry["per_language"][a]["fertility"]
            fb = entry["per_language"][b]["fertility"]
            if fa and fb:
                entry["parity"] = {
                    "pair": f"{a}/{b}",
                    "value": round(fa / fb, 4),
                    "valid": args.parallel,
                    "note": ("comparable -- parallel corpus" if args.parallel else
                             "NOT a true parity score: corpora are not parallel, so "
                             "content differences confound the ratio"),
                }
        results["tokenizers"][spec] = entry
        log.info("%s done", spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
