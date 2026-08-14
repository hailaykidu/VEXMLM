#!/usr/bin/env python3
"""Step 2-3 of the VEXMLM pipeline: expand XLM-R's vocabulary and initialize
the new embedding rows.

Implements the paper's specification:
  * all original XLM-R vocabulary entries are preserved (tokens are appended,
    never replaced or reordered);
  * only tokens absent from XLM-R are added;
  * new embeddings use mean-based initialization, e_t = (1/|V_s|) * sum e_s
    (Sec. 3.3), which is the default here.

Safeguards:
  * an encoding gate rejects byte-level-BPE tokens, which cannot match input in
    a SentencePiece tokenizer;
  * a firing check confirms the added tokens are actually reachable on real text;
  * every RNG is seeded, so runs are reproducible;
  * a manifest records exactly what happened.

    python vocabulary_expansion/expand_xlmr_vocab.py \
        --new-tokens vocabulary_expansion/artifacts/new_tokens.json \
        --init global_mean --seed 42 \
        --validation-corpus datasets/processed/tir.dev.txt \
        --output checkpoints/vexmlm-expanded
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from initialization import STRATEGIES, initialize  # noqa: E402
from vexmlm.geez import looks_bytelevel_encoded  # noqa: E402

log = logging.getLogger(__name__)


def load_new_tokens(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tokens" in data:
        return list(data["tokens"])
    if isinstance(data, dict):
        # legacy vocab.json format: {token: id}
        return list(data)
    if isinstance(data, list):
        return list(data)
    raise SystemExit(f"unrecognized token file format: {path}")


def gate_encoding(tokens: list[str], allow_broken: bool) -> None:
    bad = [t for t in tokens if looks_bytelevel_encoded(t)]
    if not bad:
        return
    msg = (f"{len(bad)}/{len(tokens)} tokens are byte-level-BPE encoded "
           f"(e.g. {bad[:3]} -> mojibake). Added to a SentencePiece tokenizer "
           f"these can NEVER match input text.")
    if allow_broken:
        log.warning("%s  Proceeding because --allow-broken-encoding was passed.", msg)
    else:
        # Print then exit non-zero: a bare SystemExit(str) exits 0, which would
        # let scripts/reproduce_paper.sh sail past a fatal data defect.
        log.error("ABORT: %s\nEmit tokens with tokenizer/extract_new_tokens.py, which "
                  "guarantees raw-Unicode output. See docs/REFERENCE_ARTIFACTS.md.", msg)
        raise SystemExit(2)


def verify_tokens_fire(tok, corpus: Path, n_old: int, limit: int = 2000) -> dict:
    """Confirm the new tokens are reachable. The missing check."""
    lines = [l for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    total = fired = 0
    for line in lines:
        ids = tok.encode(line, add_special_tokens=False)
        total += len(ids)
        fired += sum(1 for i in ids if i >= n_old)
    share = fired / total if total else 0.0
    result = {"sentences": len(lines), "tokens": total, "new_token_firings": fired,
              "new_token_share": round(share, 6)}
    if fired == 0:
        log.error("VERIFICATION FAILED: 0 new tokens fired across %d tokens. "
                  "The expansion has no effect -- this is the legacy defect.", total)
        result["status"] = "FAILED"
    else:
        log.info("verification OK: %d firings (%.2f%% of tokens)", fired, 100 * share)
        result["status"] = "OK"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="xlm-roberta-base")
    ap.add_argument("--new-tokens", required=True, type=Path)
    ap.add_argument("--init", default="global_mean", choices=STRATEGIES,
                    help="official method is global_mean (paper Sec. 3.3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="research knob; keep 0 to follow the paper")
    ap.add_argument("--legacy-unit-normal", action="store_true",
                    help="draw the random term from N(0,1) instead of base scale")
    ap.add_argument("--allow-broken-encoding", action="store_true",
                    help="permit byte-level-BPE tokens (diagnostics only)")
    ap.add_argument("--validation-corpus", type=Path,
                    help="text file used to verify the new tokens actually fire")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokens = load_new_tokens(args.new_tokens)
    log.info("loaded %d candidate tokens from %s", len(tokens), args.new_tokens)
    gate_encoding(tokens, args.allow_broken_encoding)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    base_tok = AutoTokenizer.from_pretrained(args.base_model)  # unmodified reference
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    n_old = len(tok)
    added = tok.add_tokens(tokens)
    n_new_total = len(tok)
    log.info("base vocab %d -> %d (%d added, %d skipped as duplicates)",
             n_old, n_new_total, added, len(tokens) - added)

    model.resize_token_embeddings(n_new_total)
    emb = model.get_input_embeddings().weight.data

    # add_tokens() appends sequentially from n_old, regardless of input order.
    actually_added = [t for t in tokens if tok.convert_tokens_to_ids(t) >= n_old]
    actually_added.sort(key=tok.convert_tokens_to_ids)

    kwargs: dict = {}
    if args.init in ("mixed", "random") and args.legacy_unit_normal:
        kwargs["legacy_unit_normal" if args.init == "mixed" else "match_base"] = (
            True if args.init == "mixed" else False)
    if args.init in ("global_mean", "constituent_mean") and args.noise_std:
        kwargs["noise_std"] = args.noise_std

    stats = initialize(args.init, emb, n_old, new_tokens=actually_added,
                       base_tokenizer=base_tok, seed=args.seed, **kwargs)
    log.info("init=%s new_std=%.4f dist_to_mean=%.4f",
             stats.strategy, stats.new_perdim_std, stats.dist_to_global_mean_mean)

    # Tie output embeddings if the model shares them.
    if model.get_output_embeddings() is not None:
        model.tie_weights()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)

    manifest = {
        "base_model": args.base_model,
        "base_vocab_size": n_old,
        "final_vocab_size": n_new_total,
        "candidates_supplied": len(tokens),
        "tokens_added": added,
        "duplicates_skipped": len(tokens) - added,
        "initialization": stats.as_dict(),
        "seed": args.seed,
        "encoding_gate": "bypassed" if args.allow_broken_encoding else "enforced",
        "new_tokens_source": str(args.new_tokens),
    }
    if args.validation_corpus:
        manifest["firing_verification"] = verify_tokens_fire(
            tok, args.validation_corpus, n_old)

    (args.output / "expansion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nsaved expanded model -> {args.output}")


if __name__ == "__main__":
    main()
