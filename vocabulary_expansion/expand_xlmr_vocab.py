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


def merge_into_sentencepiece(base_model: str, tokens: list[str],
                             workdir: Path) -> tuple[Path, list[str]]:
    """Append tokens to XLM-R's SentencePiece model as native pieces.

    This is the correct integration path. `tokenizer.add_tokens()` registers
    entries in added_tokens.json instead, where HuggingFace's added-token matcher
    runs on raw text *before* SentencePiece segmentation -- so a token matches
    mid-word and its U+2581 word-boundary marker is emitted inside the word,
    which decoding then turns into a spurious space. Merging into the ModelProto
    puts the pieces inside the Unigram lattice, where boundary semantics hold and
    segmentation is chosen by Viterbi rather than greedy longest-match.

    Returns the new .model path and the tokens actually appended (order matters:
    it fixes the embedding-row assignment).

    Note on <mask>: XLM-R keeps `<mask>` outside the SentencePiece model, at the
    id one past the last piece (250001 for the 250000-piece base). Appending
    pieces therefore shifts that boundary, and the caller must rebuild the
    tokenizer so `<mask>` lands after the new pieces rather than shadowing one.
    """
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    from transformers import AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(base_model)
    proto_path = getattr(base_tok, "vocab_file", None)
    if not proto_path or not Path(proto_path).is_file():
        raise SystemExit(
            f"{base_model} exposes no SentencePiece model file; native merge "
            f"requires a SentencePiece base (got vocab_file={proto_path!r}).")

    proto = sp_pb2.ModelProto()
    proto.ParseFromString(Path(proto_path).read_bytes())

    existing = {p.piece for p in proto.pieces}
    # New pieces are scored below every existing piece so they never outrank
    # pretrained segmentations; the Unigram lattice still selects them when they
    # yield fewer pieces overall.
    floor = min(p.score for p in proto.pieces)

    appended: list[str] = []
    for tok in tokens:
        if tok in existing:
            continue
        piece = proto.pieces.add()
        piece.piece = tok
        piece.score = floor - 1.0
        piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL
        existing.add(tok)
        appended.append(tok)

    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "sentencepiece.bpe.model"
    out.write_bytes(proto.SerializeToString())
    log.info("merged %d pieces into SentencePiece model -> %s (%d total)",
             len(appended), out, len(proto.pieces))
    return out, appended


def verify_roundtrip(tok, corpus: Path, limit: int = 300) -> dict:
    """Measure decode(encode(x)) == x on real text. The check that was missing.

    The add_tokens() path corrupts ~95% of Tigrinya sentences here; a correct
    merge scores ~100%.
    """
    lines = [l.strip() for l in corpus.read_text(encoding="utf-8").splitlines()
             if l.strip()][:limit]
    ok = 0
    failures: list[str] = []
    for line in lines:
        ids = tok.encode(line, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True).strip()
        if decoded == line:
            ok += 1
        elif len(failures) < 5:
            failures.append(f"{line[:40]!r} -> {decoded[:40]!r}")
    n = len(lines) or 1
    return {"sentences": len(lines), "roundtrip_ok": ok,
            "roundtrip_rate": round(ok / n, 4), "examples": failures}


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
    ap.add_argument("--integration", default="sentencepiece_merge",
                    choices=("sentencepiece_merge", "add_tokens"),
                    help="how new tokens enter the tokenizer. 'sentencepiece_merge' "
                         "(default, correct) appends native pieces to the SP "
                         "ModelProto. 'add_tokens' reproduces the legacy path that "
                         "corrupts decoding; kept only for reproducing the original "
                         "checkpoint.")
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

    base_tok = AutoTokenizer.from_pretrained(args.base_model)  # unmodified reference
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    n_old = len(base_tok)

    if args.integration == "sentencepiece_merge":
        # Rebuild the tokenizer from a ModelProto that already contains the new
        # pieces, so they are ordinary vocabulary rather than added_tokens.
        merged_model, appended = merge_into_sentencepiece(
            args.base_model, tokens, args.output / "_spm")
        # The fast tokenizer would silently ignore vocab_file and reload the
        # cached tokenizer.json, so the slow tokenizer is built from the merged
        # proto and converted afterwards.
        tok = AutoTokenizer.from_pretrained(args.base_model,
                                            vocab_file=str(merged_model),
                                            use_fast=False)
        # <mask> sat one past the old last piece and now collides with a merged
        # piece; move it past the new pieces so every piece stays reachable.
        n_pieces = tok.sp_model.get_piece_size()
        mask_id = n_pieces + tok.fairseq_offset
        tok.fairseq_tokens_to_ids["<mask>"] = mask_id
        tok.fairseq_ids_to_tokens = {
            i: t for t, i in tok.fairseq_tokens_to_ids.items()}
        added = len(appended)
        n_new_total = mask_id + 1
        # SentencePiece ids run in proto order, which is append order, so the
        # embedding rows line up with `appended` directly.
        actually_added = appended
    else:
        tok = AutoTokenizer.from_pretrained(args.base_model)
        added = tok.add_tokens(tokens)
        n_new_total = len(tok)
        # add_tokens() appends sequentially from n_old, regardless of input order.
        actually_added = [t for t in tokens if tok.convert_tokens_to_ids(t) >= n_old]
        actually_added.sort(key=tok.convert_tokens_to_ids)

    log.info("integration=%s: base vocab %d -> %d (%d added, %d skipped as duplicates)",
             args.integration, n_old, n_new_total, added, len(tokens) - added)

    model.resize_token_embeddings(n_new_total)
    emb = model.get_input_embeddings().weight.data

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
        "integration": args.integration,
        "initialization": stats.as_dict(),
        "seed": args.seed,
        "encoding_gate": "bypassed" if args.allow_broken_encoding else "enforced",
        "new_tokens_source": str(args.new_tokens),
    }
    if args.validation_corpus:
        manifest["firing_verification"] = verify_tokens_fire(
            tok, args.validation_corpus, n_old)
        manifest["roundtrip_verification"] = verify_roundtrip(
            tok, args.validation_corpus)
        rt = manifest["roundtrip_verification"]["roundtrip_rate"]
        if args.integration == "sentencepiece_merge" and rt < 0.99:
            log.error("VERIFICATION FAILED: round-trip %.4f < 0.99 after a native "
                      "merge. The tokenizer still corrupts text; do not train on it.",
                      rt)
            manifest["roundtrip_verification"]["status"] = "FAILED"
        else:
            log.info("round-trip OK: %.4f", rt)

    (args.output / "expansion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nsaved expanded model -> {args.output}")


if __name__ == "__main__":
    main()
