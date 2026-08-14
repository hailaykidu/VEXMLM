#!/usr/bin/env python3
"""Train a joint Amharic+Tigrinya SentencePiece tokenizer.

Amharic corpora are typically far larger than Tigrinya ones. Naive concatenation
would let Amharic dominate the unigram objective and starve Tigrinya of subwords,
so this script supports temperature-based rebalancing (alpha), the same technique
XLM-R itself uses for language sampling.

    python tokenizer/train_joint_tokenizer.py \
        --amharic datasets/raw/amharic/*.txt \
        --tigrinya datasets/raw/tigrinya/*.txt \
        --vocab-size 64000 --alpha 0.5
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from spm_trainer import (TokenizerSpec, prepare_corpus, summarize_vocabulary,
                         train_sentencepiece, write_manifest)

REPO = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


def rebalance(amh: Path, tir: Path, out: Path, alpha: float, seed: int) -> dict:
    """Upsample the smaller language toward p_i ∝ n_i**alpha.

    alpha=1.0 keeps natural proportions; alpha=0.0 forces a 50/50 split.
    alpha=0.5 is the usual compromise and the default here.
    """
    rng = random.Random(seed)
    amh_lines = amh.read_text(encoding="utf-8").splitlines()
    tir_lines = tir.read_text(encoding="utf-8").splitlines()
    n_a, n_t = len(amh_lines), len(tir_lines)
    if not n_a or not n_t:
        raise SystemExit(f"empty corpus: amharic={n_a} tigrinya={n_t}")

    p_a = n_a ** alpha
    p_t = n_t ** alpha
    total = n_a + n_t
    target_a = int(total * p_a / (p_a + p_t))
    target_t = total - target_a

    def resample(lines: list[str], target: int) -> list[str]:
        if target <= len(lines):
            return rng.sample(lines, target)
        out = lines * (target // len(lines))
        out += rng.sample(lines, target - len(out))
        return out

    merged = resample(amh_lines, target_a) + resample(tir_lines, target_t)
    rng.shuffle(merged)
    out.write_text("\n".join(merged) + "\n", encoding="utf-8")

    info = {
        "alpha": alpha,
        "amharic_source_lines": n_a, "tigrinya_source_lines": n_t,
        "amharic_sampled": target_a, "tigrinya_sampled": target_t,
        "amharic_share_before": round(n_a / total, 4),
        "amharic_share_after": round(target_a / total, 4),
        "upsampled": "tigrinya" if target_t > n_t else ("amharic" if target_a > n_a else "none"),
    }
    log.info("rebalanced: amh %.1f%% -> %.1f%%",
             100 * n_a / total, 100 * target_a / total)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amharic", nargs="+", required=True)
    ap.add_argument("--tigrinya", nargs="+", required=True)
    ap.add_argument("--output-dir", default=str(REPO / "tokenizer/artifacts/joint"))
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="1.0=natural proportions, 0.0=equal; default 0.5")
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = TokenizerSpec(language="amh+tir", vocab_size=args.vocab_size,
                         model_type=args.model_type, seed=args.seed)

    amh_c = out / "amharic.norm.txt"
    tir_c = out / "tigrinya.norm.txt"
    s_a = prepare_corpus(args.amharic, amh_c, spec)
    s_t = prepare_corpus(args.tigrinya, tir_c, spec)

    joint = out / "corpus.normalized.txt"
    balance = rebalance(amh_c, tir_c, joint, args.alpha, args.seed)

    model = train_sentencepiece(joint, out / "joint_sp", spec)
    vocab = summarize_vocabulary(model)

    stats = s_a
    stats.input_files = s_a.input_files + s_t.input_files
    stats.lines_read = s_a.lines_read + s_t.lines_read
    stats.lines_kept = s_a.lines_kept + s_t.lines_kept
    spec.extra = {"balance": balance,
                  "amharic_corpus": s_a.__dict__, "tigrinya_corpus": s_t.__dict__}
    write_manifest(out / "manifest.json", spec, stats, vocab)

    print(f"\nJoint tokenizer -> {model}")
    print(f"  pieces          : {vocab['piece_size']:,}")
    print(f"  Ethiopic pieces : {vocab['ethiopic_pieces']:,} ({vocab['ethiopic_fraction']:.1%})")
    print(f"  amharic share   : {balance['amharic_share_before']:.1%} -> {balance['amharic_share_after']:.1%}")


if __name__ == "__main__":
    main()
