#!/usr/bin/env python3
"""Table 5 ablation metric: Tigrinya NER accuracy restricted to OOV words.

A word is OOV when the *baseline* XLM-R tokenizer cannot represent it without
fragmenting it beyond recognition -- concretely, when XLM-R encodes it into more
subwords than the arm's own tokenizer, or cannot round-trip it. The OOV set is
therefore a property of the baseline vocabulary alone, identical for every arm,
so the four arms are scored on exactly the same words.

Accuracy is token-level over first-subword positions (the same positions
run_ner.py labels), partitioned into OOV and non-OOV words. This isolates what
vocabulary expansion is supposed to improve: words XLM-R had to shatter.

    python3 evaluation/run_ner_oov.py \
        --model checkpoints/ner-arm4-tigrinya-42 \
        --dataset tigrinya_ner --local-path datasets/raw/tigrinya_ner \
        --arm full_vexmlm --seed 42 \
        --out results/ablation/arm4-seed42.json

The model must already be fine-tuned for token classification; this script only
evaluates. Determinism settings match the fine-tuning runners.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "datasets"))

log = logging.getLogger(__name__)

BASELINE_TOKENIZER = "xlm-roberta-base"


def oov_word_set(words: list[str], base_tok, ref_tok) -> set[str]:
    """Words the baseline XLM-R vocabulary cannot represent compactly.

    The OOV set must be a property of the BASELINE and a FIXED reference
    vocabulary only -- never of the arm under evaluation. Otherwise the arm whose
    tokenizer *is* the baseline (arm 1) trivially yields an empty OOV set, and
    each arm would be scored on a different word set, making the rows
    incomparable.

    A word is OOV when, under the baseline tokenizer, it either
      * emits <unk> or fails to round-trip, or
      * needs strictly more subwords than the fixed reference (expanded)
        tokenizer -- i.e. the baseline fragments it.

    `ref_tok` is therefore always the expanded VEXMLM tokenizer, independent of
    which arm is being scored.
    """
    unk = getattr(base_tok, "unk_token_id", None)
    oov: set[str] = set()
    for w in words:
        w = w.strip()
        if not w:
            continue
        b_ids = base_tok.encode(w, add_special_tokens=False)
        if unk is not None and unk in b_ids:
            oov.add(w)
            continue
        if base_tok.decode(b_ids, skip_special_tokens=True).strip() != w:
            oov.add(w)
            continue
        r_ids = ref_tok.encode(w, add_special_tokens=False)
        if len(b_ids) > len(r_ids):
            oov.add(w)
    return oov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned token-classification model")
    ap.add_argument("--dataset", default="tigrinya_ner")
    ap.add_argument("--local-path")
    ap.add_argument("--split", default="test")
    ap.add_argument("--reference-tokenizer", default="checkpoints/vexmlm-expanded-spm",
                    help="fixed expanded tokenizer defining the OOV set; must be "
                         "the same for every arm so the rows stay comparable")
    ap.add_argument("--arm", required=True, help="ablation arm name")
    ap.add_argument("--label", default="", help="human-readable arm label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import numpy as np
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              enable_full_determinism)
    from manager import DatasetManager

    enable_full_determinism(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    base_tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER)
    ref_tok = AutoTokenizer.from_pretrained(args.reference_tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    mgr = DatasetManager()
    ds = mgr.load(args.dataset, local_path=args.local_path)
    split = ds[args.split]
    log.info("%s/%s: %d sentences", args.dataset, args.split, len(split))

    vocab = sorted({w for row in split["tokens"] for w in row})
    oov = oov_word_set(vocab, base_tok, ref_tok)
    log.info("word types %d, OOV wrt %s (ref %s): %d (%.1f%%)",
             len(vocab), BASELINE_TOKENIZER, args.reference_tokenizer,
             len(oov), 100 * len(oov) / max(len(vocab), 1))
    if not oov:
        raise SystemExit("empty OOV set — check --reference-tokenizer")

    stats = {"oov": [0, 0], "non_oov": [0, 0]}  # [correct, total]

    with torch.no_grad():
        for start in range(0, len(split), args.batch_size):
            rows = split[start:start + args.batch_size]
            words_b, tags_b = rows["tokens"], rows["ner_tags"]
            enc = tok(words_b, is_split_into_words=True, truncation=True,
                      max_length=args.max_seq_length, padding=True,
                      return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            preds = logits.argmax(-1).cpu().numpy()

            for i in range(len(words_b)):
                wids = enc.word_ids(batch_index=i)
                prev = None
                for pos, wid in enumerate(wids):
                    if wid is None or wid == prev:
                        prev = wid
                        continue
                    prev = wid
                    gold = tags_b[i][wid]
                    gold = int(gold) if not isinstance(gold, str) else \
                        model.config.label2id[gold]
                    bucket = "oov" if words_b[i][wid].strip() in oov else "non_oov"
                    stats[bucket][1] += 1
                    stats[bucket][0] += int(preds[i][pos] == gold)

    def acc(pair):
        return round(100.0 * pair[0] / pair[1], 4) if pair[1] else None

    out = {
        "arm": args.arm,
        "label": args.label or args.arm,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "baseline_tokenizer": BASELINE_TOKENIZER,
        "reference_tokenizer": args.reference_tokenizer,
        "word_types": len(vocab),
        "oov_word_types": len(oov),
        "metrics": {
            "oov_accuracy": acc(stats["oov"]),
            "non_oov_accuracy": acc(stats["non_oov"]),
            "oov_tokens": stats["oov"][1],
            "non_oov_tokens": stats["non_oov"][1],
        },
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("OOV acc %s%% (n=%d) | non-OOV %s%% (n=%d) -> %s",
             out["metrics"]["oov_accuracy"], stats["oov"][1],
             out["metrics"]["non_oov_accuracy"], stats["non_oov"][1], dest)


if __name__ == "__main__":
    main()
