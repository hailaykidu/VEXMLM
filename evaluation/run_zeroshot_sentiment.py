#!/usr/bin/env python3
"""Zero-shot sentiment evaluation.

Applies a sentiment model fine-tuned on one language directly to another
language's test split, with no further training. Used for the AfriSenti
languages that ship without a training split (`orm`, `tir`).

    python3 evaluation/run_zeroshot_sentiment.py \
        --model results/multilingual_evaluation/sentiment/afrisenti-amh-seed42 \
        --dataset afrisenti --dataset-config tir --split test \
        --source-language amh --seed 42 \
        --output results/multilingual_evaluation/sentiment_zeroshot/...

Label spaces are matched by name, not by index: the source model's id2label is
aligned to the target dataset's labels, so a mismatch is reported rather than
silently scoring against the wrong classes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "datasets"))

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned source-language model")
    ap.add_argument("--dataset", default="afrisenti")
    ap.add_argument("--dataset-config", required=True, help="target language, e.g. tir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--source-language", default=None,
                    help="language the model was fine-tuned on (recorded in output)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              set_seed)
    from manager import DatasetManager

    set_seed(args.seed)

    dm = DatasetManager()
    ds = dm.load(args.dataset, config=args.dataset_config)
    if args.split not in ds:
        raise SystemExit(f"split {args.split!r} not in dataset; have {list(ds)}")
    data = ds[args.split]
    if len(data) == 0:
        raise SystemExit(f"split {args.split!r} is empty for {args.dataset_config}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Align label spaces by name. The source model's head defines the ordering;
    # the target dataset's labels must map onto it.
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    feat = data.features.get("label")
    target_names = list(getattr(feat, "names", []) or [])
    if target_names:
        missing = [n for n in target_names if n not in label2id]
        if missing:
            raise SystemExit(
                f"label mismatch: target labels {target_names} not in source "
                f"model labels {sorted(label2id)}; missing {missing}")
        remap = {i: label2id[n] for i, n in enumerate(target_names)}
        gold = [remap[int(x)] for x in data["label"]]
    else:
        gold = [int(x) for x in data["label"]]

    text_col = "tweet" if "tweet" in data.column_names else "text"
    texts = data[text_col]

    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(gold, preds),
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, preds, average="weighted", zero_division=0),
        "micro_f1": f1_score(gold, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(gold, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(gold, preds, average="macro", zero_division=0),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "run_type": "expanded multilingual evaluation run",
        "task": "sentiment",
        "mode": "zero-shot",
        "dataset": args.dataset,
        "config": args.dataset_config,
        "split": args.split,
        "source_language": args.source_language,
        "source_model": args.model,
        "seed": args.seed,
        "examples": len(texts),
        "label_order": [id2label[i] for i in sorted(id2label)],
        "metrics": {f"test_{k}": v for k, v in metrics.items()},
    }
    (out / "results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    log.info("zero-shot %s <- %s seed %s: acc %.4f macro-F1 %.4f",
             args.dataset_config, args.source_language, args.seed,
             metrics["accuracy"], metrics["macro_f1"])
    log.info("wrote %s", out / "results.json")


if __name__ == "__main__":
    main()
