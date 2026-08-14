#!/usr/bin/env python3
"""Stage 2: sentiment fine-tuning (AfriSenti).

Primary metric: accuracy (paper Sec. 5.2). Macro-F1 and weighted-F1 are also
computed -- macro-F1 is the more informative number on imbalanced label sets and
is reported alongside.

    python finetuning/sentiment/run_sentiment.py --config configs/base.yaml \
        --model checkpoints/vexmlm-stage1 --dataset-config amh \
        --seed 42 --output checkpoints/sa-amh-42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "datasets"))

from vexmlm import config as cfgmod  # noqa: E402
from vexmlm.device import detect, log_environment, precision_flags  # noqa: E402
from vexmlm.tracking import RunContext, Tracker  # noqa: E402

log = logging.getLogger(__name__)


def compute_metrics(pred):
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="afrisenti")
    ap.add_argument("--dataset-config", help="language config, e.g. amh")
    ap.add_argument("--local-path")
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)
    from manager import DatasetManager
    from registry import get as get_spec

    cfg = cfgmod.apply_overrides(cfgmod.load_config(args.config), args.override)
    seed = args.seed if args.seed is not None else cfgmod.get(cfg, "seed", 42)
    set_seed(seed)

    info = detect()
    log_environment(info)
    prec = precision_flags(info, cfgmod.get(cfg, "hardware.precision", "auto"))
    f = cfg["finetuning"]

    dm = DatasetManager()
    ds = dm.load(args.dataset, config=args.dataset_config, local_path=args.local_path)
    spec = get_spec(args.dataset)

    cols = ds["train"].column_names
    text_col = args.text_column or (spec.text_field if spec.text_field in cols else None)
    if text_col is None:
        text_col = next((c for c in ("tweet", "text", "sentence") if c in cols), None)
    label_col = args.label_column or (spec.label_field if spec.label_field in cols else None)
    if label_col is None:
        label_col = next((c for c in ("label", "labels", "sentiment") if c in cols), None)
    if not text_col or not label_col:
        raise SystemExit(f"could not resolve text/label columns from {cols}")
    log.info("text column=%r label column=%r", text_col, label_col)

    # Map string labels to contiguous ids; keep numeric labels as-is.
    feat = ds["train"].features[label_col]
    names = getattr(feat, "names", None)
    if names:
        label_list = list(names)
        label2id = {l: i for i, l in enumerate(label_list)}
        needs_map = False
    else:
        sample = ds["train"][label_col][0]
        if isinstance(sample, str):
            label_list = sorted(set(ds["train"][label_col]))
            label2id = {l: i for i, l in enumerate(label_list)}
            needs_map = True
        else:
            label_list = [str(i) for i in sorted(set(ds["train"][label_col]))]
            label2id = {l: int(l) for l in label_list}
            needs_map = False
    id2label = {i: l for l, i in label2id.items()}
    log.info("labels: %s", label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    max_len = f["max_seq_length"]

    def prep(batch):
        enc = tokenizer(batch[text_col], truncation=True, max_length=max_len)
        raw = batch[label_col]
        enc["labels"] = [label2id[v] for v in raw] if needs_map else [int(v) for v in raw]
        return enc

    encoded = ds.map(prep, batched=True, remove_columns=cols, desc="tokenizing")
    eval_split = "validation" if "validation" in encoded else "test"

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=f["num_train_epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=f["per_device_train_batch_size"],
        per_device_eval_batch_size=f["per_device_train_batch_size"],
        gradient_accumulation_steps=f.get("gradient_accumulation_steps", 1),
        learning_rate=float(f["learning_rate"]),
        weight_decay=f["weight_decay"],
        max_grad_norm=f["max_grad_norm"],
        warmup_ratio=f.get("warmup_ratio", 0.1),
        lr_scheduler_type=f.get("lr_scheduler_type", "linear"),
        optim=f.get("optimizer", "adamw_torch"),
        save_strategy=f.get("save_strategy", "epoch"),
        eval_strategy=f.get("eval_strategy", "epoch"),
        save_total_limit=f.get("save_total_limit", 2),
        load_best_model_at_end=f.get("load_best_model_at_end", True),
        metric_for_best_model="accuracy", greater_is_better=True,
        logging_steps=50, seed=seed, data_seed=seed,
        dataloader_num_workers=cfgmod.get(cfg, "hardware.dataloader_num_workers", 4),
        report_to=[], **prec,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=encoded["train"],
                      eval_dataset=encoded[eval_split],
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)

    tracker = Tracker(
        experiment=cfgmod.get(cfg, "tracking.experiment_name", "vexmlm"),
        tracking_uri=cfgmod.get(cfg, "tracking.tracking_uri", "./experiment_tracking/mlruns"),
        enabled=not args.no_tracking)
    ctx = RunContext(run_name=f"sa-{args.dataset_config or args.dataset}-seed{seed}",
                     stage="finetuning", task="sentiment",
                     language=args.dataset_config, seed=seed,
                     config_hash=cfgmod.config_hash(cfg),
                     hardware={"device": info.device_type, "n_gpu": info.device_count})

    with tracker.start(ctx):
        tracker.log_params({"finetuning": f, "dataset": args.dataset,
                            "config": args.dataset_config, "num_labels": len(label_list)})
        trainer.train()
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

        metrics = trainer.evaluate(encoded[eval_split])
        if "test" in encoded and eval_split != "test":
            metrics |= {f"test_{k.replace('eval_', '')}": v
                        for k, v in trainer.evaluate(encoded["test"]).items()}
        tracker.log_metrics({k: v for k, v in metrics.items()
                             if isinstance(v, (int, float))})

        summary = {"task": "sentiment", "dataset": args.dataset,
                   "config": args.dataset_config, "seed": seed,
                   "model": args.model, "metrics": metrics, "labels": label_list}
        out = Path(args.output) / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str))
        tracker.log_artifact(out)

    print(json.dumps({k: round(v, 4) for k, v in metrics.items()
                      if isinstance(v, (int, float))}, indent=2))


if __name__ == "__main__":
    main()
