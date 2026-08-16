#!/usr/bin/env python3
"""Stage 2: NER fine-tuning (MasakhaNER, Tigrinya NER).

Metrics: accuracy and macro-F1 (paper Sec. 5.2), plus entity-level seqeval
scores when available -- entity-level F1 is the standard for NER and is
reported alongside the paper's required metrics.

    python finetuning/ner/run_ner.py --config configs/base.yaml \
        --model checkpoints/vexmlm-stage1 --dataset masakhaner_amh \
        --seed 42 --output checkpoints/ner-amh-42
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
from vexmlm.modes import apply_to_config, load_mode, subsample  # noqa: E402
from vexmlm.tracking import RunContext, Tracker  # noqa: E402

log = logging.getLogger(__name__)


def align_labels(examples, tokenizer, label2id, max_len: int):
    """Tokenize pre-split words and align tags to subwords.

    Only the FIRST subword of each word carries the label; continuations get
    -100 so they are ignored by the loss. This is the standard convention and
    matters more than usual here, because vocabulary expansion changes how many
    subwords a word produces.
    """
    tok = tokenizer(examples["tokens"], is_split_into_words=True,
                    truncation=True, max_length=max_len)
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        prev, labels = None, []
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                tag = tags[wid]
                labels.append(label2id[tag] if isinstance(tag, str) else int(tag))
            else:
                labels.append(-100)
            prev = wid
        all_labels.append(labels)
    tok["labels"] = all_labels
    return tok


def build_metrics(id2label: dict):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def compute(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        true, flat = [], []
        seq_true, seq_pred = [], []
        for p_row, l_row in zip(preds, labels):
            t_seq, p_seq = [], []
            for p, l in zip(p_row, l_row):
                if l != -100:
                    true.append(int(l))
                    flat.append(int(p))
                    t_seq.append(id2label[int(l)])
                    p_seq.append(id2label[int(p)])
            seq_true.append(t_seq)
            seq_pred.append(p_seq)

        out = {
            "accuracy": accuracy_score(true, flat),
            "macro_f1": f1_score(true, flat, average="macro", zero_division=0),
            "weighted_f1": f1_score(true, flat, average="weighted", zero_division=0),
        }
        try:  # entity-level scores; optional dependency
            from seqeval.metrics import f1_score as ent_f1
            from seqeval.metrics import precision_score, recall_score
            out |= {
                "entity_f1": ent_f1(seq_true, seq_pred),
                "entity_precision": precision_score(seq_true, seq_pred),
                "entity_recall": recall_score(seq_true, seq_pred),
            }
        except ImportError:
            pass
        return out

    return compute


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="masakhaner_amh")
    ap.add_argument("--dataset-config")
    ap.add_argument("--local-path")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mode", default=None,
                    choices=["debug", "research", "official"],
                    help="experiment scale/reporting profile")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments, set_seed)
    from manager import DatasetManager

    cfg = cfgmod.apply_overrides(cfgmod.load_config(args.config), args.override)
    mode = load_mode(args.mode)
    cfg = apply_to_config(cfg, mode, "finetuning")
    log.info("mode %s", mode.banner())
    seed = args.seed if args.seed is not None else cfgmod.get(cfg, "seed", 42)
    set_seed(seed)

    info = detect()
    log_environment(info)
    prec = precision_flags(info, cfgmod.get(cfg, "hardware.precision", "auto"))
    f = cfg["finetuning"]

    dm = DatasetManager()
    ds = dm.load(args.dataset, config=args.dataset_config, local_path=args.local_path)
    ds = subsample(ds, mode, seed)

    # Resolve the label set, whether tags are strings or ClassLabel ints.
    feat = ds["train"].features["ner_tags"]
    names = getattr(getattr(feat, "feature", None), "names", None)
    if names:
        label_list = list(names)
    else:
        label_list = sorted({t for row in ds["train"]["ner_tags"] for t in row},
                            key=lambda x: (x != "O", str(x)))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    log.info("labels (%d): %s", len(label_list), label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    max_len = f["max_seq_length"]
    encoded = ds.map(lambda b: align_labels(b, tokenizer, label2id, max_len),
                     batched=True, remove_columns=ds["train"].column_names,
                     desc="aligning labels")

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
        metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=50, seed=seed, data_seed=seed,
        dataloader_num_workers=cfgmod.get(cfg, "hardware.dataloader_num_workers", 4),
        report_to=[], **prec,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=encoded["train"], eval_dataset=encoded[eval_split],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_metrics(id2label))

    tracker = Tracker(
        experiment=cfgmod.get(cfg, "tracking.experiment_name", "vexmlm"),
        tracking_uri=cfgmod.get(cfg, "tracking.tracking_uri", "./experiment_tracking/mlruns"),
        enabled=not args.no_tracking)
    ctx = RunContext(run_name=f"ner-{args.dataset}-seed{seed}", stage="finetuning",
                     task="ner", seed=seed, config_hash=cfgmod.config_hash(cfg),
                     hardware={"device": info.device_type, "n_gpu": info.device_count})

    with tracker.start(ctx):
        tracker.log_params({"finetuning": f, "dataset": args.dataset,
                            "num_labels": len(label_list)})
        trainer.train()
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

        metrics = trainer.evaluate(encoded[eval_split])
        if "test" in encoded and eval_split != "test":
            metrics |= {f"test_{k.replace('eval_', '')}": v
                        for k, v in trainer.evaluate(encoded["test"]).items()}
        tracker.log_metrics({k: v for k, v in metrics.items()
                             if isinstance(v, (int, float))})

        summary = {"task": "ner", "dataset": args.dataset,
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
