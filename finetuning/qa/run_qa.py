#!/usr/bin/env python3
"""Stage 2: extractive QA fine-tuning (TIGQA, AmQA).

Metrics: Exact Match and F1 (paper Sec. 5.2), computed with SQuAD-style
normalization adapted for Ge'ez script.

Long contexts are handled with a sliding window (doc_stride), and predictions
are mapped back to character offsets in the original context -- this matters
here because vocabulary expansion changes subword counts, so windowing behaves
differently before and after expansion.

    python finetuning/qa/run_qa.py --config configs/base.yaml \
        --model checkpoints/vexmlm-stage1 --dataset tigqa \
        --local-path datasets/raw/tigqa --seed 42 --output checkpoints/qa-tir-42
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import string
import sys
import unicodedata
from pathlib import Path

# Deterministic cuBLAS GEMMs require this before CUDA initialises, so it is set
# at import time rather than inside main().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "datasets"))

from vexmlm import config as cfgmod  # noqa: E402
from vexmlm.device import detect, log_environment, precision_flags  # noqa: E402
from vexmlm.geez import GEEZ_PUNCT  # noqa: E402
from vexmlm.modes import apply_to_config, load_mode, subsample  # noqa: E402
from vexmlm.tracking import RunContext, Tracker  # noqa: E402

log = logging.getLogger(__name__)

# SQuAD's normalizer strips English articles; that is meaningless for Ge'ez and
# would be wrong to apply. We strip punctuation (Latin and Ge'ez) and whitespace.
_PUNCT = set(string.punctuation) | set(GEEZ_PUNCT)


def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def squad_metrics(predictions: dict[str, str], references: dict[str, list[str]]) -> dict:
    em = f1 = 0.0
    for qid, pred in predictions.items():
        golds = references.get(qid) or [""]
        em += max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)
        f1 += max(token_f1(pred, g) for g in golds)
    n = max(len(predictions), 1)
    return {"exact_match": 100 * em / n, "f1": 100 * f1 / n, "n_questions": n}


def prepare_train(examples, tokenizer, max_len: int, stride: int):
    """Tokenize with a sliding window and locate answer spans in token indices."""
    questions = [q.lstrip() for q in examples["question"]]
    tok = tokenizer(questions, examples["context"], truncation="only_second",
                    max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding="max_length")
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets = tok.pop("offset_mapping")
    starts, ends = [], []

    for i, offset in enumerate(offsets):
        input_ids = tok["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        seq_ids = tok.sequence_ids(i)
        sample_idx = sample_map[i]
        answer = examples["answers"][sample_idx]

        if not answer["text"] or not answer["text"][0]:
            starts.append(cls_index)
            ends.append(cls_index)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Narrow to the context portion of this window.
        t0 = 0
        while seq_ids[t0] != 1:
            t0 += 1
        t1 = len(input_ids) - 1
        while seq_ids[t1] != 1:
            t1 -= 1

        if not (offset[t0][0] <= start_char and offset[t1][1] >= end_char):
            starts.append(cls_index)   # answer not in this window
            ends.append(cls_index)
        else:
            a = t0
            while a <= t1 and offset[a][0] <= start_char:
                a += 1
            starts.append(a - 1)
            b = t1
            while b >= t0 and offset[b][1] >= end_char:
                b -= 1
            ends.append(b + 1)

    tok["start_positions"] = starts
    tok["end_positions"] = ends
    return tok


def prepare_eval(examples, tokenizer, max_len: int, stride: int):
    questions = [q.lstrip() for q in examples["question"]]
    tok = tokenizer(questions, examples["context"], truncation="only_second",
                    max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding="max_length")
    sample_map = tok.pop("overflow_to_sample_mapping")
    tok["example_id"] = [examples["id"][sample_map[i]] for i in range(len(tok["input_ids"]))]
    # Keep only context offsets so predictions can never point into the question.
    tok["offset_mapping"] = [
        [o if tok.sequence_ids(i)[k] == 1 else None
         for k, o in enumerate(tok["offset_mapping"][i])]
        for i in range(len(tok["input_ids"]))
    ]
    return tok


def postprocess(examples, features, raw_predictions, n_best: int, max_answer_len: int) -> dict:
    """Map start/end logits back to answer strings."""
    start_logits, end_logits = raw_predictions
    feature_per_example = collections.defaultdict(list)
    for i, feat_id in enumerate(features["example_id"]):
        feature_per_example[feat_id].append(i)

    predictions = {}
    for example in examples:
        ex_id = example["id"]
        context = example["context"]
        best_text, best_score = "", -1e9

        for feat_idx in feature_per_example[ex_id]:
            starts = start_logits[feat_idx]
            ends = end_logits[feat_idx]
            offsets = features["offset_mapping"][feat_idx]

            start_idx = np.argsort(starts)[-1: -n_best - 1: -1]
            end_idx = np.argsort(ends)[-1: -n_best - 1: -1]
            for s in start_idx:
                for e in end_idx:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    score = starts[s] + ends[e]
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0]: offsets[e][1]]
        predictions[ex_id] = best_text
    return predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="tigqa")
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

    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              default_data_collator, enable_full_determinism,
                              Trainer, TrainingArguments)
    from manager import DatasetManager

    cfg = cfgmod.apply_overrides(cfgmod.load_config(args.config), args.override)
    mode = load_mode(args.mode)
    cfg = apply_to_config(cfg, mode, "finetuning")
    log.info("mode %s", mode.banner())
    seed = args.seed if args.seed is not None else cfgmod.get(cfg, "seed", 42)
    # enable_full_determinism seeds Python/NumPy/torch *and* enables deterministic
    # cuDNN/cuBLAS kernels. set_seed alone left kernel selection nondeterministic,
    # so identical seeds produced slightly different metrics between runs.
    enable_full_determinism(seed)

    info = detect()
    log_environment(info)
    prec = precision_flags(info, cfgmod.get(cfg, "hardware.precision", "auto"))
    f = cfg["finetuning"]
    q = cfg.get("qa", {})
    max_len = f["max_seq_length"]

    dm = DatasetManager()
    ds = dm.load(args.dataset, config=args.dataset_config, local_path=args.local_path)
    ds = subsample(ds, mode, seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        raise SystemExit("QA requires a fast tokenizer (offset mapping).")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    # The stride must stay below the effective max length (max_seq_length minus
    # the special tokens), or the fast tokenizer raises. Clamp instead of
    # failing: a smaller stride only means more window overlap.
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    max_stride = max(max_len - n_special - 1, 16)
    stride = q.get("doc_stride", 128)
    if stride >= max_stride:
        log.warning("doc_stride=%d too large for max_seq_length=%d (%d special "
                    "tokens); clamping to %d", stride, max_len, n_special, max_stride)
        stride = max_stride

    eval_split = "validation" if "validation" in ds else "test"
    train_feats = ds["train"].map(
        lambda b: prepare_train(b, tokenizer, max_len, stride), batched=True,
        remove_columns=ds["train"].column_names, desc="preparing train")
    eval_feats = ds[eval_split].map(
        lambda b: prepare_eval(b, tokenizer, max_len, stride), batched=True,
        remove_columns=ds[eval_split].column_names, desc="preparing eval")

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
        eval_strategy="no",          # QA eval needs custom postprocessing
        save_total_limit=f.get("save_total_limit", 2),
        logging_steps=50, seed=seed, data_seed=seed,
        dataloader_num_workers=cfgmod.get(cfg, "hardware.dataloader_num_workers", 4),
        report_to=[], **prec,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_feats,
                      data_collator=default_data_collator, tokenizer=tokenizer)

    tracker = Tracker(
        experiment=cfgmod.get(cfg, "tracking.experiment_name", "vexmlm"),
        tracking_uri=cfgmod.get(cfg, "tracking.tracking_uri", "./experiment_tracking/mlruns"),
        enabled=not args.no_tracking)
    ctx = RunContext(run_name=f"qa-{args.dataset}-seed{seed}", stage="finetuning",
                     task="qa", seed=seed, config_hash=cfgmod.config_hash(cfg),
                     hardware={"device": info.device_type, "n_gpu": info.device_count})

    with tracker.start(ctx):
        tracker.log_params({"finetuning": f, "qa": q, "dataset": args.dataset})
        trainer.train()
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

        raw = trainer.predict(eval_feats.remove_columns(["example_id", "offset_mapping"]))
        preds = postprocess(ds[eval_split], eval_feats, raw.predictions[:2],
                            q.get("n_best_size", 20), q.get("max_answer_length", 64))
        refs = {ex["id"]: ex["answers"]["text"] for ex in ds[eval_split]}
        metrics = squad_metrics(preds, refs)
        tracker.log_metrics(metrics)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions.json").write_text(
            json.dumps(preds, ensure_ascii=False, indent=2))
        summary = {"task": "qa", "dataset": args.dataset, "seed": seed,
                   "model": args.model, "metrics": metrics}
        (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str))
        tracker.log_artifact(out_dir / "results.json")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
