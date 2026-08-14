#!/usr/bin/env python3
"""Stage 1 of VEXMLM: continued MLM pretraining of the vocabulary-expanded model.

All parameters are trainable (paper Sec. 4). Hyperparameters come from
configs/base.yaml, which encodes the paper's Table 1.

Single GPU:
    python pretraining/run_mlm.py --config configs/base.yaml \
        --model checkpoints/vexmlm-expanded \
        --amharic datasets/processed/amh.txt \
        --tigrinya datasets/processed/tir.txt \
        --output checkpoints/vexmlm-stage1

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 pretraining/run_mlm.py --config ... [same args]

Resumes automatically from the latest checkpoint in --output unless
--no-resume is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vexmlm import config as cfgmod  # noqa: E402
from vexmlm.device import (detect, effective_batch_size, log_environment,  # noqa: E402
                           precision_flags)
from vexmlm.modes import apply_to_config, load_mode, subsample  # noqa: E402
from vexmlm.tracking import RunContext, Tracker, hash_path  # noqa: E402

log = logging.getLogger(__name__)


def build_corpus(amharic: list[str], tigrinya: list[str], cfg: dict,
                 tokenizer, seed: int, val_fraction: float, upsample_alpha: float,
                 block_chunk: bool = True):
    """Build the MLM DatasetDict from monolingual text files.

    Amharic corpora are usually much larger than Tigrinya ones. `upsample_alpha`
    rebalances via p_i ∝ n_i**alpha so Tigrinya is not drowned out; alpha=1.0
    keeps natural proportions.
    """
    import random
    from datasets import Dataset, DatasetDict

    from vexmlm.geez import normalize

    def read(paths: list[str]) -> list[str]:
        lines: list[str] = []
        for p in paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                t = normalize(line)
                if len(t) >= cfgmod.get(cfg, "tokenizer.min_sentence_length", 10):
                    lines.append(t)
        return lines

    amh = read(amharic) if amharic else []
    tir = read(tigrinya) if tigrinya else []
    log.info("corpus: amharic=%d tigrinya=%d lines", len(amh), len(tir))
    if not amh and not tir:
        raise SystemExit("no training text found -- check --amharic/--tigrinya")

    rng = random.Random(seed)
    if amh and tir and upsample_alpha < 1.0:
        n_a, n_t = len(amh), len(tir)
        p_a, p_t = n_a ** upsample_alpha, n_t ** upsample_alpha
        total = n_a + n_t
        tgt_a = int(total * p_a / (p_a + p_t))
        tgt_t = total - tgt_a

        def resample(xs: list[str], k: int) -> list[str]:
            if k <= len(xs):
                return rng.sample(xs, k)
            out = xs * (k // len(xs))
            return out + rng.sample(xs, k - len(out))

        amh, tir = resample(amh, tgt_a), resample(tir, tgt_t)
        log.info("upsampled (alpha=%.2f): amharic=%d tigrinya=%d",
                 upsample_alpha, len(amh), len(tir))

    lines = amh + tir
    rng.shuffle(lines)
    n_val = max(1, int(len(lines) * val_fraction))
    ds = DatasetDict({
        "train": Dataset.from_dict({"text": lines[n_val:]}),
        "validation": Dataset.from_dict({"text": lines[:n_val]}),
    })

    max_len = cfgmod.get(cfg, "pretraining.max_seq_length", 256)

    if not block_chunk:
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_len,
                             return_special_tokens_mask=True)

        return ds.map(tokenize, batched=True, remove_columns=["text"],
                      desc="tokenizing", num_proc=None)

    # Block-chunked construction (approved for the official run).
    #
    # Corpus lines are short -- median 30-40 characters -- so one-line-per-example
    # leaves ~90% of every 256-token sequence as padding. Concatenating the
    # tokenized corpus and slicing it into full-length blocks removes that waste
    # and gives the model contiguous context. Standard MLM practice.
    #
    # Special tokens are added per block rather than per line: adding <s>/</s>
    # around every short line would fill the block with boundary markers.
    def tokenize_plain(batch):
        return tokenizer(batch["text"], add_special_tokens=False,
                         return_attention_mask=False)

    tokenized = ds.map(tokenize_plain, batched=True, remove_columns=["text"],
                       desc="tokenizing", num_proc=None)

    # Reserve room for the two special tokens wrapped around each block.
    inner = max_len - tokenizer.num_special_tokens_to_add(pair=False)

    def group_into_blocks(batch):
        flat: list[int] = []
        for ids in batch["input_ids"]:
            flat.extend(ids)
        n_blocks = len(flat) // inner          # drop the ragged tail
        blocks = [flat[i * inner:(i + 1) * inner] for i in range(n_blocks)]
        built = [tokenizer.build_inputs_with_special_tokens(b) for b in blocks]
        return {
            "input_ids": built,
            "attention_mask": [[1] * len(b) for b in built],
            "special_tokens_mask": [
                tokenizer.get_special_tokens_mask(b, already_has_special_tokens=True)
                for b in built
            ],
        }

    blocked = tokenized.map(group_into_blocks, batched=True, batch_size=1000,
                            remove_columns=tokenized["train"].column_names,
                            desc=f"chunking into {max_len}-token blocks")
    for split in blocked:
        log.info("block-chunked %s: %d lines -> %d blocks of %d tokens",
                 split, len(ds[split]), len(blocked[split]), max_len)
    return blocked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--model", required=True, help="vocabulary-expanded checkpoint")
    ap.add_argument("--amharic", nargs="*", default=[])
    ap.add_argument("--tigrinya", nargs="*", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mode", default=None,
                    choices=["debug", "research", "official"],
                    help="experiment scale/reporting profile")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--upsample-alpha", type=float, default=0.5,
                    help="1.0 = natural proportions, 0.0 = equal languages")
    ap.add_argument("--block-chunk", dest="block_chunk", action="store_true",
                    default=True,
                    help="concatenate and chunk into max_seq_length blocks (default; official)")
    ap.add_argument("--no-block-chunk", dest="block_chunk", action="store_false",
                    help="one line per example (not the approved configuration)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap steps (smoke tests only)")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--override", nargs="*", default=[],
                    help="config overrides, e.g. pretraining.learning_rate=3e-5")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    cfg = cfgmod.apply_overrides(cfgmod.load_config(args.config), args.override)
    mode = load_mode(args.mode)
    cfg = apply_to_config(cfg, mode, "pretraining")
    log.info("mode %s", mode.banner())
    seed = args.seed if args.seed is not None else cfgmod.get(cfg, "seed", 42)
    set_seed(seed)

    info = detect()
    hw = log_environment(info)
    prec = precision_flags(info, cfgmod.get(cfg, "hardware.precision", "auto"))
    p = cfg["pretraining"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.config.vocab_size = len(tokenizer)
    log.info("model vocab=%d params=%.1fM", len(tokenizer),
             sum(x.numel() for x in model.parameters()) / 1e6)

    ds = build_corpus(args.amharic, args.tigrinya, cfg, tokenizer, seed,
                      args.val_fraction, args.upsample_alpha,
                      block_chunk=args.block_chunk)
    log.info("example construction: %s",
             "block-chunked" if args.block_chunk else "one-line-per-example")
    ds = subsample(ds, mode, seed)
    log.info("tokenized: train=%d val=%d", len(ds["train"]), len(ds["validation"]))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True,
        mlm_probability=p.get("mlm_probability", 0.15))

    targs = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=False,
        num_train_epochs=p["num_train_epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=p["per_device_train_batch_size"],
        per_device_eval_batch_size=p["per_device_train_batch_size"],
        gradient_accumulation_steps=p.get("gradient_accumulation_steps", 1),
        learning_rate=float(p["learning_rate"]),
        weight_decay=p["weight_decay"],
        max_grad_norm=p["max_grad_norm"],
        warmup_ratio=p.get("warmup_ratio", 0.06),
        lr_scheduler_type=p.get("lr_scheduler_type", "linear"),
        optim=p.get("optimizer", "adamw_torch"),
        save_strategy=p.get("save_strategy", "epoch"),
        eval_strategy=p.get("eval_strategy", "epoch"),
        save_total_limit=p.get("save_total_limit", 3),
        load_best_model_at_end=p.get("load_best_model_at_end", True),
        metric_for_best_model=p.get("metric_for_best_model", "eval_loss"),
        greater_is_better=p.get("greater_is_better", False),
        logging_steps=p.get("logging_steps", 50),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=cfgmod.get(cfg, "hardware.dataloader_num_workers", 4),
        gradient_checkpointing=cfgmod.get(cfg, "hardware.gradient_checkpointing", False),
        ddp_find_unused_parameters=cfgmod.get(cfg, "hardware.ddp_find_unused_parameters", False),
        report_to=[],
        **prec,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["validation"], data_collator=collator)

    resume = None
    if not args.no_resume and Path(args.output).is_dir():
        resume = get_last_checkpoint(args.output)
        if resume:
            log.info("resuming from %s", resume)

    eff_bs = effective_batch_size(p["per_device_train_batch_size"],
                                  p.get("gradient_accumulation_steps", 1), info)
    tracker = Tracker(
        experiment=cfgmod.get(cfg, "tracking.experiment_name", "vexmlm"),
        tracking_uri=cfgmod.get(cfg, "tracking.tracking_uri", "./experiment_tracking/mlruns"),
        enabled=not args.no_tracking)
    ctx = RunContext(run_name=f"stage1-mlm-seed{seed}", stage="pretraining",
                     language="amh+tir", seed=seed,
                     config_hash=cfgmod.config_hash(cfg),
                     tokenizer_hash=hash_path(Path(args.model) / "tokenizer.json"),
                     hardware={"device": info.device_type, "n_gpu": info.device_count,
                               "gpu": info.device_names[0] if info.device_names else "cpu",
                               "effective_batch_size": eff_bs, **prec})

    with tracker.start(ctx):
        tracker.log_params({"pretraining": p, "corpus": {
            "amharic_files": len(args.amharic), "tigrinya_files": len(args.tigrinya),
            "train_rows": len(ds["train"]), "val_rows": len(ds["validation"]),
            "upsample_alpha": args.upsample_alpha,
            "block_chunked": args.block_chunk}})

        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss")
        if loss is not None:
            metrics["perplexity"] = math.exp(loss) if loss < 20 else float("inf")
        metrics |= {f"train_{k}": v for k, v in result.metrics.items()}
        tracker.log_metrics({k: v for k, v in metrics.items()
                             if isinstance(v, (int, float))})

        summary = {"stage": "pretraining", "seed": seed, "model": args.model,
                   "output": args.output, "metrics": metrics, "hardware": hw,
                   "effective_batch_size": eff_bs}
        out = Path(args.output) / "stage1_summary.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        tracker.log_artifact(out)

    print(json.dumps({k: v for k, v in metrics.items()
                      if isinstance(v, (int, float))}, indent=2))
    log.info("Stage 1 complete -> %s", args.output)


if __name__ == "__main__":
    main()
