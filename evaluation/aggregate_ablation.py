#!/usr/bin/env python3
"""Aggregate the Table 5 ablation into a CSV and a LaTeX-ready summary.

Reads the per-arm, per-seed records written by evaluation/run_ner_oov.py and
reports mean +/- std downstream NER OOV accuracy for each arm, in the paper's
row order, with the per-component delta.

Arms with no record are emitted as NOT YET MEASURED rather than omitted, so a
partial ablation is visible instead of looking complete.

    python3 evaluation/aggregate_ablation.py \
        --runs results/ablation --out results/table5_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics as st
from pathlib import Path

log = logging.getLogger(__name__)

NOT_MEASURED = "NOT YET MEASURED"

# Paper row order: each arm adds one component to the previous.
ARMS = [
    ("xlmr_baseline", "XLM-R baseline"),
    ("expansion_random_init", "+ VocabExp (Random Init)"),
    ("expansion_mean_init", "+ VocabExp (Mean Init)"),
    ("full_vexmlm", "+ Continued Pretraining"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/ablation")
    ap.add_argument("--out", default="results/table5_ablation.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.runs)

    collected: dict[str, list[tuple[int, float]]] = {}
    meta: dict[str, dict] = {}
    for arm, _ in ARMS:
        vals = []
        for f in sorted(root.glob(f"{arm}-seed*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            acc = d.get("metrics", {}).get("oov_accuracy")
            if acc is None:
                log.warning("%s has null oov_accuracy — skipped", f)
                continue
            vals.append((int(d["seed"]), float(acc)))
            meta[arm] = d
        if vals:
            collected[arm] = sorted(vals)

    rows, prev_mu = [], None
    print(f"\n{'Configuration':<30}{'OOV Acc. (%)':>16}{'Delta':>10}{'n':>4}")
    print("-" * 60)
    for arm, label in ARMS:
        vals = collected.get(arm)
        if not vals:
            rows.append([label, arm, NOT_MEASURED, NOT_MEASURED, 0, ""])
            print(f"{label:<30}{NOT_MEASURED:>16}{'':>10}{0:>4}")
            continue
        nums = [v for _, v in vals]
        mu = sum(nums) / len(nums)
        sd = st.stdev(nums) if len(nums) > 1 else 0.0
        delta = "" if prev_mu is None else f"{mu - prev_mu:+.2f}"
        seeds = ",".join(str(s) for s, _ in vals)
        rows.append([label, arm, f"{mu:.2f} ± {sd:.2f}", delta, len(nums), seeds])
        print(f"{label:<30}{mu:>10.2f} ± {sd:<4.2f}{delta:>10}{len(nums):>4}")
        prev_mu = mu

    if collected:
        any_meta = next(iter(meta.values()))
        print(f"\nOOV set: {any_meta['oov_word_types']} of {any_meta['word_types']} "
              f"word types ({100*any_meta['oov_word_types']/any_meta['word_types']:.1f}%)")
        print(f"baseline={any_meta['baseline_tokenizer']}  "
              f"reference={any_meta.get('reference_tokenizer')}")
        print(f"dataset={any_meta['dataset']} split={any_meta['split']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Configuration", "Arm", "OOV Accuracy (%) mean ± std",
                    "Delta", "Seeds", "Seed list"])
        w.writerows(rows)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
