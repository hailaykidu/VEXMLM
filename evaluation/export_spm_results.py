#!/usr/bin/env python3
"""Export the authoritative SP-Merge Stage 2 evaluation into tracked artifacts.

`checkpoints/` is gitignored (it holds ~145 GB of model weights and optimizer
state), so the run outputs that back the paper's downstream table never reach a
clone. This script copies the small, decision-relevant part -- the per-run
`results.json` plus provenance -- into `results/spm_stage2/`, which IS tracked,
and regenerates `results/downstream_task_metrics.csv` from those files.

The SP-Merge runs are a distinct model variant from the historical add_tokens
VEXMLM and from `results/multilingual_evaluation/`; none of the three are pooled
here. TiQuAD is emitted with a `supplementary` flag because it is a diagnostic
task, not a paper benchmark.

    python3 evaluation/export_spm_results.py \
        --runs checkpoints/vexmlm-spm-stage2 \
        --out results/spm_stage2 \
        --csv results/downstream_task_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Paper benchmarks vs. supplementary diagnostics.
SUPPLEMENTARY = {"tiquad"}

# Metric -> (csv label, formatting precision). Only metrics the paper reports.
REPORTED = {
    "test_accuracy": ("Accuracy", 4),
    "test_macro_f1": ("Macro-F1", 4),
    "test_entity_f1": ("Entity-F1", 4),
    "exact_match": ("EM", 2),
    "f1": ("F1", 2),
}

TASK_OF = {
    "masakhaner_amh": "ner", "tigrinya_ner": "ner",
    "amqa": "qa", "tigqa": "qa", "tiquad": "qa",
    "afrisenti": "sentiment",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="checkpoints/vexmlm-spm-stage2")
    ap.add_argument("--out", default="results/spm_stage2")
    ap.add_argument("--csv", default="results/downstream_task_metrics.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs = sorted(Path(args.runs).glob("*-seed*/results.json"))
    if not runs:
        raise SystemExit(f"no results.json under {args.runs}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: dict[str, set[int]] = defaultdict(set)
    models: set[str] = set()

    for src in runs:
        name = src.parent.name                 # e.g. amqa-seed42
        dataset, seed = name.rsplit("-seed", 1)
        shutil.copy2(src, out / f"{name}.json")
        d = json.loads(src.read_text(encoding="utf-8"))
        models.add(str(d.get("model")))
        seeds[dataset].add(int(seed))
        metrics = d.get("metrics", d)
        for k, v in metrics.items():
            if k in REPORTED and isinstance(v, (int, float)):
                per[dataset][k].append(float(v))

    log.info("copied %d run records -> %s", len(runs), out)
    if len(models) != 1:
        log.warning("runs span multiple models: %s", sorted(models))

    rows = []
    for dataset in sorted(per, key=lambda d: (TASK_OF.get(d, "zz"), d)):
        for metric, (label, prec) in REPORTED.items():
            vals = per[dataset].get(metric)
            if not vals:
                continue
            mu = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            rows.append([
                TASK_OF.get(dataset, ""), dataset, label,
                f"{mu:.{prec}f} ± {sd:.{prec}f}",
                len(vals),
                "supplementary" if dataset in SUPPLEMENTARY else "benchmark",
            ])

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Task", "Dataset", "Metric",
                    "VEXMLM SP-Merge (mean ± std)", "Seeds", "Status"])
        w.writerows(rows)
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    provenance = {
        "run_set": "SP-Merge Stage 2 (authoritative downstream evaluation)",
        "source": str(args.runs),
        "model": sorted(models)[0] if len(models) == 1 else sorted(models),
        "n_runs": len(runs),
        "seeds": {k: sorted(v) for k, v in sorted(seeds.items())},
        "supplementary_tasks": sorted(SUPPLEMENTARY),
        "note": ("Distinct from results/multilingual_evaluation/ and from the "
                 "historical add_tokens VEXMLM runs; these sets are never pooled."),
    }
    (out / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", out / "PROVENANCE.json")


if __name__ == "__main__":
    main()
