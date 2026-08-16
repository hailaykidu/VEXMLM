#!/usr/bin/env python3
"""Aggregate the expanded multilingual evaluation into per-language and
macro/micro-average tables.

    python3 evaluation/aggregate_multilingual.py \
        --runs results/multilingual_evaluation \
        --out  results/multilingual_evaluation/aggregated

Reads every `results.json` under --runs, groups by (task, dataset, language),
and reports mean +/- std across seeds.

Averaging conventions
---------------------
* **Macro average** — unweighted mean over languages. Every language counts
  equally regardless of test-set size.
* **Micro average** — mean over languages weighted by test-set size, i.e. the
  score an evaluator would see pooling all test examples together.

Zero-shot languages are aggregated separately and never folded into the
fine-tuned averages. Languages with no completed run are reported as NOT RUN
rather than dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

# Metric reported as the headline per task.
HEADLINE = {"sentiment": "macro_f1", "ner": "f1", "qa": "f1"}


def parse_run(path: Path) -> dict | None:
    """Extract (task, dataset, language, seed, mode, metrics) from a results.json."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    task = d.get("task")
    dataset = d.get("dataset")
    if not task or not dataset:
        return None

    lang = d.get("config") or d.get("language")
    if not lang:
        # Local single-language sets encode the language in the dataset key.
        lang = {"masakhaner_amh": "amh", "tigrinya_ner": "tir",
                "amqa": "amh", "tigqa": "tir", "tiquad": "tir"}.get(dataset)
    if not lang:
        # Multi-config sets: some runners omit `config` from results.json, so
        # recover the language from the run directory (e.g. masakhaner2-lug-seed42).
        m = re.match(rf"{re.escape(dataset)}-([a-z]{{2,4}})-seed\d+$", path.parent.name)
        if m:
            lang = m.group(1)

    seed = d.get("seed")
    if seed is None:
        m = re.search(r"seed(\d+)", path.parent.name)
        seed = int(m.group(1)) if m else None

    metrics = d.get("metrics", d)
    test = {k[len("test_"):]: v for k, v in metrics.items()
            if k.startswith("test_") and isinstance(v, (int, float))}
    if not test:
        test = {k[len("eval_"):]: v for k, v in metrics.items()
                if k.startswith("eval_") and isinstance(v, (int, float))}
    if not test:
        # Unprefixed metrics (the QA runner reports exact_match / f1 directly).
        test = {k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "n_questions"}
    if not test:
        return None

    return {"task": task, "dataset": dataset, "language": lang, "seed": seed,
            "mode": d.get("mode", "fine-tune"), "examples": d.get("examples"),
            "metrics": test}


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return stats.mean(xs), (stats.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/multilingual_evaluation")
    ap.add_argument("--out", default="results/multilingual_evaluation/aggregated")
    ap.add_argument("--coverage",
                    default="results/multilingual_evaluation/aggregated/language_coverage.csv",
                    help="language inventory, used to flag languages with no run")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in (parse_run(p) for p in runs_root.rglob("results.json")) if r]
    if not runs:
        raise SystemExit(f"no results.json found under {runs_root}")

    # group: (task, mode, dataset, language) -> {metric: [values]}
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sizes: dict[tuple, int] = {}
    for r in runs:
        key = (r["task"], r["mode"], r["dataset"], r["language"])
        for m, v in r["metrics"].items():
            grouped[key][m].append(v)
        if r.get("examples"):
            sizes[key] = r["examples"]

    # ---- per-language ----
    per_lang_rows = []
    for (task, mode, dataset, lang), metrics in sorted(grouped.items()):
        row = {"task": task, "mode": mode, "dataset": dataset, "language": lang,
               "seeds": len(next(iter(metrics.values())))}
        for m, vals in sorted(metrics.items()):
            mu, sd = mean_std(vals)
            row[f"{m}_mean"] = round(mu, 4)
            row[f"{m}_std"] = round(sd, 4)
        per_lang_rows.append(row)

    if per_lang_rows:
        cols = sorted({k for r in per_lang_rows for k in r},
                      key=lambda c: (c not in ("task", "mode", "dataset",
                                               "language", "seeds"), c))
        with (out_dir / "per_language.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(per_lang_rows)

    # ---- macro / micro averages, per (task, mode) ----
    agg_rows = []
    by_task: dict[tuple, list[tuple]] = defaultdict(list)
    for key, metrics in grouped.items():
        by_task[(key[0], key[1])].append((key, metrics))

    for (task, mode), entries in sorted(by_task.items()):
        headline = HEADLINE.get(task, "accuracy")
        metric_names = sorted({m for _, ms in entries for m in ms})
        row = {"task": task, "mode": mode, "languages": len(entries),
               "headline_metric": headline}
        for m in metric_names:
            per_lang_means, weights = [], []
            for key, ms in entries:
                if m in ms:
                    per_lang_means.append(stats.mean(ms[m]))
                    weights.append(sizes.get(key, 1))
            if not per_lang_means:
                continue
            row[f"macro_{m}"] = round(stats.mean(per_lang_means), 4)
            tot = sum(weights) or 1
            row[f"micro_{m}"] = round(
                sum(v * w for v, w in zip(per_lang_means, weights)) / tot, 4)
        agg_rows.append(row)

    if agg_rows:
        cols = sorted({k for r in agg_rows for k in r},
                      key=lambda c: (c not in ("task", "mode", "languages",
                                               "headline_metric"), c))
        with (out_dir / "aggregate_averages.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(agg_rows)

    # ---- languages present in the inventory with no completed run ----
    missing = []
    cov = Path(args.coverage)
    if cov.exists():
        done = {(r["dataset"], r["language"]) for r in runs}
        with cov.open() as fh:
            for rec in csv.DictReader(fh):
                if (rec["dataset"], rec["language_code"]) not in done:
                    missing.append(rec)
        if missing:
            with (out_dir / "not_run.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(missing[0]))
                w.writeheader()
                w.writerows(missing)

    # ---- rendered report ----
    lines = [
        "# Expanded Multilingual Evaluation — Results",
        "",
        "> **Expanded multilingual evaluation runs. Not paper results.**",
        "> The published paper results are unchanged and live in",
        "> `reports/RESULTS_SUMMARY.md`.",
        "",
        f"Runs aggregated: **{len(runs)}**",
        "",
        "Macro average is the unweighted mean over languages; micro average",
        "weights each language by its test-set size. Zero-shot languages are",
        "reported separately and are excluded from fine-tuned averages.",
        "",
        "## Aggregate",
        "",
    ]
    for row in agg_rows:
        h = row["headline_metric"]
        lines.append(f"### {row['task']} ({row['mode']}) — {row['languages']} languages")
        lines.append("")
        lines.append("| Average | " + h + " |")
        lines.append("|---|---|")
        for kind in ("macro", "micro"):
            v = row.get(f"{kind}_{h}")
            if v is not None:
                lines.append(f"| {kind} | {v * 100:.2f} |")
        lines.append("")

    lines += ["## Per language", "",
              "| Task | Mode | Dataset | Language | Headline | Seeds |",
              "|---|---|---|---|---|---|"]
    for r in per_lang_rows:
        h = HEADLINE.get(r["task"], "accuracy")
        mu, sd = r.get(f"{h}_mean"), r.get(f"{h}_std")
        cell = f"{mu * 100:.2f} ± {sd * 100:.2f}" if mu is not None else "—"
        lines.append(f"| {r['task']} | {r['mode']} | {r['dataset']} | "
                     f"{r['language']} | {cell} | {r['seeds']} |")

    if missing:
        lines += ["", "## Not run", "",
                  "| Dataset | Language | Task |", "|---|---|---|"]
        lines += [f"| {m['dataset']} | {m['language_code']} | {m['task']} |"
                  for m in missing]

    (out_dir / "MULTILINGUAL_RESULTS.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")

    print(f"aggregated {len(runs)} runs")
    print(f"  {out_dir/'per_language.csv'}")
    print(f"  {out_dir/'aggregate_averages.csv'}")
    print(f"  {out_dir/'MULTILINGUAL_RESULTS.md'}")
    if missing:
        print(f"  {out_dir/'not_run.csv'} ({len(missing)} languages with no run)")


if __name__ == "__main__":
    main()
