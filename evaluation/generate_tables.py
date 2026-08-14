#!/usr/bin/env python3
"""Generate the paper's result tables from recorded runs.

Emits:
    results/table2_parity.csv        intrinsic: parity + fertility + compression
    results/table3_oov_accuracy.csv  intrinsic: OOV word accuracy
    results/table4_downstream.csv    QA / NER / Sentiment, mean ± std over seeds
    results/table5_ablation.csv      the four ablation arms
    results/RESULTS.md               all of the above, rendered

Only measured values are written. A metric with no run behind it is emitted as
NOT YET MEASURED rather than being filled in or omitted silently.

    python evaluation/generate_tables.py --runs checkpoints --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

from aggregate import collect_runs, group_and_aggregate  # noqa: E402

log = logging.getLogger(__name__)

NOT_MEASURED = "NOT YET MEASURED"

# Table 5 arms, in the paper's order.
ABLATION_ARMS = [
    ("xlmr_baseline", "XLM-R"),
    ("expansion_random_init", "Vocab Expansion + Random Init"),
    ("expansion_mean_init", "Vocab Expansion + Mean Init"),
    ("full_vexmlm", "Full VEXMLM"),
]

# Primary metric per task (paper Sec. 5.2).
TASK_METRICS = {
    "qa": [("exact_match", "EM"), ("f1", "F1")],
    "ner": [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")],
    "sentiment": [("accuracy", "Accuracy")],
}


def _fmt(stats: dict | None, digits: int = 2) -> str:
    if not stats:
        return NOT_MEASURED
    if stats["n_seeds"] == 1:
        return f"{stats['mean']:.{digits}f} (1 seed)"
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def _find(agg: dict, task: str, dataset: str | None = None) -> dict | None:
    for entry in agg.values():
        g = entry["group"]
        if g.get("task") != task:
            continue
        if dataset and g.get("dataset") != dataset:
            continue
        return entry["metrics"]
    return None


def _pick(metrics: dict | None, name: str) -> dict | None:
    """Find a metric, tolerating eval_/test_ prefixes."""
    if not metrics:
        return None
    for key in (f"test_{name}", f"eval_{name}", name):
        if key in metrics:
            return metrics[key]
    return None


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))


def table2_parity(intrinsic: dict | None, out: Path) -> list[list]:
    header = ["Tokenizer", "Language", "Fertility", "Compression",
              "Parity (amh/tir)", "Parity valid"]
    rows: list[list] = []
    if intrinsic:
        for tname, entry in intrinsic.get("tokenizers", {}).items():
            par = entry.get("parity", {})
            for lang, m in entry.get("per_language", {}).items():
                rows.append([tname, lang, m.get("fertility", NOT_MEASURED),
                             m.get("compression", NOT_MEASURED),
                             par.get("value", NOT_MEASURED),
                             par.get("valid", NOT_MEASURED)])
    if not rows:
        rows = [[NOT_MEASURED] * len(header)]
    write_csv(out / "table2_parity.csv", header, rows)
    return rows


def table3_oov(intrinsic: dict | None, out: Path) -> list[list]:
    header = ["Tokenizer", "Language", "OOV Accuracy", "No-UNK Rate", "Words"]
    rows: list[list] = []
    if intrinsic:
        for tname, entry in intrinsic.get("tokenizers", {}).items():
            for lang, m in entry.get("oov", {}).items():
                rows.append([tname, lang, m.get("oov_accuracy", NOT_MEASURED),
                             m.get("no_unk_rate", NOT_MEASURED),
                             m.get("words", NOT_MEASURED)])
    if not rows:
        rows = [[NOT_MEASURED] * len(header)]
    write_csv(out / "table3_oov_accuracy.csv", header, rows)
    return rows


def table4_downstream(agg: dict, out: Path) -> list[list]:
    header = ["Task", "Dataset", "Metric", "VEXMLM (mean ± std)", "Seeds"]
    rows = []
    seen = set()
    for entry in agg.values():
        g = entry["group"]
        task = g.get("task")
        if task not in TASK_METRICS:
            continue
        for name, label in TASK_METRICS[task]:
            stats = _pick(entry["metrics"], name)
            key = (task, g.get("dataset"), label)
            if key in seen:
                continue
            seen.add(key)
            rows.append([task, g.get("dataset", ""), label, _fmt(stats),
                         stats["n_seeds"] if stats else 0])
    if not rows:
        for task, metrics in TASK_METRICS.items():
            for _, label in metrics:
                rows.append([task, "", label, NOT_MEASURED, 0])
    write_csv(out / "table4_downstream.csv", header, rows)
    return rows


def table5_ablation(agg: dict, out: Path) -> list[list]:
    header = ["Arm", "Description", "QA EM", "QA F1", "NER Macro-F1", "SA Accuracy", "Seeds"]
    rows = []
    for arm, label in ABLATION_ARMS:
        entry = None
        for e in agg.values():
            if e["group"].get("arm") == arm or e["group"].get("dataset") == arm:
                entry = e["metrics"]
                break
        qa_em = _pick(entry, "exact_match")
        qa_f1 = _pick(entry, "f1")
        ner = _pick(entry, "macro_f1")
        sa = _pick(entry, "accuracy")
        seeds = next((s["n_seeds"] for s in (qa_em, qa_f1, ner, sa) if s), 0)
        rows.append([arm, label, _fmt(qa_em), _fmt(qa_f1), _fmt(ner), _fmt(sa), seeds])
    write_csv(out / "table5_ablation.csv", header, rows)
    return rows


def render_markdown(tables: dict[str, tuple[list[str], list[list]]],
                    out: Path, warnings: list[str]) -> None:
    lines = ["# VEXMLM — Results", "",
             "Generated by `evaluation/generate_tables.py` from recorded runs.",
             f"Cells marked `{NOT_MEASURED}` have no run behind them yet.", ""]
    titles = {
        "table2": "Table 2 — Tokenizer parity",
        "table3": "Table 3 — OOV word accuracy",
        "table4": "Table 4 — Downstream task performance",
        "table5": "Table 5 — Ablation study",
    }
    for key, (header, rows) in tables.items():
        lines += [f"## {titles.get(key, key)}", "",
                  "| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * len(header)) + "|"]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        lines.append("")
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    out.write_text("\n".join(lines))
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="checkpoints",
                    help="directory tree containing results.json files")
    ap.add_argument("--intrinsic", default="results/tokenizer_metrics.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(Path(args.runs))
    log.info("found %d run record(s) under %s", len(runs), args.runs)
    agg = group_and_aggregate(runs, group_keys=("task", "dataset"))

    intrinsic = None
    ipath = Path(args.intrinsic)
    if ipath.exists():
        intrinsic = json.loads(ipath.read_text())
    else:
        log.warning("no intrinsic metrics at %s -- Tables 2/3 will be unmeasured", ipath)

    warnings = []
    for label, entry in agg.items():
        for name, stats in entry["metrics"].items():
            missing = sorted(set(args.seeds) - set(stats["seeds"]))
            if missing:
                warnings.append(f"{label}: missing seeds {missing}")
            break
    if not runs:
        warnings.append("No runs found: all downstream tables are unmeasured.")

    t2 = table2_parity(intrinsic, out)
    t3 = table3_oov(intrinsic, out)
    t4 = table4_downstream(agg, out)
    t5 = table5_ablation(agg, out)

    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    render_markdown({
        "table2": (["Tokenizer", "Language", "Fertility", "Compression",
                    "Parity (amh/tir)", "Parity valid"], t2),
        "table3": (["Tokenizer", "Language", "OOV Accuracy", "No-UNK Rate", "Words"], t3),
        "table4": (["Task", "Dataset", "Metric", "VEXMLM (mean ± std)", "Seeds"], t4),
        "table5": (["Arm", "Description", "QA EM", "QA F1", "NER Macro-F1",
                    "SA Accuracy", "Seeds"], t5),
    }, out / "RESULTS.md", warnings)

    for w in warnings:
        log.warning(w)
    print(f"\ntables written to {out}/")


if __name__ == "__main__":
    main()
