"""Multi-seed aggregation.

Collects per-seed `results.json` files and reduces them to mean/std/min/max/
median. Per-seed values are always preserved alongside the summary -- a mean
without its spread is not reportable on datasets this small.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Aggregate:
    metric: str
    n_seeds: int
    seeds: list[int]
    values: list[float]
    mean: float
    std: float
    min: float
    max: float
    median: float

    def as_dict(self) -> dict:
        return asdict(self)

    def formatted(self, digits: int = 2) -> str:
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"


def aggregate_values(metric: str, per_seed: dict[int, float]) -> Aggregate:
    seeds = sorted(per_seed)
    vals = [float(per_seed[s]) for s in seeds]
    return Aggregate(
        metric=metric, n_seeds=len(vals), seeds=seeds, values=vals,
        mean=round(statistics.mean(vals), 6),
        # stdev needs >=2 points; a single seed has no dispersion, not zero error.
        std=round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
        min=round(min(vals), 6), max=round(max(vals), 6),
        median=round(statistics.median(vals), 6),
    )


# Checkpoint subtrees that hold runs from models other than VEXMLM. The paper's
# tables report VEXMLM, so baseline and diagnostic runs must not be folded into
# them -- they live under checkpoints/ purely for storage convenience.
NON_VEXMLM_DIRS = ("baseline-", "tiquad-diagnostic")

# The repaired (SentencePiece-merged) model is a *different model* from the
# original add_tokens VEXMLM, not another seed of it. Averaging the two would
# report a blend of, say, AmQA 20.30 and 35.50 as one number. Runs are tagged by
# variant here so tables can separate them; SPM_DIR is excluded by default so the
# historical VEXMLM column keeps its original meaning.
SPM_DIR = "vexmlm-spm-stage2"


def variant_of(path: Path | str) -> str:
    """Which VEXMLM variant produced this run: 'spm_merge' or 'add_tokens'."""
    return "spm_merge" if SPM_DIR in str(path) else "add_tokens"


def collect_runs(root: Path, pattern: str = "**/results.json",
                 exclude: tuple[str, ...] = NON_VEXMLM_DIRS + (SPM_DIR,)) -> list[dict]:
    """Load every results.json under `root`, skipping excluded subtrees.

    Each run is tagged with `_variant`. By default the SentencePiece-merged runs
    are excluded; pass exclude=NON_VEXMLM_DIRS to include them (then group by
    `_variant`, never pooled).
    """
    runs = []
    for path in sorted(Path(root).glob(pattern)):
        if any(any(part.startswith(pref) for pref in exclude) for part in path.parts):
            log.debug("skipping excluded run %s", path)
            continue
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            data["_variant"] = variant_of(path)
            runs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("skipping %s: %s", path, exc)
    return runs


def group_and_aggregate(runs: list[dict], group_keys: tuple[str, ...] = ("task", "dataset"),
                        metric_prefix: str = "") -> dict:
    """Group runs by `group_keys` and aggregate each numeric metric across seeds.

    Runs from different model variants are never pooled. If the input mixes
    variants, `_variant` is appended to `group_keys` automatically: averaging a
    repaired-tokenizer run into the original model's seeds would report a blend
    of two different models as one number, and because runs are keyed by seed,
    the same-seed run of the other variant would silently overwrite rather than
    accumulate -- corrupting the seed count as well as the mean.
    """
    variants = {r.get("_variant") for r in runs if r.get("_variant")}
    if len(variants) > 1 and "_variant" not in group_keys:
        group_keys = group_keys + ("_variant",)
        log.info("input mixes model variants %s; grouping by _variant to avoid "
                 "pooling", sorted(variants))

    groups: dict[tuple, dict[str, dict[int, float]]] = {}
    for run in runs:
        key = tuple(str(run.get(k, "")) for k in group_keys)
        seed = int(run.get("seed", 0))
        metrics = run.get("metrics", {})
        bucket = groups.setdefault(key, {})
        for name, val in metrics.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if metric_prefix and not name.startswith(metric_prefix):
                continue
            bucket.setdefault(name, {})[seed] = float(val)

    out = {}
    for key, metrics in groups.items():
        label = "|".join(key)
        out[label] = {
            "group": dict(zip(group_keys, key)),
            "metrics": {name: aggregate_values(name, per_seed).as_dict()
                        for name, per_seed in sorted(metrics.items())},
        }
    return out


def check_seed_coverage(agg: dict, required: list[int], min_seeds: int = 3) -> list[str]:
    """Warn about groups missing seeds. Returns human-readable warnings."""
    warnings = []
    for label, entry in agg.items():
        for name, stats in entry["metrics"].items():
            missing = sorted(set(required) - set(stats["seeds"]))
            if missing:
                warnings.append(
                    f"{label}/{name}: {stats['n_seeds']} seeds, missing {missing}")
            if stats["n_seeds"] < min_seeds:
                warnings.append(
                    f"{label}/{name}: only {stats['n_seeds']} seed(s) -- "
                    f"below the {min_seeds}-seed reporting threshold")
            break  # one warning per group is enough
    return warnings
