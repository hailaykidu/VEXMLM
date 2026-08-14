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


def collect_runs(root: Path, pattern: str = "**/results.json") -> list[dict]:
    """Load every results.json under `root`."""
    runs = []
    for path in sorted(Path(root).glob(pattern)):
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            runs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("skipping %s: %s", path, exc)
    return runs


def group_and_aggregate(runs: list[dict], group_keys: tuple[str, ...] = ("task", "dataset"),
                        metric_prefix: str = "") -> dict:
    """Group runs by `group_keys` and aggregate each numeric metric across seeds."""
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
