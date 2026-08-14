"""Experiment modes: debug / research / official.

Modes control scale and reporting only. They never alter the method: learning
rate, batch size, MLM probability, and initialization come from the paper and
are identical in all three. What changes is how much data is used, how many
seeds are run, and whether the output may be reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "modes.yaml"


@dataclass
class Mode:
    name: str
    description: str = ""
    label: str = ""
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int = -1
    num_train_epochs: int | None = None
    seeds: list[int] = field(default_factory=lambda: [42])
    tracking: bool = True
    report_results: bool = True
    single_seed_warning: bool = False
    require_all_seeds: bool = False

    @property
    def reportable(self) -> bool:
        """Only `official` produces numbers fit for the paper."""
        return self.name == "official"

    def banner(self) -> str:
        return f"[{self.label or self.name.upper()}]"


def load_mode(name: str | None = None, path: Path | None = None) -> Mode:
    data = yaml.safe_load((path or CONFIG).read_text())
    name = name or data.get("default_mode", "research")
    if name not in data["modes"]:
        raise SystemExit(f"unknown mode {name!r}; expected one of {list(data['modes'])}")
    spec = dict(data["modes"][name])
    spec.pop("description", None)
    mode = Mode(name=name, description=data["modes"][name].get("description", ""), **spec)

    if not mode.reportable:
        log.warning("mode=%s -- %s", name, mode.label or "not the official protocol")
    if mode.single_seed_warning and len(mode.seeds) == 1:
        log.warning("single seed (%s): variance is unmeasured; do not report this "
                    "as a paper result. Use --mode official for reportable numbers.",
                    mode.seeds[0])
    return mode


def apply_to_config(cfg: dict, mode: Mode, section: str) -> dict:
    """Overlay a mode's scale limits onto a resolved config section."""
    out = dict(cfg)
    sec = dict(out.get(section, {}))
    if mode.max_steps != -1:
        sec["max_steps"] = mode.max_steps
    if mode.num_train_epochs is not None:
        sec["num_train_epochs"] = mode.num_train_epochs
    out[section] = sec
    out["_mode"] = {
        "name": mode.name, "label": mode.label, "seeds": mode.seeds,
        "reportable": mode.reportable,
        "max_train_samples": mode.max_train_samples,
        "max_eval_samples": mode.max_eval_samples,
    }
    return out


def subsample(dataset_dict, mode: Mode, seed: int = 42):
    """Truncate splits per the mode's sample limits (debug mode only)."""
    if mode.max_train_samples is None and mode.max_eval_samples is None:
        return dataset_dict
    for split in list(dataset_dict.keys()):
        limit = (mode.max_train_samples if split == "train" else mode.max_eval_samples)
        if limit and len(dataset_dict[split]) > limit:
            dataset_dict[split] = dataset_dict[split].shuffle(seed=seed).select(range(limit))
            log.info("mode=%s: %s truncated to %d rows", mode.name, split, limit)
    return dataset_dict
