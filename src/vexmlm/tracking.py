"""Experiment tracking (MLflow) with full provenance capture.

Every run records: git commit and dirty state, seed, config hash, dataset
hashes, tokenizer hash, checkpoint hash, hardware, and metrics. If MLflow is not
installed the tracker degrades to JSON-on-disk rather than failing the run --
losing an experiment because a logging library is missing is never acceptable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def git_info(repo: Path | None = None) -> dict:
    repo = repo or Path(__file__).resolve().parent.parent.parent

    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(args, cwd=repo, capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = _run("git", "status", "--porcelain")
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "uncommitted_files": len(status.splitlines()) if status else 0,
    }


def hash_path(path: str | Path, chunk: int = 1 << 20) -> str | None:
    """SHA-256 of a file, or of a directory's sorted file digests."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    if p.is_file():
        with open(p, "rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
        return h.hexdigest()
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix not in (".log", ".out", ".err"):
            h.update(f.name.encode())
            with open(f, "rb") as fh:
                while block := fh.read(chunk):
                    h.update(block)
    return h.hexdigest()


@dataclass
class RunContext:
    """Everything needed to identify a run after the fact."""
    run_name: str
    stage: str                      # tokenizer | expansion | pretraining | finetuning | eval
    task: str | None = None
    language: str | None = None
    seed: int = 42
    config_hash: str | None = None
    dataset_hashes: dict = field(default_factory=dict)
    tokenizer_hash: str | None = None
    checkpoint_hash: str | None = None
    hardware: dict = field(default_factory=dict)
    git: dict = field(default_factory=git_info)
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def as_flat_params(self) -> dict:
        flat = {
            "stage": self.stage, "task": self.task, "language": self.language,
            "seed": self.seed, "config_hash": self.config_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "git_commit": self.git.get("commit"),
            "git_branch": self.git.get("branch"),
            "git_dirty": self.git.get("dirty"),
        }
        flat |= {f"dataset_hash.{k}": v for k, v in self.dataset_hashes.items()}
        flat |= {f"hw.{k}": v for k, v in self.hardware.items()
                 if isinstance(v, (str, int, float, bool))}
        return {k: v for k, v in flat.items() if v is not None}


class Tracker:
    """MLflow tracker with a JSON fallback."""

    def __init__(self, experiment: str = "vexmlm",
                 tracking_uri: str = "./experiment_tracking/mlruns",
                 enabled: bool = True) -> None:
        self.experiment = experiment
        self.tracking_uri = tracking_uri
        self.enabled = enabled
        self._mlflow = None
        self._active = False
        self._fallback: dict = {}
        self._fallback_dir = Path("experiment_tracking/runs")

        if not enabled:
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except ImportError:
            log.warning("mlflow not installed -- falling back to JSON records in %s",
                        self._fallback_dir)

    def start(self, ctx: RunContext) -> "Tracker":
        self._ctx = ctx
        self._fallback = {"context": ctx.__dict__.copy(), "metrics": {}, "artifacts": []}
        if self._mlflow:
            self._mlflow.start_run(run_name=ctx.run_name)
            self._mlflow.log_params(ctx.as_flat_params())
            self._active = True
        log.info("run started: %s (stage=%s seed=%s)", ctx.run_name, ctx.stage, ctx.seed)
        return self

    def log_params(self, params: dict) -> None:
        clean = {k: v for k, v in _flatten(params).items()
                 if isinstance(v, (str, int, float, bool)) or v is None}
        self._fallback.setdefault("params", {}).update(clean)
        if self._active:
            # MLflow rejects >250 params per call and duplicate keys.
            items = list(clean.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i:i + 100]))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for k, v in numeric.items():
            self._fallback["metrics"].setdefault(k, []).append({"step": step, "value": v})
        if self._active:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self._fallback["artifacts"].append(str(path))
        if self._active and Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def log_dict(self, obj: dict, name: str) -> None:
        self._fallback.setdefault("dicts", {})[name] = obj
        if self._active:
            self._mlflow.log_dict(obj, name)

    def end(self, status: str = "FINISHED") -> None:
        self._fallback["status"] = status
        self._fallback["ended"] = datetime.now().isoformat(timespec="seconds")
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        name = getattr(self, "_ctx", None)
        stem = f"{name.run_name}_{name.seed}" if name else "run"
        out = self._fallback_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps(self._fallback, default=str, indent=2))
        if self._active:
            self._mlflow.end_run(status=status)
            self._active = False
        log.info("run ended (%s); record: %s", status, out)

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "_ctx"):
            self.end("FAILED" if exc_type else "FINISHED")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out |= _flatten(v, key)
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v)) if v else ""
        else:
            out[key] = v
    return out
