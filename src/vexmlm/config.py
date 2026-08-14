"""Configuration loading with `inherits:` support and hashing.

Configs are plain YAML. A config may declare `inherits: <relative path>`; parent
and child are deep-merged, child wins. The resolved config is hashed so every
experiment records exactly which settings produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path, *, _seen: set[Path] | None = None) -> dict:
    """Load a YAML config, resolving `inherits:` chains."""
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular inherits detected at {path}")
    _seen.add(path)

    data = yaml.safe_load(path.read_text()) or {}
    parent_ref = data.pop("inherits", None)
    if parent_ref:
        parent_path = (path.parent / parent_ref).resolve()
        data = deep_merge(load_config(parent_path, _seen=_seen), data)
    return data


def config_hash(cfg: dict, length: int = 12) -> str:
    """Stable hash of a resolved config, for experiment provenance."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Fetch a nested value: get(cfg, 'pretraining.learning_rate')."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides of the form `section.key=value` (YAML-parsed)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out
