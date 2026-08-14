#!/usr/bin/env python3
"""Normalize raw corpora into train/dev splits for tokenizer training and Stage 1.

Reads datasets/raw/{amharic,tigrinya}/*.txt and writes normalized, deduplicated
splits to datasets/processed/. Also emits an OOV word list per language (words
held out of training) for the Table 3 evaluation.

    python datasets/prepare.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vexmlm import config as cfgmod  # noqa: E402
from vexmlm.geez import ethiopic_ratio, normalize  # noqa: E402

log = logging.getLogger(__name__)


def prepare_language(name: str, raw_dir: Path, out_dir: Path, cfg: dict,
                     dev_fraction: float, seed: int) -> dict:
    files = sorted(p for p in raw_dir.glob("*.txt") if p.is_file())
    if not files:
        log.warning("no .txt files in %s — skipping %s", raw_dir, name)
        return {}

    min_len = cfgmod.get(cfg, "tokenizer.min_sentence_length", 10)
    min_ratio = cfgmod.get(cfg, "tokenizer.min_ethiopic_ratio", 0.5)

    seen: set[str] = set()
    kept: list[str] = []
    read = dropped_short = dropped_script = dropped_dup = 0

    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            read += 1
            text = normalize(line)
            if len(text) < min_len:
                dropped_short += 1
                continue
            if ethiopic_ratio(text) < min_ratio:
                dropped_script += 1
                continue
            if text in seen:
                dropped_dup += 1
                continue
            seen.add(text)
            kept.append(text)

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_dev = max(1, int(len(kept) * dev_fraction)) if kept else 0
    dev, train = kept[:n_dev], kept[n_dev:]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / f"{name}.dev.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")

    # OOV list: words appearing in dev but never in train -- genuinely unseen.
    train_words = {w for line in train for w in line.split()}
    oov = sorted({w for line in dev for w in line.split()} - train_words)
    (out_dir / f"{name}.oov.txt").write_text("\n".join(oov) + "\n", encoding="utf-8")

    stats = {"language": name, "files": len(files), "lines_read": read,
             "kept": len(kept), "train": len(train), "dev": len(dev),
             "oov_words": len(oov), "dropped_short": dropped_short,
             "dropped_low_ethiopic": dropped_script, "dropped_duplicate": dropped_dup}
    log.info("%s: %d read -> %d kept (train %d / dev %d), %d OOV words",
             name, read, len(kept), len(train), len(dev), len(oov))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--raw-dir", default=str(ROOT / "datasets/raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "datasets/processed"))
    ap.add_argument("--dev-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = cfgmod.load_config(args.config)
    raw, out = Path(args.raw_dir), Path(args.out_dir)

    report = {}
    for name in ("amharic", "tigrinya"):
        stats = prepare_language(name, raw / name, out, cfg,
                                 args.dev_fraction, args.seed)
        if stats:
            report[name] = stats

    if not report:
        raise SystemExit(
            f"no corpora found under {raw}/. Place raw .txt files in "
            f"{raw}/amharic/ and {raw}/tigrinya/ — see docs/DATA_SETUP.md.")

    out.mkdir(parents=True, exist_ok=True)
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
