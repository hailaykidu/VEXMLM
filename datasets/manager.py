"""Unified dataset manager: load, hash, cache, and document every dataset.

Guarantees:
  * every load returns a `datasets.DatasetDict`, whatever the on-disk format;
  * every load records a content hash and split statistics, so a run can always
    be traced to the exact data it used;
  * local files are supported for the datasets with no public loader.

    from datasets.manager import DatasetManager
    dm = DatasetManager()
    ds = dm.load("afrisenti", config="amh")
    dm.write_card("afrisenti")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from registry import REGISTRY, DatasetSpec, get

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


@dataclass
class SplitStats:
    rows: int
    columns: list[str]
    sha256: str
    label_distribution: dict | None = None
    mean_chars: float | None = None


class DatasetManager:
    """Loads datasets by registry key and records provenance for each."""

    def __init__(self, cache_dir: Path | None = None,
                 metadata_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or ROOT / "cache"
        self.metadata_dir = metadata_dir or ROOT / "metadata"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # --- loading ----------------------------------------------------------
    def load(self, key: str, *, config: str | None = None,
             local_path: str | Path | None = None):
        """Load a dataset as a DatasetDict, from the Hub or from local files."""
        from datasets import DatasetDict, load_dataset

        spec = get(key)
        if local_path:
            ds = self._load_local(spec, Path(local_path))
        elif spec.hub_id:
            ds = load_dataset(spec.hub_id, config or spec.hub_config,
                              cache_dir=str(self.cache_dir))
        else:
            raise SystemExit(
                f"{spec.name} has no public loader. Supply --local-path.\n"
                f"  {spec.notes}")

        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        self._record(spec, ds, config=config, local_path=local_path)
        return ds

    def _load_local(self, spec: DatasetSpec, path: Path):
        """Load from local files, dispatching on task and file extension."""
        from datasets import Dataset, DatasetDict, load_dataset

        if path.is_file():
            return DatasetDict({"train": self._load_file(spec, path)})

        splits = {}
        for split in spec.splits:
            for pat in (f"{split}.*", f"*{split}*"):
                hits = sorted(p for p in path.glob(pat) if p.is_file())
                if hits:
                    splits[split] = self._load_file(spec, hits[0])
                    break
        if not splits:
            raise SystemExit(f"no split files found under {path} for {spec.key}")
        return DatasetDict(splits)

    def _load_file(self, spec: DatasetSpec, path: Path):
        from datasets import Dataset, load_dataset

        suffix = path.suffix.lower()
        if suffix in (".conll", ".conllu", ".txt") and spec.task == "ner":
            return Dataset.from_list(read_conll(path))
        if suffix == ".txt":
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            return Dataset.from_dict({"text": lines})
        if suffix in (".json", ".jsonl"):
            if spec.task == "qa":
                return Dataset.from_list(read_squad(path))
            return load_dataset("json", data_files=str(path), split="train")
        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            return load_dataset("csv", data_files=str(path), split="train", sep=sep)
        raise SystemExit(f"unsupported file type for {spec.key}: {path}")

    # --- provenance --------------------------------------------------------
    def _record(self, spec: DatasetSpec, ds, **ctx) -> dict:
        stats = {name: asdict(self._stats(split, spec)) for name, split in ds.items()}
        record = {
            "dataset": spec.as_dict(),
            "context": {k: str(v) for k, v in ctx.items() if v},
            "splits": stats,
            "total_rows": sum(s["rows"] for s in stats.values()),
        }
        out = self.metadata_dir / f"{spec.key}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        log.info("%s: %s rows -> %s", spec.key,
                 {k: v["rows"] for k, v in stats.items()}, out)
        return record

    @staticmethod
    def _stats(split, spec: DatasetSpec) -> SplitStats:
        h = hashlib.sha256()
        text_col = spec.text_field if spec.text_field in split.column_names else None
        if text_col is None:
            for cand in ("text", "tweet", "question", "tokens", "context"):
                if cand in split.column_names:
                    text_col = cand
                    break

        mean_chars = None
        if text_col:
            total = 0
            for val in split[text_col]:
                s = " ".join(val) if isinstance(val, list) else str(val)
                h.update(s.encode("utf-8"))
                total += len(s)
            mean_chars = round(total / len(split), 2) if len(split) else 0.0
        else:
            h.update(repr(split[:100]).encode("utf-8"))

        labels = None
        label_col = spec.label_field if spec.label_field in split.column_names else None
        if label_col is None and "label" in split.column_names:
            label_col = "label"
        if label_col:
            counts: dict = {}
            for v in split[label_col]:
                counts[str(v)] = counts.get(str(v), 0) + 1
            labels = dict(sorted(counts.items()))

        return SplitStats(rows=len(split), columns=list(split.column_names),
                          sha256=h.hexdigest()[:32], label_distribution=labels,
                          mean_chars=mean_chars)

    # --- documentation -----------------------------------------------------
    def write_card(self, key: str, out_dir: Path | None = None) -> Path:
        """Emit DATASET_CARD.md from the registry entry plus recorded stats."""
        spec = get(key)
        meta_path = self.metadata_dir / f"{spec.key}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

        out_dir = out_dir or ROOT / "cards"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{spec.key}.md"

        lines = [
            f"# Dataset Card — {spec.name}", "",
            "| Field | Value |", "|---|---|",
            f"| Key | `{spec.key}` |",
            f"| Task | {spec.task} |",
            f"| Languages | {', '.join(spec.languages)} |",
            f"| Licence | {spec.license} |",
            f"| Source | {spec.url or 'n/a'} |",
            f"| Hub ID | {spec.hub_id or '**none — supply locally**'} |",
            "",
        ]
        if spec.notes:
            lines += ["## Notes", "", spec.notes, ""]

        if meta:
            lines += ["## Split statistics", "",
                      "| Split | Rows | SHA-256 (32) | Mean chars |", "|---|---|---|---|"]
            for name, s in meta["splits"].items():
                lines.append(f"| {name} | {s['rows']:,} | `{s['sha256']}` | "
                             f"{s['mean_chars'] if s['mean_chars'] is not None else '—'} |")
            lines.append("")
            for name, s in meta["splits"].items():
                if s.get("label_distribution"):
                    lines += [f"### Label distribution — {name}", "",
                              "| Label | Count |", "|---|---|"]
                    lines += [f"| {k} | {v:,} |" for k, v in s["label_distribution"].items()]
                    lines.append("")
        else:
            lines += ["## Split statistics", "",
                      "_Not yet loaded. Run the manager to populate._", ""]

        if spec.citation:
            lines += ["## Citation", "", "```bibtex", spec.citation, "```", ""]

        out.write_text("\n".join(lines))
        log.info("wrote %s", out)
        return out


# --- format readers ---------------------------------------------------------

def read_conll(path: Path) -> list[dict]:
    """Read CoNLL-format NER data into {'tokens', 'ner_tags'} records."""
    records, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            if tokens:
                records.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
            continue
        parts = line.split()
        if len(parts) >= 2:
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        records.append({"tokens": tokens, "ner_tags": tags})
    return records


def read_squad(path: Path) -> list[dict]:
    """Read SQuAD-style JSON (nested or flat JSONL) into flat QA records."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if "data" not in data:
        raise SystemExit(f"unrecognized QA JSON structure in {path}")

    out = []
    for article in data["data"]:
        for para in article.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                answers = qa.get("answers", [])
                out.append({
                    "id": qa.get("id", str(len(out))),
                    "title": article.get("title", ""),
                    "context": context,
                    "question": qa["question"],
                    "answers": {
                        "text": [a["text"] for a in answers],
                        "answer_start": [a["answer_start"] for a in answers],
                    },
                })
    return out
