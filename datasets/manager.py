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
        # Resolution order: explicit --local-path, then this repository's own
        # datasets/raw/<local_dir> (populated by scripts/acquire_datasets.py),
        # then the Hub. The repository never reads through to an external one.
        resolved = Path(local_path) if local_path else None
        if resolved is None and spec.local_dir:
            own = ROOT / "raw" / spec.local_dir
            if own.is_dir() and any(own.iterdir()):
                resolved = own
                log.info("%s: using in-repository copy %s", key, own)

        if resolved is not None:
            ds = self._load_local(spec, resolved)
        elif spec.hub_id:
            ds = load_dataset(spec.hub_id, config or spec.hub_config,
                              cache_dir=str(self.cache_dir))
        else:
            raise SystemExit(
                f"{spec.name} has no public loader and no local copy.\n"
                f"  Run: python scripts/acquire_datasets.py --copy\n"
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

        # Corpora on disk name the dev split variously; map every alias onto the
        # canonical HuggingFace split name so downstream code sees 'validation'.
        aliases = {
            "train": ("train",),
            "validation": ("validation", "dev", "valid", "val"),
            "test": ("test", "eval"),
        }
        splits = {}
        for split in spec.splits:
            for alias in aliases.get(split, (split,)):
                hits = sorted(p for p in path.glob(f"{alias}.*")
                              if p.is_file() and p.name != "upstream_manifest.json")
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
        if suffix == ".parquet":
            # Already SQuAD-shaped on disk (id/question/context/answers), so it
            # needs no read_squad() reshaping the way the JSON corpora do.
            return load_dataset("parquet", data_files=str(path), split="train")
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

        # Source-file checksums, recorded at acquisition time.
        acq_path = self.metadata_dir / "acquisition.json"
        acq = {}
        if acq_path.exists():
            acq = json.loads(acq_path.read_text()).get("datasets", {}).get(spec.key, {})

        lines = [
            f"# Dataset Card — {spec.name}", "",
            "| Field | Value |", "|---|---|",
            f"| Key | `{spec.key}` |",
            f"| Task | {spec.task} |",
            f"| Languages | {', '.join(spec.languages)} |",
            f"| Version | {spec.version} |",
            f"| Licence | {spec.license} |",
            f"| Source | {spec.url or 'n/a'} |",
            f"| Hub ID | {spec.hub_id or '**none — local copy**'} |",
            f"| Imported from | `{acq.get('origin_dir') or acq.get('source_dir', 'n/a')}` |",
            f"| Imported on | {acq.get('first_imported', 'n/a')} |",
            "",
        ]

        if acq.get("files"):
            lines += ["## Checksums (source files)", "",
                      "| Split | Bytes | SHA-256 | Verified |", "|---|---|---|---|"]
            for split, info in acq["files"].items():
                verified = info.get("checksum_match")
                mark = "✅" if verified else ("—" if verified is None else "❌")
                lines.append(f"| {split} | {info['bytes']:,} | "
                             f"`{info['sha256'][:32]}` | {mark} |")
            lines.append("")
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

# Known upstream tag typos, repaired on read. Each entry is documented in
# docs/DATASET_PROVENANCE.md; the source files are never modified.
CONLL_TAG_REPAIRS = {
    "B-LO": "B-LOC",   # Tigrinya NER train.conll:1350, truncated 'B-LOC'
    "I-LO": "I-LOC",
}


def read_conll(path: Path, *, repair_tags: bool = True) -> list[dict]:
    """Read CoNLL-format NER data into {'tokens', 'ner_tags'} records.

    Malformed tags listed in CONLL_TAG_REPAIRS are corrected in memory and the
    substitutions logged. Leaving them would create spurious label classes that
    inflate the label set and distort macro-F1.
    """
    records, tokens, tags = [], [], []
    repairs: dict[str, int] = {}

    def flush() -> None:
        if tokens:
            records.append({"tokens": list(tokens), "ner_tags": list(tags)})
            tokens.clear()
            tags.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            flush()
            continue
        parts = line.split()
        if len(parts) >= 2:
            tag = parts[-1]
            if repair_tags and tag in CONLL_TAG_REPAIRS:
                repairs[tag] = repairs.get(tag, 0) + 1
                tag = CONLL_TAG_REPAIRS[tag]
            tokens.append(parts[0])
            tags.append(tag)
    flush()

    for bad, count in repairs.items():
        log.warning("%s: repaired %d occurrence(s) of malformed tag %r -> %r",
                    path.name, count, bad, CONLL_TAG_REPAIRS[bad])
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

    out: list[dict] = []
    dropped_unanswerable = 0

    def emit(article: dict, context: str, qas: list[dict]) -> None:
        nonlocal dropped_unanswerable
        for qa in qas:
            answers = qa.get("answers", []) or []
            texts = [a["text"] for a in answers]
            starts = [a["answer_start"] for a in answers]
            # answer_start == -1 marks an answer that is not present verbatim in
            # the context (abstractive). Span extraction cannot score it, and
            # feeding it in would silently train the model toward the CLS index.
            keep = [(t, s) for t, s in zip(texts, starts) if s is not None and s >= 0]
            if texts and not keep:
                dropped_unanswerable += 1
                continue
            out.append({
                "id": qa.get("id", str(len(out))),
                "title": article.get("title", ""),
                "context": context,
                "question": qa["question"],
                "answers": {"text": [t for t, _ in keep],
                            "answer_start": [s for _, s in keep]},
            })

    for article in data["data"]:
        if "paragraphs" in article:
            paragraphs = article["paragraphs"]
            # One AmQA article stores paragraphs as a dict rather than a list.
            if isinstance(paragraphs, dict):
                paragraphs = [paragraphs]
            for para in paragraphs:
                emit(article, para["context"], para.get("qas", []))
        elif "context" in article:
            # TIGQA ships a flat title/context/qas structure with no
            # intermediate 'paragraphs' level.
            emit(article, article["context"], article.get("qas", []))
        else:
            raise SystemExit(
                f"article in {path} has neither 'paragraphs' nor 'context'")

    if dropped_unanswerable:
        log.warning("%s: dropped %d question(s) whose answers are not present "
                    "verbatim in the context (answer_start == -1)",
                    path.name, dropped_unanswerable)
    return out
