#!/usr/bin/env python3
"""Acquire the VEXMLM datasets: local copies first, remote only if missing.

Search order (repository isolation policy):
  1. this repository's own datasets/raw/;
  2. other local repositories on this machine (READ-ONLY sources);
  3. only if absent, report the official source URL for manual download.

Data is always **copied** into this repository, never moved, never symlinked:
a symlink would leave the official implementation depending on -- and reading
through to -- an external repository. Source directories are opened read-only
and are never modified.

Nothing synthetic is ever created. Every acquired file is checksummed and its
origin recorded in datasets/metadata/acquisition.json, which feeds
docs/DATASET_PROVENANCE.md and docs/IMPORTED_ARTIFACTS.md.

    python scripts/acquire_datasets.py --report        # search only
    python scripts/acquire_datasets.py --copy          # copy into datasets/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "datasets" / "raw"

# Candidate locations searched in order. First hit wins.
SOURCES: dict[str, dict] = {
    "tigqa": {
        "dest": "tigqa",
        "candidates": [
            HOME / "lgse-repro/data/qa/tigqa_squad",
            HOME / "TigrinyaTokenizer/LGSE-Project/data/qa/tigqa_squad",
        ],
        "files": {"train": "train.json", "validation": "dev.json", "test": "test.json"},
        "official_url": "https://zenodo.org/records/11423987",
        "citation": "Teklehaymanot et al. (2024), TIGQA",
        "language": "tir",
        "task": "qa",
    },
    "amqa": {
        "dest": "amqa",
        "candidates": [
            HOME / "lgse-repro/data/qa/amqa",
            HOME / "TigrinyaTokenizer/LGSE-Project/data/qa/amqa",
        ],
        "files": {"train": "train.json", "validation": "dev.json", "test": "test.json"},
        "official_url": "https://github.com/semantic-systems/amharic-qa",
        "citation": "Taffa et al. (2024), AmQA",
        "language": "amh",
        "task": "qa",
    },
    "masakhaner_amh": {
        "dest": "masakhaner_amh",
        "candidates": [HOME / "lgse-repro/data/ner/amharic"],
        "files": {"train": "train.conll", "validation": "dev.conll", "test": "test.conll"},
        "official_url": "https://github.com/masakhane-io/masakhane-ner",
        "citation": "Adelani et al. (2021), MasakhaNER",
        "language": "amh",
        "task": "ner",
    },
    "tigrinya_ner": {
        "dest": "tigrinya_ner",
        "candidates": [HOME / "lgse-repro/data/ner/tigrinya"],
        "files": {"train": "train.conll", "validation": "dev.conll", "test": "test.conll"},
        "official_url": "https://github.com/mehari-eng/Tigrinya-NER",
        "citation": "Yohannes and Amagasa (2022), Tigrinya NER",
        "language": "tir",
        "task": "ner",
    },
    "amharic_mlm": {
        "dest": "amharic",
        "candidates": [HOME / "lgse-repro/data/lapt"],
        "files": {"corpus": "amharic.txt"},
        "official_url": "https://github.com/asmelashteka/HornMT",
        "citation": "see docs/DATASET_PROVENANCE.md -- origin under verification",
        "language": "amh",
        "task": "mlm",
    },
    "tigrinya_mlm": {
        "dest": "tigrinya",
        "candidates": [HOME / "lgse-repro/data/lapt"],
        "files": {"corpus": "tigrinya.txt"},
        "official_url": "https://github.com/asmelashteka/HornMT",
        "citation": "see docs/DATASET_PROVENANCE.md -- origin under verification",
        "language": "tir",
        "task": "mlm",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def locate(spec: dict) -> tuple[Path | None, list[str], bool]:
    """Find the dataset, preferring this repository's own copy.

    Returns (directory, paths_checked, already_local). `already_local` is True
    when the data is already inside this repository, in which case nothing is
    copied again.
    """
    checked = []
    own = RAW / spec["dest"]
    checked.append(str(own))
    if own.is_dir() and all((own / fn).is_file() and not (own / fn).is_symlink()
                            for fn in spec["files"].values()):
        return own, checked, True

    for cand in spec["candidates"]:
        checked.append(str(cand))
        if not cand.is_dir():
            continue
        if all((cand / fn).is_file() for fn in spec["files"].values()):
            return cand, checked, False
    return None, checked, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="search only (default)")
    mode.add_argument("--copy", action="store_true",
                      help="copy into datasets/raw (sources are never modified)")
    ap.add_argument("--out", default="datasets/metadata/acquisition.json")
    args = ap.parse_args()

    records = {}
    missing = []
    imported = datetime.now().isoformat(timespec="seconds")

    # Preserve the ORIGINAL import provenance across re-runs. Once data lives in
    # this repository, later runs resolve to the in-repo copy; without this the
    # record would overwrite the external origin it came from and the
    # provenance trail would be lost.
    prior_path = REPO / args.out
    prior = {}
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text()).get("datasets", {})
        except (json.JSONDecodeError, OSError):
            prior = {}

    for key, spec in SOURCES.items():
        found, checked, already_local = locate(spec)
        rec = {
            "key": key, "task": spec["task"], "language": spec["language"],
            "official_url": spec["official_url"], "citation": spec["citation"],
            "searched": checked,
        }
        if not found:
            rec["status"] = "MISSING"
            missing.append((key, spec["official_url"]))
            print(f"MISSING  {key:16} -> download from {spec['official_url']}")
            records[key] = rec
            continue

        rec["status"] = "already_in_repository" if already_local else "found_in_local_repository"
        rec["source_dir"] = str(found)
        rec["files"] = {}
        dest_dir = RAW / spec["dest"]

        # Keep the first-recorded external origin; it is the real provenance.
        was = prior.get(key, {})
        origin = was.get("origin_dir") or (
            was.get("source_dir") if was.get("status") == "found_in_local_repository"
            else None)
        rec["origin_dir"] = origin or (None if already_local else str(found))
        if was.get("first_imported"):
            rec["first_imported"] = was["first_imported"]
        elif not already_local:
            rec["first_imported"] = imported

        for split, fname in spec["files"].items():
            src = found / fname
            info = {"source": str(src), "bytes": src.stat().st_size,
                    "sha256": sha256(src)}
            prev = was.get("files", {}).get(split, {})
            for carried in ("origin", "imported", "checksum_match"):
                if carried in prev:
                    info[carried] = prev[carried]
            if "origin" not in info and not already_local:
                info["origin"] = str(src)
            if args.copy and not already_local:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dst = dest_dir / fname
                # Replace any prior symlink with a real file: the repository must
                # not read through to an external source at training time.
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                shutil.copy2(src, dst)
                info["copied_to"] = str(dst)
                info["imported"] = imported
                info["copy_sha256"] = sha256(dst)
                info["checksum_match"] = info["copy_sha256"] == info["sha256"]
            rec["files"][split] = info

        # Carry forward any upstream manifest -- it is primary provenance.
        man = found / "manifest.json"
        if man.is_file():
            rec["upstream_manifest"] = json.loads(man.read_text())
            if args.copy and not already_local:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(man, dest_dir / "upstream_manifest.json")

        total = sum(f["bytes"] for f in rec["files"].values())
        tag = "IN-REPO " if already_local else "FOUND   "
        print(f"{tag} {key:16} {total:>12,} B  {found}")
        records[key] = rec

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mode": "copy" if args.copy else ("link" if args.link else "report"),
        "datasets": records,
    }, ensure_ascii=False, indent=2))

    print(f"\nwrote {out}")
    if missing:
        print("\nDownload the missing datasets manually, then re-run:")
        for key, url in missing:
            print(f"  {key}: {url}")
    else:
        print("All datasets resolved from the local workspace; nothing downloaded.")


if __name__ == "__main__":
    main()
