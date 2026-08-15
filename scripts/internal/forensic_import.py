#!/usr/bin/env python3
"""Forensic import of pre-existing VEXMLM/EXLMR artifacts.

Every artifact is recorded with its source path, size, SHA-256, and mtime *before*
anything is copied. Nothing is copied without provenance; nothing is modified.

Small artifacts (code, configs, vocabularies) are copied into ``legacy/``.
Multi-hundred-MB weights are recorded by reference only -- they stay where they are,
because duplicating 3.5 GB of checkpoints into a git repository is not useful.

Usage:
    python scripts/forensic_import.py [--copy] [--out docs/FORENSIC_IMPORT_REPORT.md]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO = Path(__file__).resolve().parent.parent

# Artifacts to record. copy=True => small enough to vendor into legacy/.
SOURCES: list[dict] = [
    # --- ~/EXLMR : the vocabulary-expansion prototype -------------------------
    {"src": HOME / "EXLMR/EXLMR.py", "dest": "legacy/exlmr/EXLMR.py", "copy": True,
     "role": "Vocabulary expansion + Tigrinya sentiment fine-tune. The ONLY existing "
             "implementation of the paper's method."},
    {"src": HOME / "EXLMR/EXLMR.py.save", "dest": "legacy/exlmr/EXLMR.py.save", "copy": True,
     "role": "Earlier editor backup of EXLMR.py."},
    {"src": HOME / "EXLMR/EXLMR.sh", "dest": "legacy/exlmr/EXLMR.sh", "copy": True,
     "role": "SLURM submit script (1x A100-80GB, standby partition)."},
    {"src": HOME / "EXLMR/vocab.json", "dest": "legacy/exlmr/vocab.json", "copy": True,
     "role": "30,522 candidate tokens fed to tokenizer.add_tokens(). "
             "BYTE-LEVEL-BPE ENCODED -- see docs/TOKENIZER_DEFECT_REPORT.md. "
             "Generating script is MISSING."},
    {"src": HOME / "EXLMR/README.md", "dest": "legacy/exlmr/README.md", "copy": True,
     "role": "Prototype model card / pipeline description."},
    {"src": HOME / "EXLMR/Ex.out", "dest": "legacy/exlmr/Ex.out", "copy": True,
     "role": "stdout of the run that produced the 80.3% figure (job 50530)."},
    {"src": HOME / "EXLMR/cleaned_train.csv", "dest": None, "copy": False,
     "role": "Binary Tigrinya train set, 49,373 rows. PROVENANCE UNKNOWN -- "
             "no citation or licence. Release blocker."},
    {"src": HOME / "EXLMR/cleaned_test.csv", "dest": None, "copy": False,
     "role": "Binary Tigrinya test set, 3,999 rows. PROVENANCE UNKNOWN."},

    # --- ~/VEXMLM_Model : the shipped vocab-expanded checkpoint ---------------
    {"src": HOME / "VEXMLM_Model/config.json", "dest": "legacy/vexmlm_model/config.json",
     "copy": True, "role": "vocab_size=280147. Authoritative final vocabulary size."},
    {"src": HOME / "VEXMLM_Model/added_tokens.json",
     "dest": "legacy/vexmlm_model/added_tokens.json", "copy": True,
     "role": "30,145 added tokens, contiguous IDs 250002-280146. Mojibake -- see defect report."},
    {"src": HOME / "VEXMLM_Model/special_tokens_map.json",
     "dest": "legacy/vexmlm_model/special_tokens_map.json", "copy": True,
     "role": "Standard XLM-R special tokens; unmodified."},
    {"src": HOME / "VEXMLM_Model/sentencepiece.bpe.model", "dest": None, "copy": False,
     "role": "STOCK XLM-R SentencePiece model, 250,000 pieces. Byte-identical to the "
             "copies in EXLMR_Model/ and EXLMR_release/. NOT retrained."},
    {"src": HOME / "VEXMLM_Model/tokenizer_config.json", "dest": None, "copy": False,
     "role": "5.7 MB config, dominated by the added_tokens_decoder block."},
    {"src": HOME / "VEXMLM_Model/model.safetensors", "dest": None, "copy": False,
     "role": "1.2 GB. Embedding matrix (280147, 768). Source of the initialization proof."},

    # --- sibling model dirs ---------------------------------------------------
    {"src": HOME / "EXLMR_Model/config.json", "dest": "legacy/exlmr/EXLMR_Model.config.json",
     "copy": True, "role": "Vocab-extended base model, pre-fine-tuning. transformers 4.53.2."},
    {"src": HOME / "EXLMR_release/config.json", "dest": "legacy/exlmr/EXLMR_release.config.json",
     "copy": True, "role": "Release candidate config."},
    {"src": HOME / "EXLMR_release/README.md", "dest": "legacy/exlmr/EXLMR_release.README.md",
     "copy": True, "role": "Public model card for the released classifier."},
]


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="actually copy artifacts into legacy/")
    ap.add_argument("--out", default="docs/FORENSIC_IMPORT_REPORT.md")
    args = ap.parse_args()

    records = []
    for spec in SOURCES:
        src: Path = spec["src"]
        rec = {"source": str(src), "role": spec["role"], "dest": spec["dest"]}
        if not src.exists():
            rec |= {"status": "MISSING", "bytes": None, "sha256": None, "mtime": None}
        else:
            st = src.stat()
            rec |= {
                "status": "present",
                "bytes": st.st_size,
                "sha256": sha256(src),
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }
            if args.copy and spec["dest"]:
                dst = REPO / spec["dest"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                rec["copied"] = True
        records.append(rec)

    manifest = REPO / "legacy/PROVENANCE.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(
        {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
         "host": os.uname().nodename, "records": records}, indent=2))
    print(f"wrote {manifest}")
    for r in records:
        flag = "  " if r["status"] == "present" else "!!"
        size = f"{r['bytes']:,}" if r["bytes"] else "-"
        print(f"{flag} {r['status']:8} {size:>15}  {r['sha256'][:16] if r['sha256'] else '-':16}  {r['source']}")


if __name__ == "__main__":
    main()
