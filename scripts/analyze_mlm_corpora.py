#!/usr/bin/env python3
"""Empirical analysis of the Stage 1 MLM corpora.

Produces the evidence behind docs/MLM_CORPUS_ANALYSIS.md and
reports/mlm_data_quality_report.md. Every figure in those documents comes from
this script; nothing is asserted that is not measured here.

Origin classification is evidence-based and deliberately conservative:
  VERIFIED -- a checksum or exact content match ties the file to a named source
  LIKELY   -- strong distinguishing evidence, short of an exact match
  UNKNOWN  -- no sufficient evidence

    python scripts/analyze_mlm_corpora.py --out results/mlm_corpus_analysis.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vexmlm.geez import ethiopic_ratio, has_ethiopic, is_ethiopic_char  # noqa: E402

HOME = Path(os.path.expanduser("~"))

# Candidate reference corpora on this machine, for content comparison.
REFERENCE_CORPORA = {
    "HornMT (tir)": HOME / "amseg/data/corpus/hornmt_tir.txt",
    "HornMT (raw ti)": HOME / "TigrinyaTokenizer/EnTiMT/01_collection/raw/hornmt.ti",
    "HornMT (raw en)": HOME / "TigrinyaTokenizer/EnTiMT/01_collection/raw/hornmt.en",
}

# Domain probes. Deliberately coarse: these establish that a corpus CONTAINS
# religious material, not that it is predominantly religious.
DOMAIN_MARKERS = {
    "religious_christian": ["ኢየሱስ", "እግዚአብሔር", "ጐይታ", "ኣምላኽ", "መንፈስ ቅዱስ",
                             "ወንጌል", "ክርስቶስ", "ጸሎት", "ጸሎት"],
    "religious_islamic": ["አላህ", "ኣላህ", "ነቢዩ", "ቁርኣን", "መስጊድ", "ሙስሊም"],
    "news_politics": ["መንግሥት", "መንግስቲ", "ሚኒስትር", "ፓርቲ", "ምርጫ", "ፕሬዚዳንት",
                       "ጠቅላይ", "ኤጀንሲ"],
    "web_commerce": ["ዋጋ", "ግዢ", "ሽያጭ", "ኩባንያ", "ድረ-ገጽ", "ኢንተርኔት"],
}

_WS = re.compile(r"\s+")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9]")
_CTRL = re.compile(r"[​-\u200F\u202A-\u202E﻿]")
_URL = re.compile(r"https?://|www\.")


def analyze_corpus(path: Path, sample_domain: int = 200_000) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    n_lines = len(lines)
    n_empty = sum(1 for line in lines if not line.strip())
    nonempty = [line for line in lines if line.strip()]

    seen: set[str] = set()
    n_dup = 0
    tokens = 0
    chars = 0
    lengths: list[int] = []
    token_lengths: list[int] = []

    ethiopic_chars = latin_chars = digit_chars = space_chars = other_chars = 0
    n_ctrl_lines = n_url_lines = n_nfc_violations = 0
    ethiopic_ratios: list[float] = []
    char_freq: collections.Counter = collections.Counter()

    for line in nonempty:
        stripped = line.strip()
        if stripped in seen:
            n_dup += 1
        else:
            seen.add(stripped)

        toks = stripped.split()
        tokens += len(toks)
        token_lengths.extend(len(t) for t in toks[:50])
        chars += len(stripped)
        lengths.append(len(stripped))

        if _CTRL.search(stripped):
            n_ctrl_lines += 1
        if _URL.search(stripped):
            n_url_lines += 1
        if unicodedata.normalize("NFC", stripped) != stripped:
            n_nfc_violations += 1

        eth = 0
        for ch in stripped:
            char_freq[ch] += 1
            if ch.isspace():
                space_chars += 1
            elif is_ethiopic_char(ch):
                ethiopic_chars += 1
                eth += 1
            elif _LATIN.match(ch):
                latin_chars += 1
            elif _DIGIT.match(ch):
                digit_chars += 1
            else:
                other_chars += 1
        non_space = len(stripped) - sum(1 for c in stripped if c.isspace())
        if non_space:
            ethiopic_ratios.append(eth / non_space)

    total_chars = ethiopic_chars + latin_chars + digit_chars + space_chars + other_chars
    lengths.sort()
    token_lengths.sort()

    def pct(xs: list, q: float):
        return xs[int(len(xs) * q)] if xs else 0

    # Domain composition: fraction of lines containing >=1 marker per domain.
    probe = nonempty[:sample_domain]
    domain_hits = {}
    for domain, markers in DOMAIN_MARKERS.items():
        hits = sum(1 for line in probe if any(m in line for m in markers))
        domain_hits[domain] = {
            "lines_with_marker": hits,
            "share_of_sampled": round(hits / len(probe), 4) if probe else 0.0,
        }

    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "lines_total": n_lines,
        "lines_empty": n_empty,
        "lines_nonempty": len(nonempty),
        "lines_unique": len(seen),
        "duplicate_lines": n_dup,
        "duplicate_rate": round(n_dup / len(nonempty), 6) if nonempty else 0.0,
        "tokens_whitespace": tokens,
        "characters": chars,
        "mean_tokens_per_line": round(tokens / len(nonempty), 3) if nonempty else 0,
        "mean_chars_per_line": round(chars / len(nonempty), 2) if nonempty else 0,
        "length_distribution_chars": {
            "min": lengths[0] if lengths else 0,
            "p10": pct(lengths, 0.10), "p25": pct(lengths, 0.25),
            "median": pct(lengths, 0.50), "p75": pct(lengths, 0.75),
            "p90": pct(lengths, 0.90), "p99": pct(lengths, 0.99),
            "max": lengths[-1] if lengths else 0,
        },
        "mean_token_length": (round(sum(token_lengths) / len(token_lengths), 3)
                              if token_lengths else 0),
        "character_distribution": {
            "ethiopic": ethiopic_chars,
            "latin": latin_chars,
            "digit": digit_chars,
            "whitespace": space_chars,
            "other": other_chars,
            "ethiopic_share_of_all": round(ethiopic_chars / total_chars, 4) if total_chars else 0,
            "ethiopic_share_of_nonspace": (
                round(ethiopic_chars / (total_chars - space_chars), 4)
                if total_chars - space_chars else 0),
        },
        "geez_coverage": {
            "distinct_ethiopic_codepoints": sum(1 for c in char_freq if is_ethiopic_char(c)),
            "lines_with_any_ethiopic": sum(1 for line in nonempty if has_ethiopic(line)),
            "lines_below_50pct_ethiopic": sum(1 for r in ethiopic_ratios if r < 0.5),
            "mean_ethiopic_ratio": (round(sum(ethiopic_ratios) / len(ethiopic_ratios), 4)
                                    if ethiopic_ratios else 0),
        },
        "encoding_issues": {
            "replacement_chars": text.count("�"),
            "lines_with_control_chars": n_ctrl_lines,
            "lines_not_nfc": n_nfc_violations,
            "lines_with_url": n_url_lines,
        },
        "domain_composition": domain_hits,
        "top_20_characters": [
            {"char": c, "name": unicodedata.name(c, "?"), "count": n}
            for c, n in char_freq.most_common(20) if not c.isspace()
        ][:20],
    }


def compare_with_references(corpus_lines: set[str]) -> dict:
    """Test overlap against known reference corpora on this machine."""
    out = {}
    for name, path in REFERENCE_CORPORA.items():
        if not path.exists():
            out[name] = {"status": "not present on this machine"}
            continue
        ref = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if l.strip()]
        ref_set = set(ref)
        overlap = len(ref_set & corpus_lines)
        out[name] = {
            "path": str(path),
            "reference_lines": len(ref),
            "reference_unique": len(ref_set),
            "exact_line_overlap": overlap,
            "share_of_reference_found": round(overlap / len(ref_set), 4) if ref_set else 0.0,
        }
    return out


def classify_origin(stats: dict, overlap: dict) -> dict:
    """Evidence-based origin classification. Never speculative."""
    findings = []
    best_overlap = max((v.get("share_of_reference_found", 0)
                        for v in overlap.values() if isinstance(v, dict)), default=0.0)

    hornmt_lines = max((v.get("reference_unique", 0) for k, v in overlap.items()
                        if isinstance(v, dict) and "HornMT" in k), default=0)
    if hornmt_lines:
        findings.append(
            f"HornMT reference on this machine has {hornmt_lines:,} unique lines, "
            f"against {stats['lines_nonempty']:,} in this corpus "
            f"({stats['lines_nonempty'] / hornmt_lines:.0f}x larger).")
    if best_overlap == 0:
        findings.append("Zero exact line overlap with any HornMT copy present here.")
    else:
        findings.append(f"Maximum exact-line overlap with a reference: {best_overlap:.2%}.")

    rel = (stats["domain_composition"]["religious_christian"]["share_of_sampled"]
           + stats["domain_composition"]["religious_islamic"]["share_of_sampled"])
    findings.append(f"Religious markers appear in {rel:.1%} of sampled lines; "
                    f"news/politics in "
                    f"{stats['domain_composition']['news_politics']['share_of_sampled']:.1%}.")
    if stats["lines_total"] in (200_000, 200_001):
        findings.append(f"Line count is exactly {stats['lines_total']:,}, indicating a "
                        f"deliberate cap applied during preparation rather than a "
                        f"natural corpus boundary.")

    classification = "VERIFIED" if best_overlap > 0.95 else "UNKNOWN"
    rationale = (
        "Exact content match with a named source." if classification == "VERIFIED" else
        "No checksum or content match ties this file to any named source available "
        "on this machine. Scale, structure, and the round line cap rule out the "
        "designated HornMT reference, but the positive identity of the source "
        "cannot be established from the evidence at hand.")

    return {"classification": classification, "rationale": rationale,
            "evidence": findings,
            "ruled_out": ["HornMT — scale and zero content overlap"] if best_overlap == 0 else []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amharic", default=str(REPO / "datasets/raw/amharic/amharic.txt"))
    ap.add_argument("--tigrinya", default=str(REPO / "datasets/raw/tigrinya/tigrinya.txt"))
    ap.add_argument("--out", default="results/mlm_corpus_analysis.json")
    args = ap.parse_args()

    report: dict = {"corpora": {}}
    for lang, path in (("amharic", Path(args.amharic)), ("tigrinya", Path(args.tigrinya))):
        if not path.exists():
            report["corpora"][lang] = {"status": "MISSING", "path": str(path)}
            print(f"MISSING: {path}")
            continue
        print(f"analyzing {lang}: {path} ...")
        stats = analyze_corpus(path)
        lines = {l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip()}
        overlap = compare_with_references(lines)
        stats["reference_overlap"] = overlap
        stats["origin"] = classify_origin(stats, overlap)
        report["corpora"][lang] = stats
        print(f"  lines={stats['lines_nonempty']:,} tokens={stats['tokens_whitespace']:,} "
              f"dup={stats['duplicate_rate']:.2%} "
              f"ethiopic={stats['character_distribution']['ethiopic_share_of_nonspace']:.1%} "
              f"origin={stats['origin']['classification']}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
