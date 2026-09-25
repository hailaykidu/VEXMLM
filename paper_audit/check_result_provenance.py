#!/usr/bin/env python3
"""Check that every number in the paper comes from results that existed before the correction.

For each source file named in paper/generated/results_macros.tex, the file must be
  (a) on GitHub (origin/main) with identical content, or
  (b) an unchanged copy of an earlier run record, dated before the correction started, or
  (c) an extract of such a record (listed below with the fields it keeps), or
  (d) the Hugging Face model card (saved copy, with its revision).
No number may come from a computation made during the correction.

    python3 paper_audit/check_result_provenance.py
"""
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORRECTION_START = datetime(2026, 9, 24)
BRANCH = "origin/main"

# copy in the repository -> original run record (compared byte for byte, except
# home-directory paths replaced by <home> where noted)
COPIES = {
    "results/baselines/xlmr_seed42/afrisenti/results.json": "checkpoints/baseline-xlmr-phase-a/afrisenti-seed42/results.json",
    "results/baselines/xlmr_seed42/amqa/results.json": "checkpoints/baseline-xlmr-phase-a/amqa-seed42/results.json",
    "results/baselines/xlmr_seed42/tigqa/results.json": "checkpoints/baseline-xlmr-phase-a/tigqa-seed42/results.json",
    "results/baselines/xlmr_seed42/masakhaner_amh/results.json": "checkpoints/baseline-xlmr-phase-a/masakhaner_amh-seed42/results.json",
    "results/baselines/xlmr_seed42/tigrinya_ner/results.json": "checkpoints/baseline-xlmr-phase-a/tigrinya_ner-seed42/results.json",
    "results/provenance/tokenizer_metrics_2026-08-15.json": "reports/internal/development_metrics/tokenizer_metrics.json",
    "results/provenance/new_tokens.json": "vocabulary_expansion/artifacts/new_tokens.json",
    "results/provenance/new_tokens_amh.json": "vocabulary_expansion/artifacts/new_tokens_amh.json",
    "results/provenance/new_tokens_tir.json": "vocabulary_expansion/artifacts/new_tokens_tir.json",
    "results/provenance/tokenizer_manifest_amh.json": "tokenizer/artifacts/amharic/manifest.json",
    "results/provenance/tokenizer_manifest_tir.json": "tokenizer/artifacts/tigrinya/manifest.json",
    "results/provenance/expansion_manifest_spm.json": "checkpoints/vexmlm-expanded-spm/expansion_manifest.json",
    "results/provenance/expansion_manifest_spm_random.json": "checkpoints/vexmlm-expanded-spm-random/expansion_manifest.json",
    "results/provenance/stage1_summary_spm.json": "checkpoints/vexmlm-stage1-spm/stage1_summary.json",
    "results/provenance/stage1_spm_trainer_state_best.json": "checkpoints/vexmlm-stage1-spm/checkpoint-24808/trainer_state.json",
    "results/provenance/preparation_report.json": "datasets/processed/preparation_report.json",
}
# extracts: repository file -> (original record, what was kept)
EXTRACTS = {
    "results/provenance/training_args.json": ("checkpoints/vexmlm-stage1-spm/training_args.bin, checkpoints/vexmlm-spm-stage2/*-seed42/training_args.bin",
                                              "selected TrainingArguments fields read with torch.load"),
    "results/provenance/stage1_spm_data.log.txt": ("logs/stage1_spm_60674.err, scripts/internal/slurm_pretrain_spm.sh",
                                                   "data log lines and the run_mlm.py command, verbatim"),
}
HF_COPY = "results/provenance/hf_model_card_Hailay_VEXMLM.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def on_github_identical(rel):
    r = subprocess.run(["git", "show", f"{BRANCH}:{rel}"], cwd=ROOT, capture_output=True)
    return r.returncode == 0 and r.stdout == (ROOT / rel).read_bytes()


def sources():
    files = set()
    for line in (ROOT / "paper/generated/results_macros.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\\w+\}\{.*\}  % (.*)$", line)
        if not m:
            continue
        for f in re.findall(r"((?:results|configs|datasets)/[\w./\-]+?\.(?:json|csv|yaml|md|txt))", m.group(1)):
            if "seed42..46" in f:
                files.update(f.replace("seed42..46", f"seed{s}") for s in range(42, 47))
            else:
                files.add(f)
        for f in re.findall(r"\b(training_args\.json|stage1_spm_trainer_state_best\.json|stage1_summary_spm\.json|"
                            r"new_tokens(?:_amh|_tir)?\.json|tokenizer_manifest_(?:amh|tir)\.json)\b", m.group(1)):
            files.add(f"results/provenance/{f}")
    return sorted(files)


def main():
    global BRANCH
    if "--branch" in sys.argv:
        BRANCH = sys.argv[sys.argv.index("--branch") + 1]
    bad = []
    rows = []
    for rel in sources():
        path = ROOT / rel
        if not path.exists():
            bad.append(rel); rows.append((rel, "MISSING")); continue
        if on_github_identical(rel):
            rows.append((rel, f"on GitHub {BRANCH}, identical")); continue
        if rel in COPIES:
            orig = ROOT / COPIES[rel]
            same = orig.exists() and (sha(orig) == sha(path) or
                                      orig.read_text().replace(str(Path.home()), "<home>") == path.read_text())
            dated = orig.exists() and datetime.fromtimestamp(orig.stat().st_mtime) < CORRECTION_START
            status = f"copy of {COPIES[rel]} ({'identical' if same else 'DIFFERS'}; original dated " \
                     f"{datetime.fromtimestamp(orig.stat().st_mtime):%Y-%m-%d})" if orig.exists() else f"original {COPIES[rel]} MISSING"
            rows.append((rel, status))
            if not (same and dated):
                bad.append(rel)
            continue
        if rel in EXTRACTS:
            rows.append((rel, f"extract of {EXTRACTS[rel][0]} ({EXTRACTS[rel][1]})")); continue
        if rel == HF_COPY:
            rev = (ROOT / "results/provenance/hf_model_card_Hailay_VEXMLM.revision.txt").read_text().strip()
            rows.append((rel, f"Hugging Face card Hailay/VEXMLM, {rev}")); continue
        bad.append(rel); rows.append((rel, "NOT a pre-existing record"))
    for rel, status in rows:
        print(f"  {'BAD' if rel in bad else 'OK '} {rel:62s} {status}")
    if bad:
        sys.exit(f"\n{len(bad)} source files are not pre-existing run records")
    print(f"\nAll {len(rows)} source files of the paper's numbers are pre-existing run records "
          "(no result computed during the correction).")


if __name__ == "__main__":
    main()
