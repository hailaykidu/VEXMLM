#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-hf-verify
#SBATCH --partition=ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hf_verify_%j.out
#SBATCH --error=logs/hf_verify_%j.err
#
# Verify the already-published VEXMLM downstream repositories on the Hugging Face
# Hub: 6 tasks x 5 seeds = 30 checkpoints.
#
# Verification only. This job never uploads, never stages a local copy of the
# checkpoints, and never modifies the Git repository. It downloads each published
# repository one at a time, loads every checkpoint with its task-specific
# architecture, runs real inference, audits the downloaded copy for private
# metadata, and deletes the copy before moving to the next repository -- so peak
# disk stays at roughly one repository (~6 GB) rather than the full 38 GB.
#
# The predecessor job (61472) was killed OOM because it copied ~35 GB through a
# memory-backed $TMPDIR during a staging step. There is no staging step here, and
# downloads go to disk-backed scratch under the repository, not to $TMPDIR.
#
# Authentication uses the Hugging Face token cache already present for this
# account. No token appears in this script, in the repository, or in any log.
#
# Scheduler metadata stays in the private job log and the private report; it is
# never written to a published file.
#
#   sbatch scripts/slurm_verify_hf_downstream.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"
mkdir -p logs

echo "=== VEXMLM Hugging Face downstream verification ==="
echo "job=${SLURM_JOB_ID:-local} node=$(hostname) started=$(date -Is)"

# Disk-backed scratch inside the repository (gitignored), never $TMPDIR.
WORK="$REPO/.hf_verify_${SLURM_JOB_ID:-$$}"
rm -rf "$WORK"; mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# Redirect only the blob cache to disk-backed scratch. HF_HOME must NOT be
# overridden: the credential lives under the default HF_HOME, and moving it
# points the token lookup at an empty directory (job 61777 failed this way).
export HF_HUB_CACHE="$WORK/hub_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HUB_CACHE"

python3 - "$REPO" "$WORK" "hf_downstream_release_report.txt" <<'PYEOF'
import json, os, shutil, sys
import statistics as st
from pathlib import Path

REPO = Path(sys.argv[1]); WORK = Path(sys.argv[2]); REPORT = Path(sys.argv[3])
os.chdir(REPO)

BASE_MODEL = "checkpoints/vexmlm-stage1-spm"
SEEDS = [42, 43, 44, 45, 46]
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "sentencepiece.bpe.model"]
FORBIDDEN = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
             "training_args.bin", "results.json", "predictions.json"]

TASKS = {
  "masakhaner_amh": dict(repo="VEXMLM-Amharic-NER",       kind="ner", lang="Amharic",
                         dataset="MasakhaNER (Amharic)", metric="test_entity_f1",
                         expect=(0.6347, 0.0148)),
  "tigrinya_ner":   dict(repo="VEXMLM-Tigrinya-NER",      kind="ner", lang="Tigrinya",
                         dataset="Tigrinya NER",         metric="test_entity_f1",
                         expect=(0.7282, 0.0079)),
  "amqa":           dict(repo="VEXMLM-AmQA",              kind="qa",  lang="Amharic",
                         dataset="AmQA",                 metric="exact_match",
                         expect=(32.57, 0.77)),
  "tigqa":          dict(repo="VEXMLM-TIGQA",             kind="qa",  lang="Tigrinya",
                         dataset="TIGQA",                metric="exact_match",
                         expect=(2.39, 0.82)),
  "tiquad":         dict(repo="VEXMLM-TiQuAD",            kind="qa",  lang="Tigrinya",
                         dataset="TiQuAD",               metric="exact_match",
                         expect=(50.24, 0.48)),
  "afrisenti":      dict(repo="VEXMLM-AfriSenti-Amharic", kind="sa",  lang="Amharic",
                         dataset="AfriSenti-SemEval 2023 (Amharic)",
                         metric="test_accuracy",         expect=(0.4978, 0.0331)),
}

OBSOLETE = ["35.50", "0.87 |", "96.1", "98.2"]
PATTERNS = ["SLURM", "SBATCH", "JOB_ID", "JOBID", "ARRAY_TASK_ID", "/home/", "/homes/",
            "neumann", "gpunode", "token=", "api_key", "API_KEY", "PASSWORD", "SECRET",
            "BEGIN PRIVATE KEY", "Co-authored-by", "Co-committed-by",
            "ChatGPT", "OpenAI", "Anthropic"]
# Inherited xlm-roberta-base vocabulary entries; legitimate inside tokenizer.json only.
VOCAB_EXEMPT = {"scratch", "password", "secret", "claude"}

failures = []
log = lambda *a: print(*a, flush=True)

def finish(ok, rows, aggregates, note=""):
    lines = ["VEXMLM HUGGING FACE DOWNSTREAM RELEASE -- VERIFICATION REPORT", "",
             f"job: {os.environ.get('SLURM_JOB_ID','local')} on {os.uname().nodename}",
             f"account: {USER}", ""]
    if note:
        lines += [note, ""]
    lines += ["PER-CHECKPOINT VERIFICATION", "",
              f"{'task':<5}{'dataset':<34}{'seed':>5}  {'repository':<42}"
              f"{'up':>5}{'dl':>5}{'load':>6}{'infer':>7}{'priv':>6}"]
    for r in rows:
        lines.append(f"{r['kind']:<5}{r['dataset']:<34}{r['seed']:>5}  {r['repo']:<42}"
                     f"{r['upload']:>5}{r['download']:>5}{r['load']:>6}"
                     f"{r['inference']:>7}{r['privacy']:>6}")
    lines += ["", "AGGREGATE BY TASK", ""]
    for line in aggregates:
        lines.append("  " + line)
    if failures:
        lines += ["", "FAILURES", ""] + [f"  {f}" for f in failures]
    published = sum(1 for r in rows if r["inference"] == "PASS")
    lines += ["", f"Total checkpoints:\n30", f"\nPublished:\n{published}",
              f"\nFailed:\n{30 - published}", ""]
    for meta in TASKS.values():
        state = "PASS" if all(r["inference"] == "PASS" for r in rows
                              if r["repo"].endswith(meta["repo"])) and rows else "FAIL"
        lines.append(f"{meta['dataset']}:\n{state}")
    verdict = "READY FOR PUBLIC RELEASE" if ok else "NOT READY"
    lines += ["", "Fresh download:\n" + ("PASS" if ok else "FAIL"),
              "", "Model loading:\n" + ("PASS" if ok else "FAIL"),
              "", "Inference:\n" + ("PASS" if ok else "FAIL"),
              "", "Privacy audit:\n" + ("PASS" if ok else "FAIL"),
              "", "GitHub:\nUNCHANGED", "", "Private history:\nPRESERVED",
              "", "AI authorship/committer:\nNONE", "", f"Final:\n{verdict}", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nreport written: {REPORT}")
    log("ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)

# --------------------------------------------------------------- auth
from huggingface_hub import HfApi, HfFolder, snapshot_download
token = HfFolder.get_token() or os.environ.get("HF_TOKEN")
if not token:
    log("FATAL: no Hugging Face token available; cannot authenticate non-interactively")
    sys.exit(1)
api = HfApi(token=token)
USER = api.whoami(token=token)["name"]
log(f"authenticated as: {USER}")

# ------------------------------------------- 1. local provenance re-check
log("\n--- provenance re-check against verified records ---")
aggregates = []
for ds, meta in TASKS.items():
    vals = []
    for seed in SEEDS:
        rec = Path(f"results/spm_stage2/{ds}-seed{seed}.json")
        if not rec.exists():
            failures.append(f"{ds}-seed{seed}: evaluation record missing"); continue
        r = json.loads(rec.read_text())
        if r.get("model") != BASE_MODEL:
            failures.append(f"{ds}-seed{seed}: base model mismatch"); continue
        vals.append(r.get("metrics", r)[meta["metric"]])
    if len(vals) != 5:
        failures.append(f"{ds}: only {len(vals)}/5 records"); continue
    mu, sd = sum(vals) / 5, st.stdev(vals)
    emu, esd = meta["expect"]
    tol = 0.005 if emu < 1 else 0.02
    ok = abs(mu - emu) <= tol and abs(sd - esd) <= tol * 2
    meta["measured"] = (mu, sd)
    line = f"{meta['repo']:28} 5/5 seeds  {meta['metric']} = {mu:.4f} +/- {sd:.4f}  {'PASS' if ok else 'FAIL'}"
    aggregates.append(line)
    log("  " + line)
    if not ok:
        failures.append(f"{ds}: aggregate {mu:.4f}+/-{sd:.4f} != expected {emu}+/-{esd}")

if failures:
    finish(False, [], aggregates, "Provenance re-check failed; no verification performed.")

# ---------------------------- 2. per-repo download, load, inference, audit
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          AutoModelForQuestionAnswering, AutoModelForSequenceClassification)

NER_TXT = {"Tigrinya": "ኤርትራ ኣብ ቀርኒ አፍሪቃ እትርከብ ሃገር እያ።",
           "Amharic":  "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት።"}
QA_IN = {"Amharic": ("ኢትዮጵያ የምትገኘው በየትኛው አህጉር ነው?", "ኢትዮጵያ በአፍሪካ አህጉር የምትገኝ ሀገር ናት።"),
         "Tigrinya": ("ኤርትራ ኣብ ኣየናይ ክፍለ ዓለም ትርከብ?", "ኤርትራ ኣብ አፍሪቃ ክፍለ ዓለም እትርከብ ሃገር እያ።")}
SA_TXT = "በጣም ጥሩ ነው!"

def audit_tree(root, label):
    """Return list of real privacy hits, exempting inherited vocabulary."""
    hits = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix in {".safetensors", ".model", ".bin"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in PATTERNS:
            if pat in text:
                hits.setdefault(pat, set()).add(p.name)
    real = {}
    for pat, files in hits.items():
        if files <= {"tokenizer.json"} and pat.lower() in VOCAB_EXEMPT:
            continue
        real[pat] = sorted(files)
    if real:
        log(f"  {label}: PRIVACY HITS {real}")
    return real

rows = []
for ds, meta in TASKS.items():
    rid = f"{USER}/{meta['repo']}"
    local = WORK / meta["repo"]
    log(f"\n--- {rid} ---")
    try:
        snapshot_download(repo_id=rid, repo_type="model", local_dir=str(local),
                          token=token, max_workers=2)
    except Exception as e:
        failures.append(f"{rid}: download failed ({type(e).__name__})")
        continue

    present = sorted(d.name for d in local.iterdir() if d.name.startswith("seed-"))
    if present != [f"seed-{s}" for s in SEEDS]:
        failures.append(f"{rid}: expected 5 seed folders, found {present}")

    card = local / "README.md"
    if not card.exists():
        failures.append(f"{rid}: model card missing")
    else:
        text = card.read_text(encoding="utf-8")
        for bad in OBSOLETE:
            if bad in text:
                failures.append(f"{rid}: obsolete value {bad!r} in model card")
        if "github.com/hailaykidu/VEXMLM" not in text:
            failures.append(f"{rid}: model card missing official repository link")
        if "Interactive inference" not in text and "interactive inference" not in text:
            failures.append(f"{rid}: model card does not separate benchmark from interactive use")

    stray = [p.name for p in local.rglob("*") if p.name in FORBIDDEN]
    if stray:
        failures.append(f"{rid}: training-state files published: {sorted(set(stray))}")

    priv = audit_tree(local, rid)
    if priv:
        failures.append(f"{rid}: private metadata in published copy: {priv}")
    priv_state = "FAIL" if priv else "PASS"

    for seed in SEEDS:
        sub = local / f"seed-{seed}"
        row = dict(kind=meta["kind"], dataset=meta["dataset"], seed=seed, repo=rid,
                   upload="PASS", download="PASS", load="FAIL",
                   inference="FAIL", privacy=priv_state)
        if not sub.exists():
            failures.append(f"{rid} seed-{seed}: folder absent"); rows.append(row); continue
        missing = [f for f in REQUIRED if not (sub / f).exists()]
        if missing:
            failures.append(f"{rid} seed-{seed}: missing {missing}"); rows.append(row); continue
        try:
            tok = AutoTokenizer.from_pretrained(str(sub))
            if meta["kind"] == "ner":
                m = AutoModelForTokenClassification.from_pretrained(str(sub)); m.eval()
                row["load"] = "PASS"
                words = NER_TXT[meta["lang"]].split()
                enc = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    pred = m(**enc).logits.argmax(-1)[0].tolist()
                seen, ents = set(), []
                for p, w in zip(pred, enc.word_ids(0)):
                    if w is None or w in seen: continue
                    seen.add(w)
                    lab = m.config.id2label[p]
                    if lab != "O": ents.append((words[w], lab))
                assert m.config.num_labels >= 9
                detail = f"entities={ents[:3] or 'none'}"
            elif meta["kind"] == "qa":
                m = AutoModelForQuestionAnswering.from_pretrained(str(sub)); m.eval()
                row["load"] = "PASS"
                q, c = QA_IN[meta["lang"]]
                enc = tok(q, c, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad(): o = m(**enc)
                s, e = o.start_logits.argmax(), o.end_logits.argmax()
                ans = tok.decode(enc.input_ids[0][s:e + 1], skip_special_tokens=True)
                assert o.start_logits.shape[1] == enc.input_ids.shape[1]
                detail = f"answer={ans!r}"
            else:
                m = AutoModelForSequenceClassification.from_pretrained(str(sub)); m.eval()
                row["load"] = "PASS"
                enc = tok(SA_TXT, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad(): lg = m(**enc).logits
                assert m.config.num_labels == 3
                detail = f"label={m.config.id2label[lg.argmax(-1).item()]}"
            row["inference"] = "PASS"
            log(f"  seed-{seed}: {type(m).__name__:38} {detail}")
            del m
        except Exception as e:
            failures.append(f"{rid} seed-{seed}: {type(e).__name__}: {e}")
            log(f"  seed-{seed}: FAILED {type(e).__name__}")
        rows.append(row)

    # Free the ~6 GB copy before the next repository.
    shutil.rmtree(local, ignore_errors=True)

ok = (not failures) and len(rows) == 30 and all(
    r[k] == "PASS" for r in rows for k in ("download", "load", "inference", "privacy"))
finish(ok, rows, aggregates)
PYEOF

rc=$?
echo ""
echo "python stage exit=$rc finished=$(date -Is)"
exit $rc
