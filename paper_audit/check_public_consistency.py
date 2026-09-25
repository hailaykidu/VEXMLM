#!/usr/bin/env python3
"""Check that every result in the paper appears verbatim in the public artifacts.

Public artifacts: the GitHub repository (branch origin/main, read with `git show`) and the
Hugging Face model card of Hailay/VEXMLM (read over HTTPS). Read-only; nothing is changed.

    git fetch origin && python3 paper_audit/check_public_consistency.py

Exit status 1 if a paper value is missing from, or contradicts, a public artifact that
reports it ("n/a" = that artifact does not report the value). Values that
are in the paper but not yet published anywhere are listed separately (they come from files that
are committed with the corrected release).
"""
import csv
import io
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "generated"
BRANCH = "origin/main"
HF_CARD = "https://huggingface.co/Hailay/VEXMLM/raw/main/README.md"


def git_show(path):
    r = subprocess.run(["git", "show", f"{BRANCH}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def norm(text):
    text = text.replace("{,}", ",").replace("\u2212", "-").replace("$\\pm$", "±")
    return text.replace("\\textbf{", "").replace("}", "")


def macros():
    out = {}
    for line in (GEN / "results_macros.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}  % ", line)
        if m:
            out[m.group(1)] = norm(m.group(2))
    return out


def main():
    global BRANCH
    if "--branch" in sys.argv:
        BRANCH = sys.argv[sys.argv.index("--branch") + 1]
    mac = macros()
    gh_down = git_show("results/downstream_task_metrics.csv")
    gh_abl = git_show("results/table5_ablation.csv")
    gh_tok = git_show("results/spmerge_tokenizer_metrics.json")
    gh_summary = git_show("reports/RESULTS_SUMMARY.md") or ""
    try:
        hf = norm(urllib.request.urlopen(HF_CARD, timeout=30).read().decode())
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot read the Hugging Face card: {e}")

    down = {(r["Dataset"], r["Metric"]): r["VEXMLM SP-Merge (mean ± std)"]
            for r in csv.DictReader(io.StringIO(gh_down))}
    abl = {r["Arm"]: r for r in csv.DictReader(io.StringIO(gh_abl))}
    tok = json.loads(gh_tok)["tokenizers"]

    checks, unpublished = [], []

    def check(what, value, github_ok, hf_ok):
        checks.append((what, value, github_ok, hf_ok))

    # Table 4 and Table 6: VEXMLM values
    spec = [("dsSAAmhVex", "afrisenti", "Accuracy"), ("dsNERAmhVex", "masakhaner_amh", "Accuracy"),
            ("dsNERTirVex", "tigrinya_ner", "Accuracy"), ("dsQAEMAmhVex", "amqa", "EM"),
            ("dsQAEMTirVex", "tigqa", "EM"), ("dsQAFoneAmhVex", "amqa", "F1"), ("dsQAFoneTirVex", "tigqa", "F1"),
            ("appSAMacroFAmhVex", "afrisenti", "Macro-F1"), ("appNERMacroFAmhVex", "masakhaner_amh", "Macro-F1"),
            ("appNERMacroFTirVex", "tigrinya_ner", "Macro-F1"), ("appNEREntityFAmhVex", "masakhaner_amh", "Entity-F1"),
            ("appNEREntityFTirVex", "tigrinya_ner", "Entity-F1")]
    for name, ds, metric in spec:
        v = mac[name]
        check(f"Table 4/6 VEXMLM {ds} {metric}", v, down.get((ds, metric)) == v, v in hf)

    # Table 5 and Table 3 (fragmented-word column)
    for name, arm in [("ablBase", "xlmr_baseline"), ("ablRand", "expansion_random_init"),
                      ("ablMean", "expansion_mean_init"), ("ablFull", "full_vexmlm")]:
        v = mac[name]
        check(f"Table 5 {arm}", v, abl[arm]["OOV Accuracy (%) mean ± std"] == v, v in hf)
    for name, arm in [("ablRandDelta", "expansion_random_init"), ("ablMeanDelta", "expansion_mean_init"),
                      ("ablFullDelta", "full_vexmlm")]:
        v = mac[name]
        check(f"Table 5 delta {arm}", v, abl[arm]["Delta"] == v, v in hf)
    check("Sec. 5.3 full model vs baseline", mac["ablFullVsBase"],
          mac["ablFullVsBase"].lstrip("+") in gh_summary, mac["ablFullVsBase"].lstrip("+") in hf)

    # Table 2 / Figure 1: XLM-R and VEXMLM tokenizer metrics
    key = {"Xlmr": "xlm-roberta-base", "Vex": "checkpoints/vexmlm-stage1-spm"}
    for short, tname in key.items():
        for lang, code in (("amh", "Amh"), ("tir", "Tir")):
            for metric, field in (("Fertility", "fertility"), ("Compression", "compression")):
                v = mac[f"tok{short}{code}{metric}"]
                gh = f"{tok[tname]['per_language'][lang][field]:.4f}" == v
                check(f"Table 2 {tname} {lang} {field}", v, gh, v in hf)
            v = mac[f"tok{short}{code}Roundtrip"]
            gh = f"{tok[tname]['oov'][lang]['roundtrip_rate']:.4f}" == v
            check(f"Table 2 {tname} {lang} round-trip", v, gh, v in hf)
    check("Fertility reduction Tigrinya", mac["tokFertRedTir"] + "%",
          f"{mac['tokFertRedTir']}%" in gh_summary, f"{mac['tokFertRedTir']}%" in hf)
    check("Fertility reduction Amharic", mac["tokFertRedAmh"] + "%",
          f"{mac['tokFertRedAmh']}%" in gh_summary, None)

    # Model facts on the card
    check("Parameters (VEXMLM)", mac["paramVex"], None, mac["paramVex"] in hf)
    check("Vocabulary", mac["vocabFinal"], mac["vocabFinal"] in gh_summary or mac["vocabFinal"].replace(",", "") in gh_summary, mac["vocabFinal"] in hf)
    check("Best validation loss", mac["ptBestLoss"], mac["ptBestLoss"] in gh_summary, mac["ptBestLoss"] in hf)

    for name in [n for n in mac if n.startswith("tokGlot")]:
        unpublished.append((f"Table 2 Glot500 {name}", mac[name], "results/provenance/tokenizer_metrics_2026-08-15.json (earlier local run)"))
    for name in [n for n in mac if n.endswith("Xlmr") and n.startswith(("ds", "app"))]:
        unpublished.append((f"Table 4/6 XLM-R {name}", mac[name], "results/baselines/xlmr_seed42/"))

    bad = [c for c in checks if c[2] is False or c[3] is False]
    fmt = lambda x: "n/a" if x is None else ("yes" if x else "NO")  # noqa: E731
    print(f"{len(checks)} paper values checked against GitHub {BRANCH} and the Hugging Face card")
    for what, v, g, h in checks:
        flag = "BAD" if (g is False or h is False) else "OK "
        print(f"  {flag} {what:48s} {v:>18s}  GitHub={fmt(g)}  HF={fmt(h)}")
    print(f"\n{len(unpublished)} paper values not yet in any public artifact (published with the corrected release):")
    for what, v, src in unpublished:
        print(f"  --  {what:48s} {v:>18s}  {src}")
    if bad:
        sys.exit(f"\n{len(bad)} inconsistencies with public artifacts")
    print("\nNo inconsistency between the paper and the public artifacts.")


if __name__ == "__main__":
    main()
