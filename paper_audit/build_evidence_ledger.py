#!/usr/bin/env python3
"""Build paper_audit/evidence_ledger.csv for the corrected paper and check that it is complete.

Rows come from two places:
  * every macro in paper/generated/results_macros.tex (value, source file and key are taken
    from the macro's generated comment), and
  * every number typed literally in paper/*.tex, listed in LITERALS below with its source.
The script fails if the paper uses a macro that is not generated, or contains a literal
number that is not listed.

    python3 paper_audit/build_evidence_ledger.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MACROS = PAPER / "generated" / "results_macros.tex"
OUT = ROOT / "paper_audit" / "evidence_ledger.csv"

# Macros whose value replaces a published value; the ID points into correction_notice.md.
CORRECTS = {
    "tokXlmrTirFertility": ("2.1", "F1-1"), "tokVexTirFertility": ("1.3", "F1-2"),
    "tokGlotTirFertility": ("bar without stated value", "F1-3"),
    "tokXlmrAmhFertility": ("1.2", "F1-4"), "tokVexAmhFertility": ("1.1", "F1-5"),
    "tokGlotAmhFertility": ("bar without stated value", "F1-6"),
    "tokFertRedTir": ("38%", "F1-8"), "tokFertRedAmh": ("8%", "F1-8"),
    "dsSAAmhXlmr": ("0.77", "T4-1"), "dsSAAmhVex": ("0.79", "T4-2"),
    "dsNERAmhXlmr": ("0.75", "T4-4"), "dsNERTirXlmr": ("0.75", "T4-4"),
    "dsNERAmhVex": ("0.78", "T4-5"), "dsNERTirVex": ("0.78", "T4-5"),
    "dsQAEMAmhXlmr": ("0.66", "T4-7"), "dsQAEMTirXlmr": ("0.66", "T4-7"),
    "dsQAEMAmhVex": ("0.77", "T4-8"), "dsQAEMTirVex": ("0.77", "T4-8"),
    "dsQAFoneAmhXlmr": ("0.78", "T4-10"), "dsQAFoneTirXlmr": ("0.78", "T4-10"),
    "dsQAFoneAmhVex": ("0.79", "T4-11"), "dsQAFoneTirVex": ("0.79", "T4-11"),
    "tirOShare": ("", ""), "ablBase": ("96.1", "T5-1"), "ablRand": ("97.3", "T5-2"), "ablMean": ("97.8", "T5-3"),
    "ablFull": ("98.2", "T5-4"), "ablRandDelta": ("+1.2", "T5-5"), "ablMeanDelta": ("+2.3", "T5-6"),
    "ablFullDelta": ("+7.1", "T5-7"),
    "ablBaseNonOOV": ("96.1 (Table 3)", "T3-2"), "ablFullNonOOV": ("98.2 (Table 3)", "T3-3"),
    "oovGainTir": ("+11.4", "T3-5"), "nonOOVGainTir": ("+2.1", "T3-5"),
    "ptPrecision": ("fp16", "M-1"), "ftPrecision": ("fp16", "M-1"),
    "ptEpochsConfigured": ("56", "M-3"), "ftWarmup": ("unspecified warmup", "M-5"),
    "paramXlmr": ("279M", "M-7"), "paramVex": ("301M", "M-7"),
    "vocabBase": ("250K", "M-8"), "vocabFinal": ("280K", "M-8"),
}

# Literal numbers in the paper: (file, number, context, source, status).
LITERALS = [
    ("01_intro.tex", "500", "pretraining over 500+ languages", "ImaniGooghari et al. (2023)", "EXTERNAL"),
    ("02_related.tex", "4", "4th century CE inscriptions", "cited source flagged CHECK-CITATION", "EXTERNAL"),
    ("02_related.tex", "511", "Glot500-m, 511 languages", "ImaniGooghari et al. (2023)", "EXTERNAL"),
    ("03_proposed.tex", "100", "XLM-R pretraining languages", "Conneau et al. (2020)", "EXTERNAL"),
    ("05_result_eval.tex", "100", "languages supported by the base model", "Conneau et al. (2020)", "EXTERNAL"),
    ("04_experiment.tex", "512", "XLM-R maximum input length (514 position embeddings incl. 2 offset positions)", "checkpoints/vexmlm-stage1-spm/config.json max_position_embeddings", "VERIFIED"),
    ("04_experiment.tex", "128", "QA sliding-window stride", "configs/base.yaml qa.doc_stride", "VERIFIED"),
    ("04_experiment.tex", "256", "sequence limit", "configs/base.yaml pretraining/finetuning.max_seq_length", "VERIFIED"),
    ("05_result_eval.tex", "256", "sequence limit", "configs/base.yaml max_seq_length", "VERIFIED"),
    ("04_experiment.tex", "12", "layers / attention heads", "checkpoints/vexmlm-stage1-spm/config.json num_hidden_layers / num_attention_heads", "VERIFIED"),
    ("04_experiment.tex", "768", "hidden size", "checkpoints/vexmlm-stage1-spm/config.json hidden_size", "VERIFIED"),
    ("04_experiment.tex", "1", "layer-norm epsilon 1e-5", "checkpoints/vexmlm-stage1-spm/config.json layer_norm_eps", "VERIFIED"),
    ("04_experiment.tex", "42", "seeds 42-46", "results/spm_stage2/*-seed42..46.json", "VERIFIED"),
    ("04_experiment.tex", "46", "seeds 42-46", "results/spm_stage2/*-seed42..46.json", "VERIFIED"),
    ("04_experiment.tex", "10", "min line length 10 chars; 80:10:10 split", "configs/base.yaml tokenizer.min_sentence_length; datasets/registry.py TIGQA notes", "VERIFIED"),
    ("04_experiment.tex", "50", "min 50% Ge'ez characters", "configs/base.yaml tokenizer.min_ethiopic_ratio", "VERIFIED"),
    ("04_experiment.tex", "2", "2% development split; Stage 2; models 2-4", "datasets/prepare.py --dev-fraction default 0.02", "VERIFIED"),
    ("04_experiment.tex", "1", "1% Stage 1 validation split", "pretraining/run_mlm.py --val-fraction default 0.01; stage1_spm_data.log.txt", "VERIFIED"),
    ("04_experiment.tex", "80", "TIGQA 80:10:10 split", "datasets/registry.py TIGQA notes", "VERIFIED"),
    ("04_experiment.tex", "14", "AfriSenti covers 14 languages", "Muhammad et al. (2023); results/multilingual_evaluation/aggregated/language_coverage.csv", "EXTERNAL"),
    ("04_experiment.tex", "395", "Glot500 parameters", "ImaniGooghari et al. (2023)", "EXTERNAL"),
    ("04_experiment.tex", "400", "Glot500 vocabulary (tokenizer: 401,145)", "ImaniGooghari et al. (2023)", "EXTERNAL"),
    ("XX_appendix.tex", "42", "XLM-R baseline seed", "results/baselines/xlmr_seed42/*/results.json seed", "VERIFIED"),
    ("05_result_eval.tex", "42", "XLM-R baseline seed", "results/baselines/xlmr_seed42/*/results.json seed", "VERIFIED"),
]
STRUCTURAL = {"1", "2", "3", "4"}  # enumeration labels "(1)".."(4)", "Stage 1/2", "Models 2--4"


def parse_macros():
    out = {}
    for line in MACROS.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}  % (.*)$", line)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def paper_sources():
    files = [PAPER / "acl_lualatex.tex"] + sorted((PAPER / "Sections").glob("*.tex"))
    return {f: f.read_text() for f in files}


def check(macros, sources):
    used = set()
    for text in sources.values():
        used |= set(re.findall(r"\\([a-zA-Z]+)\b", text))
    generated_tables = "".join(p.read_text() for p in (PAPER / "generated").glob("table_*.tex"))
    candidates = {u for u in used if re.match(r"(tok|ds|app|abl|pt|ft|param|vocab|spm|cand|new|corpus|split|oov|nonOOV)[A-Z]", u)}
    missing = sorted(c for c in candidates if c not in macros)
    listed = {(f, n) for f, n, *_ in LITERALS}
    stray = []
    for f, text in sources.items():
        body = text
        if f.name == "acl_lualatex.tex":
            body = body[body.index("\\begin{abstract}"):body.index("\\end{abstract}")]
        body = re.sub(r"%.*", "", body)
        body = re.sub(r"\\(label|ref|cite|citet|inputtable|input|includegraphics|url|href|setlength|begin|end|textsuperscript)\*?(\[[^\]]*\])?\{[^}]*\}", "", body)
        body = re.sub(r"\\TODO\{[^}]*\}", "", body)
        body = re.sub(r"\{[lcrp|]+\}", "", body)
        for m in re.finditer(r"(?<![\\\w{.])(\d[\d,.]*\d|\d)(?![\w}])", body):
            n = m.group(1)
            if n in STRUCTURAL and (f.name, n) not in listed:
                continue
            if (f.name, n) not in listed:
                stray.append((f.name, n, body[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")))
    return missing, stray, generated_tables


def main():
    macros = parse_macros()
    sources = paper_sources()
    missing, stray, _ = check(macros, sources)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "paper_location", "claim_text_or_value", "source_file",
                    "line_or_key_or_cell", "commit", "producing_script_and_command", "status",
                    "draft_value", "correction_id"])
        for i, (name, (value, comment)) in enumerate(macros.items(), 1):
            src, _, key = comment.partition(" ")
            draft, cid = CORRECTS.get(name, ("", ""))
            w.writerow([f"M{i:03d}", f"macro \\{name}", value, src, key or comment, commit,
                        "python3 scripts/make_paper_artifacts.py", "CORRECTED" if cid else "VERIFIED",
                        draft, cid])
        for j, (fname, n, ctx, src, status) in enumerate(LITERALS, 1):
            w.writerow([f"L{j:03d}", f"paper/Sections/{fname}" if fname != "acl_lualatex.tex" else fname,
                        f"{n} ({ctx})", src, "", commit, "", status, "", ""])
    print(f"wrote {OUT} ({len(macros)} macro rows, {len(LITERALS)} literal rows)")
    if missing or stray:
        for m in missing:
            print("MACRO NOT GENERATED:", m)
        for s in stray:
            print("UNLISTED LITERAL:", s)
        sys.exit(1)
    print("0 untraced numbers")


if __name__ == "__main__":
    main()
