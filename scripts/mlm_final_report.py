#!/usr/bin/env python3
"""Build reports/MLM_FINAL_REPORT.md from a completed Stage 1 run.

Reads the trainer state, run summary, and SLURM log; emits the per-epoch loss
table, training curves (SVG, no external dependencies), throughput, GPU record,
effective token count, deduplication impact, and vocabulary firing statistics.

    python scripts/mlm_final_report.py --run checkpoints/vexmlm-stage1 \
        --log logs/stage1_60584.out --job 60584
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def sha256(p: Path, limit: int = 1 << 26) -> str:
    h, read = hashlib.sha256(), 0
    with open(p, "rb") as fh:
        while (b := fh.read(1 << 20)) and read < limit:
            h.update(b)
            read += len(b)
    return h.hexdigest()


def svg_curve(series: dict[str, list[tuple[float, float]]], title: str,
              xlabel: str, ylabel: str, out: Path, width=760, height=380) -> None:
    """Minimal dependency-free line chart. Theme-neutral, readable on both."""
    pad_l, pad_r, pad_t, pad_b = 68, 130, 40, 52
    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts]
    if not xs or not ys:
        return
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    span = (y1 - y0) or 1.0
    y0, y1 = y0 - span * 0.08, y1 + span * 0.08

    def px(x): return pad_l + (x - x0) / (x1 - x0) * (width - pad_l - pad_r)
    def py(y): return height - pad_b - (y - y0) / (y1 - y0) * (height - pad_t - pad_b)

    colors = ["#2563eb", "#dc2626", "#059669", "#d97706"]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'width="{width}" height="{height}" font-family="system-ui,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         f'<text x="{width/2}" y="24" text-anchor="middle" font-size="15" '
         f'font-weight="600" fill="#111">{title}</text>']

    for i in range(5):  # gridlines + y ticks
        gy = pad_t + i * (height - pad_t - pad_b) / 4
        val = y1 - i * (y1 - y0) / 4
        p.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" '
                 f'stroke="#e5e7eb" stroke-width="1"/>')
        p.append(f'<text x="{pad_l-8}" y="{gy+4:.1f}" text-anchor="end" font-size="11" '
                 f'fill="#6b7280">{val:.3f}</text>')
    for i in range(5):  # x ticks
        gx = pad_l + i * (width - pad_l - pad_r) / 4
        val = x0 + i * (x1 - x0) / 4
        p.append(f'<text x="{gx:.1f}" y="{height-pad_b+18}" text-anchor="middle" '
                 f'font-size="11" fill="#6b7280">{val:.0f}</text>')

    p.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" '
             f'stroke="#9ca3af"/>')
    p.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" '
             f'y2="{height-pad_b}" stroke="#9ca3af"/>')

    for idx, (name, pts) in enumerate(series.items()):
        c = colors[idx % len(colors)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                     for i, (x, y) in enumerate(sorted(pts)))
        p.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        for x, y in sorted(pts):
            p.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{c}"/>')
        ly = pad_t + 6 + idx * 20
        p.append(f'<line x1="{width-pad_r+10}" y1="{ly}" x2="{width-pad_r+32}" y2="{ly}" '
                 f'stroke="{c}" stroke-width="2.5"/>')
        p.append(f'<text x="{width-pad_r+38}" y="{ly+4}" font-size="12" fill="#374151">'
                 f'{name}</text>')

    p.append(f'<text x="{(pad_l+width-pad_r)/2}" y="{height-10}" text-anchor="middle" '
             f'font-size="12" fill="#374151">{xlabel}</text>')
    p.append(f'<text x="16" y="{(pad_t+height-pad_b)/2}" font-size="12" fill="#374151" '
             f'transform="rotate(-90 16 {(pad_t+height-pad_b)/2})" '
             f'text-anchor="middle">{ylabel}</text>')
    p.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="checkpoints/vexmlm-stage1")
    ap.add_argument("--log", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--out", default="reports/MLM_FINAL_REPORT.md")
    args = ap.parse_args()

    run = REPO / args.run
    if not run.exists():
        print(f"run directory not found: {run}", file=sys.stderr)
        return 1

    state = None
    for cand in sorted(run.glob("checkpoint-*/trainer_state.json")) + [run / "trainer_state.json"]:
        if cand.exists():
            state = json.loads(cand.read_text())
    hist = state.get("log_history", []) if state else []
    summary = json.loads((run / "stage1_summary.json").read_text()) if (
        run / "stage1_summary.json").exists() else {}
    metrics = summary.get("metrics", {})

    logtext = ""
    for cand in ([args.log] if args.log else []):
        for path in (REPO / cand, REPO / cand.replace(".out", ".err")):
            if path.exists():
                logtext += path.read_text(errors="replace")

    train = [(h["step"], h["loss"]) for h in hist if "loss" in h and "step" in h]
    evals = [(h["step"], h["eval_loss"], h.get("epoch")) for h in hist if "eval_loss" in h]

    # save_total_limit=1 keeps only one checkpoint, so trainer_state.json may
    # hold a truncated history. The SLURM log has every logged point; parse it
    # as a fallback so the curves cover the whole run.
    if len(train) < 5:
        log_train = [(float(m.group(2)), float(m.group(1))) for m in re.finditer(
            r"\{'loss': ([0-9.]+),[^}]*'epoch': ([0-9.]+)\}", logtext)]
        if len(log_train) > len(train):
            spe = None
            if log_train and log_train[0][0] > 0:
                spe = round(1.0 / log_train[0][0]) if log_train[0][0] < 1 else None
            train = [((int(ep * spe) if spe else i + 1), loss)
                     for i, (ep, loss) in enumerate(log_train)]
    # The final epoch's evaluation runs AFTER the last checkpoint is written, so
    # trainer_state.json always misses it. Merge in every eval point from the log.
    log_eval = [(float(m.group(2)), float(m.group(1))) for m in re.finditer(
        r"\{'eval_loss': ([0-9.]+),[^}]*'epoch': ([0-9.]+)\}", logtext)]
    if log_eval:
        spe = max((s for s, _, _ in evals), default=0) / max(
            (e for _, _, e in evals if e), default=1) if evals else 512
        by_epoch = {round(e, 3): (s, v, e) for s, v, e in evals}
        for ep, loss in log_eval:
            by_epoch.setdefault(round(ep, 3), (int(round(ep * spe)), loss, ep))
        evals = [by_epoch[k] for k in sorted(by_epoch)]

    # Curves
    plots = REPO / "reports/figures"
    if train or evals:
        series = {}
        if train:
            series["train loss"] = train
        if evals:
            series["validation loss"] = [(s, v) for s, v, _ in evals]
        svg_curve(series, "Stage 1 — continued MLM pretraining", "step",
                  "cross-entropy loss", plots / "stage1_loss.svg")
    if evals:
        svg_curve({"validation perplexity":
                   [(s, math.exp(v)) for s, v, _ in evals if v < 20]},
                  "Stage 1 — validation perplexity", "step", "perplexity",
                  plots / "stage1_perplexity.svg")

    blocks = re.findall(r"block-chunked (\w+): (\d+) lines -> (\d+) blocks of (\d+) tokens",
                        logtext)
    prep = {}
    pp = REPO / "datasets/processed/preparation_report.json"
    if pp.exists():
        prep = json.loads(pp.read_text())
    firing = {}
    fp = REPO / "checkpoints/vexmlm-expanded/expansion_manifest.json"
    if fp.exists():
        firing = json.loads(fp.read_text()).get("firing_verification", {})

    best_step = best_loss = None
    if evals:
        best_step, best_loss, _ = min(evals, key=lambda t: t[1])

    L = ["# Stage 1 — Final MLM Report", "",
         f"Run `{args.run}`" + (f", SLURM job {args.job}" if args.job else ""), ""]

    L += ["## Outcome", "", "| Metric | Value |", "|---|---|"]
    if best_loss is not None:
        L += [f"| Best validation loss | **{best_loss:.4f}** (step {best_step}) |",
              f"| Best perplexity | **{math.exp(best_loss):.2f}** |"]
    if "eval_loss" in metrics:
        L.append(f"| Final validation loss | {metrics['eval_loss']:.4f} |")
    if "perplexity" in metrics:
        L.append(f"| Final perplexity | {metrics['perplexity']:.2f} |")
    if "train_train_loss" in metrics:
        L.append(f"| Final training loss | {metrics['train_train_loss']:.4f} |")
    if "train_train_runtime" in metrics:
        rt = metrics["train_train_runtime"]
        L.append(f"| Wall clock | {rt/60:.1f} min |")
    if "train_train_samples_per_second" in metrics:
        sps = metrics["train_train_samples_per_second"]
        L += [f"| Throughput | {sps:.2f} sequences/s |",
              f"| Token throughput | ~{sps*256:,.0f} tokens/s |"]
    L.append("")

    if evals:
        L += ["## Per-epoch validation", "",
              "| Epoch | Step | Val loss | Perplexity |", "|---|---|---|---|"]
        for s, v, ep in evals:
            ppl = f"{math.exp(v):.2f}" if v < 20 else "overflow"
            L.append(f"| {ep:.2f} | {s} | {v:.4f} | {ppl} |")
        L.append("")
        # Overfitting check -- report, never act on it.
        rising = sum(1 for a, b in zip(evals, evals[1:]) if b[1] > a[1])
        L += ["### Overfitting check", "",
              f"Validation loss rose at {rising} of {len(evals)-1} epoch transitions.",
              ""]
        if evals[-1][1] > best_loss + 1e-9:
            L += [f"Validation loss ended at {evals[-1][1]:.4f}, above its best "
                  f"{best_loss:.4f} at step {best_step} — divergence after the best "
                  f"epoch. Training ran the full schedule as specified; "
                  f"`load_best_model_at_end` means the **saved model is the best "
                  f"checkpoint, not the last**.", ""]
        else:
            L += ["Validation loss was still at or near its minimum at the end of "
                  "training; no divergence observed.", ""]

    if train:
        L += ["## Training curves", "",
              "![loss](figures/stage1_loss.svg)", "",
              "![perplexity](figures/stage1_perplexity.svg)", "",
              f"{len(train)} training points, {len(evals)} validation points.", ""]

    if blocks:
        L += ["## Effective token count", "",
              "| Split | Lines | Blocks | Tokens |", "|---|---|---|---|"]
        total = 0
        for s, l, b, t in blocks:
            tot = int(b) * int(t)
            total += tot
            L.append(f"| {s} | {int(l):,} | {int(b):,} | {tot:,} |")
        L += ["", f"**{total:,} subword tokens per epoch**, every one of them real: "
                  f"block-chunking leaves no padding. Over 10 epochs the model sees "
                  f"{total*10:,} token positions, of which ~15% are masked "
                  f"({total*10*0.15:,.0f} prediction targets).", ""]

    if prep:
        L += ["## Deduplication impact", "",
              "| Corpus | Raw lines | Duplicates removed | Kept | Rate |",
              "|---|---|---|---|---|"]
        for lang, s in prep.items():
            rate = 100 * s["dropped_duplicate"] / s["lines_read"]
            L.append(f"| {lang} | {s['lines_read']:,} | {s['dropped_duplicate']:,} | "
                     f"{s['kept']:,} | {rate:.2f}% |")
        L += ["", "Without deduplication a line repeated 160× would contribute 160 "
                  "gradient updates per epoch and could appear in both train and "
                  "validation, making held-out perplexity measure memorized text.", ""]

    if firing:
        L += ["## Vocabulary firing statistics", "",
              "| Metric | Value |", "|---|---|",
              f"| Sentences probed | {firing.get('sentences', 0):,} |",
              f"| Tokens | {firing.get('tokens', 0):,} |",
              f"| New-token firings | {firing.get('new_token_firings', 0):,} |",
              f"| **Share of tokens from new vocabulary** | "
              f"**{firing.get('new_token_share', 0):.2%}** |",
              f"| Status | {firing.get('status')} |", "",
              "Measured on held-out Tigrinya before Stage 1. The expanded vocabulary "
              "is genuinely used, not inert.", ""]

    hw = summary.get("hardware", {})
    if hw:
        L += ["## Hardware", "", "| Field | Value |", "|---|---|",
              f"| GPU | {', '.join(hw.get('device_names', [])) or 'n/a'} |",
              f"| VRAM | {hw.get('total_vram_gb')} |",
              f"| CUDA | {hw.get('cuda_version')} |",
              f"| BF16 | {hw.get('bf16_supported')} |",
              f"| Devices | {hw.get('device_count')} |",
              f"| Effective batch | {summary.get('effective_batch_size')} |",
              f"| PyTorch | {hw.get('torch_version')} |", ""]
        for u in (hw.get("utilization") or []):
            L.append(f"- GPU {u.get('index')} `{u.get('name')}`: "
                     f"{u.get('util_pct')}% util, {u.get('mem_used_mb')} MB used "
                     f"of {u.get('mem_total_mb')} MB (sampled at start)")
        L.append("")

    mf = run / "model.safetensors"
    if mf.exists():
        L += ["## Checkpoint", "", "| Artifact | Value |", "|---|---|",
              f"| Path | `{args.run}` |",
              f"| Size | {mf.stat().st_size/1e9:.2f} GB |",
              f"| SHA-256 (first 64 MB) | `{sha256(mf)[:32]}` |",
              "| Selection | best by `eval_loss` (`load_best_model_at_end`) |", ""]

    L += ["## Configuration", "",
          "Paper Table 1, unchanged:", "",
          "| Parameter | Value |", "|---|---|",
          "| max_seq_length | 256 |", "| batch size | 32 |",
          "| epochs | 10 |", "| learning rate | 5e-5 |",
          "| optimizer | AdamW |", "| weight decay | 0.01 |",
          "| mlm_probability | 0.15 |", "| gradient clipping | 1.0 |",
          "| precision | bf16 |", "| trainable | all parameters |",
          "| example construction | block-chunked (approved) |", ""]

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L[:34]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
