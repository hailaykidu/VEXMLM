#!/usr/bin/env python3
"""Identify which initialization strategy produced a checkpoint, from its weights.

The test exploits the fact that the strategies leave separable statistical
signatures in E[||new_row - global_mean||]:

    global_mean       -> 0
    mixed (N(0,1))    -> 0.5*sqrt(d)   = 13.856 for d=768
    random (N(0,1))   -> 1.0*sqrt(d)   = 27.713 for d=768
    constituent_mean  -> small, and HIGH VARIANCE across rows (each row is the
                         mean of a different subword set), which distinguishes
                         it from global_mean's near-zero variance.

    python vocabulary_expansion/verify_vocab.py \
        --model ~/VEXMLM_Model --n-old 250002 --out results/init_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from initialization import predict_distance  # noqa: E402


def load_embedding(model_dir: Path) -> torch.Tensor:
    st = list(model_dir.glob("*.safetensors"))
    if st:
        from safetensors import safe_open
        with safe_open(str(st[0]), framework="pt") as f:
            key = next((k for k in f.keys() if "word_embeddings" in k), None)
            if key is None:
                raise SystemExit(f"no word_embeddings tensor in {st[0]}")
            return f.get_tensor(key).float()
    bins = list(model_dir.glob("pytorch_model*.bin"))
    if bins:
        sd = torch.load(bins[0], map_location="cpu", weights_only=True)
        key = next(k for k in sd if "word_embeddings" in k)
        return sd[key].float()
    raise SystemExit(f"no weights found in {model_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--n-old", type=int, default=250002)
    ap.add_argument("--out", type=Path, default=Path("results/init_verification.json"))
    args = ap.parse_args()

    w = load_embedding(args.model.expanduser())
    n_old = args.n_old
    if w.shape[0] <= n_old:
        raise SystemExit(f"embedding has {w.shape[0]} rows, not more than n_old={n_old}")

    old, new = w[:n_old], w[n_old:]
    hidden = int(w.shape[1])
    gmean = old.mean(0)
    d = (new - gmean).norm(dim=1)
    base_std = old.std().item()
    observed = d.mean().item()

    candidates = {}
    for strat in ("global_mean", "mixed", "random"):
        pred = predict_distance(strat, hidden, base_std, legacy_unit_normal=True)
        candidates[strat] = {
            "predicted_distance": round(pred, 4),
            "relative_error": (round(abs(observed - pred) / pred, 6)
                               if pred else round(observed, 6)),
        }
    # constituent_mean has no closed form; identify it by dispersion instead.
    candidates["constituent_mean"] = {
        "predicted_distance": "no closed form (data-dependent)",
        "signature": "small mean distance with HIGH relative row-to-row variance",
        "observed_cv": round((d.std() / d.mean()).item(), 4),
    }

    best = min(("global_mean", "mixed", "random"),
               key=lambda s: candidates[s]["relative_error"])

    report = {
        "model": str(args.model),
        "embedding_shape": list(w.shape),
        "n_old": n_old, "n_new": int(new.shape[0]), "hidden_size": hidden,
        "observed": {
            "old_perdim_std": round(base_std, 6),
            "new_perdim_std": round(new.std().item(), 6),
            "old_norm_mean": round(old.norm(dim=1).mean().item(), 4),
            "new_norm_mean": round(new.norm(dim=1).mean().item(), 4),
            "global_mean_norm": round(gmean.norm().item(), 4),
            "dist_to_global_mean": {
                "mean": round(observed, 4),
                "std": round(d.std().item(), 4),
                "min": round(d.min().item(), 4),
                "max": round(d.max().item(), 4),
                "coefficient_of_variation": round((d.std() / d.mean()).item(), 4),
            },
            "rows_identical_to_global_mean": int((d < 1e-6).sum().item()),
        },
        "candidates": candidates,
        "verdict": best,
        "confidence": ("high" if candidates[best]["relative_error"] < 0.01 else "low"),
        "histogram": {
            "bin_edges": [round(x, 3) for x in
                          torch.linspace(d.min(), d.max(), 21).tolist()],
            "counts": torch.histc(d, bins=20, min=d.min().item(),
                                  max=d.max().item()).int().tolist(),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "histogram"}, indent=2))
    print(f"\nVERDICT: {best} (relative error "
          f"{candidates[best]['relative_error']}, confidence {report['confidence']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
