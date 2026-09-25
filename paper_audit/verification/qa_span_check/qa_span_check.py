"""Verify QA answer-span mapping of run_qa.prepare_train for VEXMLM vs XLM-R tokenizers.

Uses the repo's own prepare_train so the check exercises the exact labelling code.
Read-only: loads datasets and tokenizers, writes a JSON report to the scratchpad.
"""
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "finetuning/qa"), str(REPO / "datasets"), str(REPO / "src"), str(REPO)]

from transformers import AutoTokenizer  # noqa: E402
from run_qa import prepare_train  # noqa: E402
from manager import DatasetManager  # noqa: E402

N = 50
MAX_LEN, STRIDE = 256, 128
TOKENIZERS = {
    "vexmlm_spm": str(REPO / "checkpoints/vexmlm-stage1-spm"),
    "xlmr_base": "xlm-roberta-base",
}
DATASETS = {"amqa": REPO / "datasets/raw/amqa", "tigqa": REPO / "datasets/raw/tigqa"}


def classify(recon: str, gold: str) -> str:
    if recon == gold:
        return "exact"
    if recon.strip() == gold.strip():
        return "exact_ws"
    if unicodedata.normalize("NFC", recon.strip()) == unicodedata.normalize("NFC", gold.strip()):
        return "exact_nfc"
    if gold.strip() and gold.strip() in recon:
        return "superset"
    if recon and recon.strip() in gold:
        return "subset"
    return "mismatch"


def check(ds_name, split, tok_name, tok):
    dm = DatasetManager()
    ds = dm.load(ds_name, local_path=str(DATASETS[ds_name]))[split]
    ds = ds.filter(lambda ex: bool(ex["answers"]["text"]) and bool(ex["answers"]["text"][0]))
    ds = ds.select(range(min(N, len(ds))))
    rows, counts = [], {}
    for ex in ds:
        batch = {k: [ex[k]] for k in ("question", "context", "answers")}
        feats = prepare_train(batch, tok, MAX_LEN, STRIDE)
        gold = ex["answers"]["text"][0]
        a0 = ex["answers"]["answer_start"][0]
        gold_in_ctx = ex["context"][a0:a0 + len(gold)] == gold
        enc = tok(ex["question"].lstrip(), ex["context"], truncation="only_second",
                  max_length=MAX_LEN, stride=STRIDE, return_overflowing_tokens=True,
                  return_offsets_mapping=True, padding="max_length")
        cls_id = tok.cls_token_id
        found = None
        for i, (s, e) in enumerate(zip(feats["start_positions"], feats["end_positions"])):
            cls_index = feats["input_ids"][i].index(cls_id)
            if s == cls_index and e == cls_index:
                continue
            off = enc["offset_mapping"][i]
            recon = ex["context"][off[s][0]:off[e][1]]
            found = {"window": i, "start_tok": s, "end_tok": e, "n_tokens": e - s + 1,
                     "recon": recon, "status": classify(recon, gold)}
            break
        status = found["status"] if found else "not_in_any_window"
        counts[status] = counts.get(status, 0) + 1
        rows.append({"id": ex["id"], "gold": gold, "gold_matches_context": gold_in_ctx,
                     "n_windows": len(feats["input_ids"]), **(found or {"status": status})})
    return {"dataset": ds_name, "split": split, "tokenizer": tok_name, "n": len(rows),
            "counts": counts, "gold_text_mismatch_with_context": sum(not r["gold_matches_context"] for r in rows),
            "rows": rows}


def main():
    report = []
    for tok_name, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(path)
        assert tok.is_fast, f"{tok_name} is not a fast tokenizer"
        for ds_name in DATASETS:
            for split in ("train", "validation"):
                try:
                    r = check(ds_name, split, tok_name, tok)
                except KeyError:
                    continue
                report.append(r)
                print(f"{tok_name:11s} {ds_name:6s} {split:10s} n={r['n']} {r['counts']} "
                      f"gold!=ctx:{r['gold_text_mismatch_with_context']}")
    out = Path(__file__).with_name("qa_span_check_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
