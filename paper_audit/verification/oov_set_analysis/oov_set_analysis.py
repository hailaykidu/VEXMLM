"""Composition of the word set used for Table 3 / Table 5 (evaluation/run_ner_oov.py).

Answers two review questions without running any model:
  1. Which clause of oov_word_set puts each word type in the set?
  2. What is the gold NER tag distribution of the two subsets?

Writes paper_audit/verification/oov_set_composition/word_set_composition.json (verification only; not a paper source). Uses the repository's own oov_word_set and the same first-subword positions as the
metric (VEXMLM SP-Merge tokenizer, max length 256), so the token totals must equal the
stored oov_tokens / non_oov_tokens of results/ablation/full_vexmlm-seed42.json.

    HF_HUB_OFFLINE=1 python3 paper_audit/verification/oov_set_analysis/oov_set_analysis.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "evaluation"), str(ROOT / "datasets"), str(ROOT / "src"), str(ROOT)]

from transformers import AutoTokenizer  # noqa: E402
from manager import DatasetManager  # noqa: E402
from run_ner_oov import BASELINE_TOKENIZER, oov_word_set  # noqa: E402

REF = ROOT / "checkpoints/vexmlm-expanded-spm"
ARM4 = ROOT / "checkpoints/ablation/full_vexmlm-tigrinya-seed42"
STORED = ROOT / "results/ablation/full_vexmlm-seed42.json"


def clause(w, base_tok, ref_tok):
    unk = base_tok.unk_token_id
    b = base_tok.encode(w, add_special_tokens=False)
    if unk in b:
        return "unk"
    if base_tok.decode(b, skip_special_tokens=True).strip() != w:
        return "no_roundtrip"
    if len(b) > len(ref_tok.encode(w, add_special_tokens=False)):
        return "more_pieces"
    return None


def main():
    base_tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER)
    ref_tok = AutoTokenizer.from_pretrained(str(REF))
    arm_tok = AutoTokenizer.from_pretrained(str(ARM4))
    split = DatasetManager().load("tigrinya_ner", local_path=str(ROOT / "datasets/raw/tigrinya_ner"))["test"]
    feat = getattr(split.features["ner_tags"], "feature", None)
    names = getattr(feat, "names", None)
    tag_name = (lambda t: names[t]) if names else (lambda t: t)

    vocab = sorted({w for row in split["tokens"] for w in row})
    oov = oov_word_set(vocab, base_tok, ref_tok)
    by_clause = collections.Counter(clause(w.strip(), base_tok, ref_tok) for w in vocab if w.strip() in oov)

    tags = {"oov": collections.Counter(), "non_oov": collections.Counter()}
    for words, gold in zip(split["tokens"], split["ner_tags"]):
        enc = arm_tok(words, is_split_into_words=True, truncation=True, max_length=256)
        prev = None
        for wid in enc.word_ids():
            if wid is None or wid == prev:
                prev = wid
                continue
            prev = wid
            bucket = "oov" if words[wid].strip() in oov else "non_oov"
            tags[bucket][tag_name(gold[wid])] += 1

    stored = json.loads(STORED.read_text())
    totals = {b: sum(c.values()) for b, c in tags.items()}
    report = {
        "word_types": len(vocab),
        "oov_word_types": len(oov),
        "oov_types_by_clause": dict(by_clause),
        "token_totals": totals,
        "stored_totals": {"oov": stored["metrics"]["oov_tokens"], "non_oov": stored["metrics"]["non_oov_tokens"]},
        "gold_tag_counts": {b: dict(sorted(c.items())) for b, c in tags.items()},
        "entity_share_pct": {b: round(100 * (1 - c["O"] / totals[b]), 1) for b, c in tags.items()},
    }
    assert len(oov) == stored["oov_word_types"], "OOV type count differs from stored ablation result"
    assert totals == report["stored_totals"], "token totals differ from stored ablation result"
    out = ROOT / "paper_audit/verification/oov_set_composition/word_set_composition.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
