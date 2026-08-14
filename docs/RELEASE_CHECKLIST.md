# Release Checklist

Gate for publishing VEXMLM models and results. Items marked **BLOCKER** must be
resolved before any public release.

## Code

- [x] Vocabulary expansion preserves all original XLM-R entries
- [x] Mean-based initialization implements the paper's formula (test-asserted)
- [x] All four initialization strategies available; `global_mean` default
- [x] Encoding gate rejects byte-level-BPE tokens
- [x] Firing check verifies new tokens are reachable after expansion
- [x] Every RNG seeded; runs deterministic given a seed
- [x] No hardcoded absolute paths
- [x] GPU auto-detection with BF16/FP16 policy and DDP support
- [x] Checkpoint resume on all long-running stages
- [x] Test suite passes (`pytest -q`)
- [ ] Ruff clean (`ruff check .`)

## Data

- [ ] **BLOCKER** — Amharic MLM corpus: source, licence, and citation recorded
- [ ] **BLOCKER** — Tigrinya MLM corpus: source, licence, and citation recorded
- [ ] **BLOCKER** — Tigrinya NER corpus identified and licensed
      (MasakhaNER covers Amharic but *not* Tigrinya)
- [ ] TIGQA obtained; licence confirmed redistributable
- [ ] AmQA obtained; licence confirmed redistributable
- [x] AfriSenti identified (CC BY 4.0) with citation
- [x] MasakhaNER identified (CC BY 4.0) with citation
- [ ] Sentence-aligned Amharic–Tigrinya parallel corpus for Table 2 parity
- [ ] Dataset card written for every dataset (`datasets/cards/`)
- [ ] Every card states source, licence, citation, hash, and split statistics

Do not redistribute any dataset whose licence is `UNKNOWN`.

## Experiments

- [ ] Stage 1 continued MLM completed and logged
- [ ] QA fine-tuned on TIGQA and AmQA, seeds 42–46
- [ ] NER fine-tuned on MasakhaNER and Tigrinya NER, seeds 42–46
- [ ] Sentiment fine-tuned on AfriSenti, seeds 42–46
- [ ] All four Table 5 ablation arms run
- [ ] Every reported number is mean ± std over ≥3 seeds
- [ ] No single-seed value reported without that label

## Evaluation

- [ ] Intrinsic metrics computed for XLM-R and VEXMLM
- [ ] Parity computed on a **parallel** corpus, or explicitly marked invalid
- [ ] New-token firing rate reported (guards against an inert expansion)
- [ ] Tables 2–5 generated from recorded runs
- [ ] Remaining `NOT YET MEASURED` cells are intentional and explained

## Models

- [ ] `MODEL_CARD.md` for every released checkpoint
- [ ] Each card records vocab size, init method, corpora, hyperparameters,
      metrics, limitations, and ethical considerations
- [ ] `verify_vocab.py` run on the pre-Stage-1 checkpoint; verdict `global_mean`
- [ ] Config hash, git commit, and tokenizer hash recorded per checkpoint

## Documentation

- [x] README with method, quick start, and layout
- [x] DATA_SETUP.md
- [x] INITIALIZATION_VERIFICATION.md
- [x] REFERENCE_ARTIFACTS.md
- [x] MODEL_CARD_TEMPLATE.md
- [x] LICENSE (Apache-2.0) and CITATION.cff
- [ ] Paper citation updated with venue and DOI on acceptance

## Integrity

- [x] No fabricated results anywhere; unmeasured cells say so
- [x] Discrepancies between reference artifacts and the paper documented with
      evidence, both versions preserved
- [ ] Every number in the paper traceable to a run record in
      `experiment_tracking/`
- [ ] Independent rerun from a clean checkout reproduces reported numbers within
      the reported std

## Known blockers

1. **Corpus provenance.** The Stage 1 corpora and the Tigrinya NER set have no
   recorded source or licence. They cannot be redistributed, and results from
   them cannot be independently checked, until this is resolved.
2. **Tigrinya NER source.** Must be named explicitly; it cannot be inferred, and
   MasakhaNER does not cover Tigrinya.
3. **Parallel corpus.** Without sentence-aligned text, Table 2 parity cannot be
   reported as a valid fairness measure.
