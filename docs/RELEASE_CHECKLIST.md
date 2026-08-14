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

- [x] TIGQA integrated (797 QA pairs / 365 contexts; Zenodo 11423987)
- [x] AmQA integrated (1,723/600/299 questions; MIT)
- [x] MasakhaNER integrated (1,750/250/500 sentences; CC BY 4.0)
- [x] Tigrinya NER integrated (4,562/570/571 sentences; Yohannes & Amagasa 2022)
- [x] AfriSenti identified (CC BY 4.0) with citation
- [x] MLM corpora integrated (200K lines per language)
- [x] Dataset card written for every dataset (`datasets/cards/`, 7 cards)
- [x] Every card states source, licence, version, checksum, and split statistics
- [x] `docs/DATASET_PROVENANCE.md` records origin and preprocessing for each
- [x] All copies checksum-verified against source (14/14)
- [ ] **BLOCKER** — MLM corpus provenance: designated source is HornMT, but the
      imported files are 200K lines each while HornMT holds ~2,030 sentence
      pairs. True source and licence unknown. See DATASET_PROVENANCE.md.
- [ ] Tigrinya NER licence confirmed redistributable (source repo states none)
- [ ] TIGQA licence confirmed redistributable (check the Zenodo record)
- [ ] Sentence-aligned Amharic–Tigrinya parallel corpus for Table 2 parity

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

1. **MLM corpus provenance (unresolved).** The designated source is HornMT, but
   HornMT is a ~2,030-sentence parallel corpus and the imported files hold
   200,000 monolingual lines each — two orders of magnitude apart. Content
   sampling suggests a CC-100/OSCAR-style crawl. Until the authors confirm the
   real source, these corpora cannot be redistributed and no model trained on
   them can ship a complete data statement. Evidence in
   [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md).
2. **Licences to confirm.** Tigrinya NER and TIGQA are integrated and usable,
   but their redistribution terms need checking against the source repository
   and Zenodo record respectively.
3. **Parallel corpus.** Without sentence-aligned text, Table 2 parity cannot be
   reported as a valid fairness measure.
4. **Upstream data defect (handled).** Tigrinya NER `train.conll:1350` carries a
   malformed `B-LO` tag. It is repaired in memory at read time and logged; the
   source file is untouched. Worth reporting upstream.
