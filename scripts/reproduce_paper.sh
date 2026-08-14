#!/usr/bin/env bash
# End-to-end VEXMLM pipeline: data -> tokenizers -> expansion -> Stage 1 ->
# Stage 2 (QA/NER/SA, 5 seeds) -> evaluation -> tables.
#
# Usage:
#   scripts/reproduce_paper.sh [--dry-run] [--stage N] [--seeds "42 43"]
#
# Every stage is idempotent: existing outputs are skipped unless --force.
# Corpora and the QA/NER datasets have no public downloader; place them under
# datasets/raw/ first (see docs/DATA_SETUP.md) or the pipeline stops with a
# clear message rather than inventing data.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DRY_RUN=0
FORCE=0
ONLY_STAGE=""
SEEDS="42 43 44 45 46"
CONFIG="configs/base.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    --stage)   ONLY_STAGE="$2"; shift 2 ;;
    --seeds)   SEEDS="$2"; shift 2 ;;
    --config)  CONFIG="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  log "\$ $*"
  [[ $DRY_RUN -eq 1 ]] && return 0
  "$@"
}

skip_if_exists() {  # $1 = path, $2 = description
  if [[ $FORCE -eq 0 && -e "$1" ]]; then
    log "skip: $2 already exists ($1)"
    return 0
  fi
  return 1
}

stage_enabled() { [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == "$1" ]]; }

require_data() {  # $1 = path, $2 = what, $3 = how to get it
  if [[ ! -e "$1" ]]; then
    warn "missing $2: $1"
    warn "  -> $3"
    return 1
  fi
  return 0
}

AMH_CORPUS="datasets/raw/amharic"
TIR_CORPUS="datasets/raw/tigrinya"
TOK_DIR="tokenizer/artifacts"
EXPANDED="checkpoints/vexmlm-expanded"
STAGE1="checkpoints/vexmlm-stage1"

log "VEXMLM pipeline — repo $REPO"
log "config=$CONFIG seeds='$SEEDS' dry_run=$DRY_RUN"
python3 -c "import sys;sys.path.insert(0,'src');from vexmlm.device import log_environment;log_environment()" \
  2>/dev/null || warn "device probe failed"

# ---------------------------------------------------------------- Stage 1: data
if stage_enabled 1; then
  log "=== Step 1/7: dataset preparation ==="
  MISSING=0
  require_data "$AMH_CORPUS" "Amharic MLM corpus" \
    "place .txt files in $AMH_CORPUS (see docs/DATA_SETUP.md)" || MISSING=1
  require_data "$TIR_CORPUS" "Tigrinya MLM corpus" \
    "place .txt files in $TIR_CORPUS (see docs/DATA_SETUP.md)" || MISSING=1
  if [[ $MISSING -eq 1 ]]; then
    die "pretraining corpora absent — cannot continue. See docs/DATA_SETUP.md."
  fi
  run python3 datasets/prepare.py --config "$CONFIG"
fi

# ----------------------------------------------------------- Stage 2: tokenizers
if stage_enabled 2; then
  log "=== Step 2/7: SentencePiece tokenizers (amh 32k / tir 50k) ==="
  if ! skip_if_exists "$TOK_DIR/amharic/amharic_sp.model" "Amharic tokenizer"; then
    run python3 tokenizer/train_amharic_tokenizer.py \
      --input "$AMH_CORPUS"/*.txt --output-dir "$TOK_DIR/amharic"
  fi
  if ! skip_if_exists "$TOK_DIR/tigrinya/tigrinya_sp.model" "Tigrinya tokenizer"; then
    run python3 tokenizer/train_tigrinya_tokenizer.py \
      --input "$TIR_CORPUS"/*.txt --output-dir "$TOK_DIR/tigrinya"
  fi
fi

# ------------------------------------------------------- Stage 3: vocab expansion
if stage_enabled 3; then
  log "=== Step 3/7: vocabulary expansion + mean-based initialization ==="
  if ! skip_if_exists "vocabulary_expansion/artifacts/new_tokens.json" "token selection"; then
    run python3 tokenizer/extract_new_tokens.py \
      --spm "$TOK_DIR/tigrinya/tigrinya_sp.model" \
      --corpus datasets/processed/tigrinya.train.txt \
      --max-new-tokens 15000 \
      --out vocabulary_expansion/artifacts/new_tokens_tir.json
    run python3 tokenizer/extract_new_tokens.py \
      --spm "$TOK_DIR/amharic/amharic_sp.model" \
      --corpus datasets/processed/amharic.train.txt \
      --max-new-tokens 15000 \
      --out vocabulary_expansion/artifacts/new_tokens_amh.json
    run python3 vocabulary_expansion/merge_vocabularies.py \
      --inputs vocabulary_expansion/artifacts/new_tokens_amh.json \
               vocabulary_expansion/artifacts/new_tokens_tir.json \
      --target-total 30000 \
      --out vocabulary_expansion/artifacts/new_tokens.json
  fi
  if ! skip_if_exists "$EXPANDED/config.json" "expanded model"; then
    run python3 vocabulary_expansion/expand_xlmr_vocab.py \
      --new-tokens vocabulary_expansion/artifacts/new_tokens.json \
      --init global_mean --seed 42 \
      --validation-corpus datasets/processed/tigrinya.dev.txt \
      --output "$EXPANDED"
  fi
  run python3 vocabulary_expansion/verify_vocab.py --model "$EXPANDED" \
    --out results/init_verification.json
fi

# ------------------------------------------------------ Stage 4: MLM pretraining
if stage_enabled 4; then
  log "=== Step 4/7: Stage 1 continued MLM pretraining ==="
  if ! skip_if_exists "$STAGE1/config.json" "Stage 1 model"; then
    run python3 pretraining/run_mlm.py --config "$CONFIG" \
      --model "$EXPANDED" \
      --amharic "$AMH_CORPUS"/*.txt --tigrinya "$TIR_CORPUS"/*.txt \
      --output "$STAGE1"
  fi
fi

# ------------------------------------------------- Stage 5: downstream fine-tuning
if stage_enabled 5; then
  log "=== Step 5/7: Stage 2 fine-tuning over seeds [$SEEDS] ==="
  for SEED in $SEEDS; do
    log "--- seed $SEED ---"

    if [[ -d datasets/raw/tigqa ]]; then
      run python3 finetuning/qa/run_qa.py --config "$CONFIG" --model "$STAGE1" \
        --dataset tigqa --local-path datasets/raw/tigqa \
        --seed "$SEED" --output "checkpoints/qa-tigqa-$SEED"
    else warn "TIGQA absent — skipping (docs/DATA_SETUP.md)"; fi

    if [[ -d datasets/raw/amqa ]]; then
      run python3 finetuning/qa/run_qa.py --config "$CONFIG" --model "$STAGE1" \
        --dataset amqa --local-path datasets/raw/amqa \
        --seed "$SEED" --output "checkpoints/qa-amqa-$SEED"
    else warn "AmQA absent — skipping"; fi

    run python3 finetuning/ner/run_ner.py --config "$CONFIG" --model "$STAGE1" \
      --dataset masakhaner_amh --seed "$SEED" \
      --output "checkpoints/ner-amh-$SEED"

    if [[ -d datasets/raw/tigrinya_ner ]]; then
      run python3 finetuning/ner/run_ner.py --config "$CONFIG" --model "$STAGE1" \
        --dataset tigrinya_ner --local-path datasets/raw/tigrinya_ner \
        --seed "$SEED" --output "checkpoints/ner-tir-$SEED"
    else warn "Tigrinya NER absent — skipping"; fi

    run python3 finetuning/sentiment/run_sentiment.py --config "$CONFIG" \
      --model "$STAGE1" --dataset afrisenti --dataset-config amh \
      --seed "$SEED" --output "checkpoints/sa-amh-$SEED"
  done
fi

# ---------------------------------------------------------- Stage 6: evaluation
if stage_enabled 6; then
  log "=== Step 6/7: intrinsic evaluation ==="
  if [[ -f datasets/processed/parallel.amh.txt && -f datasets/processed/parallel.tir.txt ]]; then
    run python3 evaluation/run_intrinsic.py \
      --tokenizer xlm-roberta-base "$EXPANDED" \
      --corpus amh=datasets/processed/parallel.amh.txt \
               tir=datasets/processed/parallel.tir.txt \
      --parallel --out results/tokenizer_metrics.json
  else
    warn "no parallel corpus — parity will be reported as NOT VALID"
    run python3 evaluation/run_intrinsic.py \
      --tokenizer xlm-roberta-base "$EXPANDED" \
      --corpus amh=datasets/processed/amharic.dev.txt \
               tir=datasets/processed/tigrinya.dev.txt \
      --out results/tokenizer_metrics.json
  fi
fi

# ------------------------------------------------------------- Stage 7: tables
if stage_enabled 7; then
  log "=== Step 7/7: table generation ==="
  run python3 evaluation/generate_tables.py --runs checkpoints --out results \
    --seeds $SEEDS
fi

log "pipeline complete — see results/RESULTS.md"
