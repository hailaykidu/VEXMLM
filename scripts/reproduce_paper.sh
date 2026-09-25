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
#
# The commands below are those that produced the released artifacts, as recorded
# in REPRODUCE.md section 3. They cannot be executed end to end outside the
# original environment, because the pretraining text is of undocumented origin
# and unknown licence and is not redistributed (docs/DATASET_PROVENANCE.md).
# Use --dry-run to print every command without running it.
#
# This script never writes into the released run's directories. Outputs go under
# a run root (default: runs/repro), so the artifacts of the original experiment
# -- checkpoints/vexmlm-expanded-spm, checkpoints/vexmlm-stage1-spm,
# checkpoints/vexmlm-spm-stage2, results/ and logs/ -- are left untouched.
# Override with --run-root DIR. The paths of the released run are given in
# comments at each step, for comparison only.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DRY_RUN=0
FORCE=0
ONLY_STAGE=""
SEEDS="42 43 44 45 46"
CONFIG="configs/base.yaml"
RUN_ROOT="runs/repro"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --force)    FORCE=1; shift ;;
    --stage)    ONLY_STAGE="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --config)   CONFIG="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Refuse to write over the released run, whatever --run-root is set to.
case "${RUN_ROOT%/}" in
  checkpoints|checkpoints/*|results|results/*|logs|logs/*|tokenizer/artifacts*|\
  vocabulary_expansion/artifacts*|datasets/processed*|.|"")
    echo "refusing --run-root '$RUN_ROOT': it would write into the released run's" >&2
    echo "artifacts. Choose a separate directory, e.g. runs/repro." >&2
    exit 2 ;;
esac

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
# Training splits written by datasets/prepare.py. The released tokenizers, token
# selection and Stage 1 all read these, not the raw files (REPRODUCE.md sec. 3).
AMH_TRAIN="datasets/processed/amharic.train.txt"
TIR_TRAIN="datasets/processed/tigrinya.train.txt"

# Everything this script produces goes under RUN_ROOT, never into the released
# run's directories. Released counterparts, for comparison only:
#   $RUN_ROOT/tokenizer            <- tokenizer/artifacts
#   $RUN_ROOT/vexmlm-expanded-spm  <- checkpoints/vexmlm-expanded-spm
#   $RUN_ROOT/vexmlm-stage1-spm    <- checkpoints/vexmlm-stage1-spm
#   $RUN_ROOT/stage2               <- checkpoints/vexmlm-spm-stage2
#   $RUN_ROOT/results              <- results/
TOK_DIR="$RUN_ROOT/tokenizer"
NEWTOK_DIR="$RUN_ROOT/new_tokens"
EXPANDED="$RUN_ROOT/vexmlm-expanded-spm"
EXPANDED_RANDOM="$RUN_ROOT/vexmlm-expanded-spm-random"
STAGE1="$RUN_ROOT/vexmlm-stage1-spm"
S2_OUT="$RUN_ROOT/stage2"
RESULTS_DIR="$RUN_ROOT/results"
[[ $DRY_RUN -eq 1 ]] || mkdir -p "$RUN_ROOT"

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
      --input "$AMH_TRAIN" --output-dir "$TOK_DIR/amharic"
  fi
  if ! skip_if_exists "$TOK_DIR/tigrinya/tigrinya_sp.model" "Tigrinya tokenizer"; then
    run python3 tokenizer/train_tigrinya_tokenizer.py \
      --input "$TIR_TRAIN" --output-dir "$TOK_DIR/tigrinya"
  fi
fi

# ------------------------------------------------------- Stage 3: vocab expansion
if stage_enabled 3; then
  log "=== Step 3/7: vocabulary expansion + mean-based initialization ==="
  if ! skip_if_exists ""$NEWTOK_DIR"/new_tokens.json" "token selection"; then
    run python3 tokenizer/extract_new_tokens.py \
      --spm "$TOK_DIR/tigrinya/tigrinya_sp.model" \
      --corpus "$TIR_TRAIN" \
      --max-new-tokens 20000 \
      --out "$NEWTOK_DIR"/new_tokens_tir.json
    run python3 tokenizer/extract_new_tokens.py \
      --spm "$TOK_DIR/amharic/amharic_sp.model" \
      --corpus "$AMH_TRAIN" \
      --max-new-tokens 20000 \
      --out "$NEWTOK_DIR"/new_tokens_amh.json
    run python3 vocabulary_expansion/merge_vocabularies.py \
      --inputs "$NEWTOK_DIR"/new_tokens_amh.json \
               "$NEWTOK_DIR"/new_tokens_tir.json \
      --target-total 30000 \
      --out "$NEWTOK_DIR"/new_tokens.json
  fi
  # Integration defaults to sentencepiece_merge, which is the released method;
  # add_tokens exists only to rebuild the superseded checkpoint and corrupts
  # Ge'ez decoding.
  if ! skip_if_exists "$EXPANDED/config.json" "expanded model (global_mean)"; then
    run python3 vocabulary_expansion/expand_xlmr_vocab.py \
      --new-tokens "$NEWTOK_DIR"/new_tokens.json \
      --init global_mean --seed 42 \
      --integration sentencepiece_merge \
      --validation-corpus datasets/processed/tigrinya.dev.txt \
      --output "$EXPANDED"
  fi
  # Table 5's "+VocabExp (Random Init)" arm.
  if ! skip_if_exists "$EXPANDED_RANDOM/config.json" "expanded model (random init)"; then
    run python3 vocabulary_expansion/expand_xlmr_vocab.py \
      --new-tokens "$NEWTOK_DIR"/new_tokens.json \
      --init random --seed 42 \
      --integration sentencepiece_merge \
      --validation-corpus datasets/processed/tigrinya.dev.txt \
      --output "$EXPANDED_RANDOM"
  fi
  run python3 vocabulary_expansion/verify_vocab.py --model "$EXPANDED" \
    --out "$RESULTS_DIR/init_verification.json"
fi

# ------------------------------------------------------ Stage 4: MLM pretraining
if stage_enabled 4; then
  log "=== Step 4/7: Stage 1 continued MLM pretraining ==="
  # configs/base.yaml defaults to 10 epochs; the released checkpoint used 60,
  # and the checkpoint of epoch 56 (lowest validation loss) is the one released.
  if ! skip_if_exists "$STAGE1/config.json" "Stage 1 model"; then
    run python3 pretraining/run_mlm.py --config "$CONFIG" \
      --model "$EXPANDED" \
      --amharic "$AMH_TRAIN" --tigrinya "$TIR_TRAIN" \
      --block-chunk --mode official \
      --output "$STAGE1" \
      --override pretraining.num_train_epochs=60 \
                 pretraining.save_total_limit=1 \
                 pretraining.logging_steps=50
  fi
fi

# ------------------------------------------------- Stage 5: downstream fine-tuning
if stage_enabled 5; then
  log "=== Step 5/7: Stage 2 fine-tuning over seeds [$SEEDS] ==="
  # The released runs were launched as scripts/slurm_stage2_spm_seeds.sh, which
  # calls these same runners with --mode official and writes to
  # checkpoints/vexmlm-spm-stage2/<task>-seed<N>. On a cluster, prefer that job.
  for SEED in $SEEDS; do
    log "--- seed $SEED ---"

    if [[ -d datasets/raw/tigqa ]]; then
      run python3 finetuning/qa/run_qa.py --config "$CONFIG" --model "$STAGE1" \
        --dataset tigqa --local-path datasets/raw/tigqa \
        --seed "$SEED" --mode official --output "$S2_OUT/tigqa-seed$SEED"
    else warn "TIGQA absent — skipping (docs/DATA_SETUP.md)"; fi

    if [[ -d datasets/raw/amqa ]]; then
      run python3 finetuning/qa/run_qa.py --config "$CONFIG" --model "$STAGE1" \
        --dataset amqa --local-path datasets/raw/amqa \
        --seed "$SEED" --mode official --output "$S2_OUT/amqa-seed$SEED"
    else warn "AmQA absent — skipping"; fi

    run python3 finetuning/ner/run_ner.py --config "$CONFIG" --model "$STAGE1" \
      --dataset masakhaner_amh --seed "$SEED" --mode official \
      --output "$S2_OUT/masakhaner_amh-seed$SEED"

    if [[ -d datasets/raw/tigrinya_ner ]]; then
      run python3 finetuning/ner/run_ner.py --config "$CONFIG" --model "$STAGE1" \
        --dataset tigrinya_ner --local-path datasets/raw/tigrinya_ner \
        --seed "$SEED" --mode official --output "$S2_OUT/tigrinya_ner-seed$SEED"
    else warn "Tigrinya NER absent — skipping"; fi

    run python3 finetuning/sentiment/run_sentiment.py --config "$CONFIG" \
      --model "$STAGE1" --dataset afrisenti --dataset-config amh \
      --seed "$SEED" --mode official --output "$S2_OUT/afrisenti-seed$SEED"
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
      --parallel --out "$RESULTS_DIR/tokenizer_metrics.json"
  else
    warn "no parallel corpus — parity will be reported as NOT VALID"
    run python3 evaluation/run_intrinsic.py \
      --tokenizer xlm-roberta-base "$EXPANDED" \
      --corpus amh=datasets/processed/amharic.dev.txt \
               tir=datasets/processed/tigrinya.dev.txt \
      --out "$RESULTS_DIR/tokenizer_metrics.json"
  fi
fi

# ------------------------------------------------------------- Stage 7: tables
if stage_enabled 7; then
  log "=== Step 7/7: table generation ==="
  run python3 evaluation/generate_tables.py --runs "$S2_OUT" --out "$RESULTS_DIR" \
    --seeds $SEEDS
fi

log "pipeline complete — outputs under $RUN_ROOT"
