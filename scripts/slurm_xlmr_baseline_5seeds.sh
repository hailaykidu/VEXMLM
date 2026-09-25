#!/usr/bin/env bash
#SBATCH --job-name=xlmr-baseline-5seeds
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/xlmr_baseline_%A_%a.out
#SBATCH --error=logs/xlmr_baseline_%A_%a.err
#SBATCH --array=0-4
#
# XLM-R downstream baseline over seeds 42-46, one array task per seed. PREPARED,
# NOT RUN. It replaces the single-seed baseline in results/baselines/xlmr_seed42/.
#
# Mirrors scripts/slurm_stage2_spm_seeds.sh exactly -- same runners, config,
# --mode official, dataset handling and deterministic settings -- with the model
# set to xlm-roberta-base and only the five paper tasks (no TiQuAD). Seed 42 is
# re-run because the existing seed-42 run predates enable_full_determinism.
#
#   sbatch scripts/slurm_xlmr_baseline_5seeds.sh
#
# Afterwards, aggregate with the same code path as VEXMLM:
#   python3 evaluation/export_spm_results.py --runs checkpoints/xlmr-baseline-5seeds \
#       --out results/baselines/xlmr_5seeds --csv results/baselines/xlmr_5seeds/metrics.csv
# Check the CSV's model column label before use; the exporter was written for VEXMLM.

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"
mkdir -p logs

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
MODEL="xlm-roberta-base"
OUTPUT_BASE="checkpoints/xlmr-baseline-5seeds"
MODE="official"

echo "=== XLM-R baseline seed=$SEED node=$(hostname) $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from vexmlm import config as c
print("config_hash:", c.config_hash(c.load_config("configs/base.yaml")))
PY

mkdir -p "$OUTPUT_BASE"

TASKS=("amqa" "tigqa" "masakhaner_amh" "tigrinya_ner" "afrisenti")
TYPES=("qa" "qa" "ner" "ner" "sentiment")

FAILURES=""
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"; type="${TYPES[$i]}"
    echo ""
    echo "--- [$((i+1))/${#TASKS[@]}] $task (seed $SEED) ---"

    args=(
        --config configs/base.yaml
        --model "$MODEL"
        --dataset "$task"
        --seed "$SEED"
        --mode "$MODE"
        --output "${OUTPUT_BASE}/${task}-seed${SEED}"
    )
    [ -d "datasets/raw/${task}" ] && args+=(--local-path "datasets/raw/${task}")
    [ "$task" = "afrisenti" ] && args+=(--dataset-config amh)

    if python3 "finetuning/${type}/run_${type}.py" "${args[@]}" \
        2>&1 | tee "logs/xlmr-baseline-${task}-${SEED}.log"; then
        echo "OK: $task seed=$SEED"
    else
        echo "FAILED: $task seed=$SEED (continuing with remaining tasks)" >&2
        FAILURES="${FAILURES}${task} "
    fi
done

echo ""
if [[ -n "$FAILURES" ]]; then
  echo "=== seed $SEED finished with failures: $FAILURES ==="
  exit 1
fi
echo "=== seed $SEED: all ${#TASKS[@]} tasks OK $(date -Is) ==="
