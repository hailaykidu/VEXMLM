#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-spm-stage2
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/spm_stage2_%A_%a.out
#SBATCH --error=logs/spm_stage2_%A_%a.err
#SBATCH --array=0-4
#
# SP-Merge Stage 2: all six downstream tasks for one seed, one array task per
# seed (42-46). This is the authoritative downstream evaluation set.
#
# Procedure is identical to scripts/internal/stage2_spm_seed42.sh, which
# produced the original seed-42 run, except that the seed is taken from the
# array index. Same model, same six tasks, same --mode official, same
# --local-path handling, same output layout.
#
# All five seeds run under the deterministic configuration added in
# fix/deterministic-evaluation (enable_full_determinism,
# CUBLAS_WORKSPACE_CONFIG=:4096:8, dataloader_num_workers=0), so every run in
# the set shares one config hash. Seed 42 is re-run for exactly this reason:
# its original execution predates those settings.
#
# TiQuAD is included as a supplementary diagnostic task, not a paper benchmark.
#
# --gres requests gpu:a100:1, the generic type available on gpunode02-04,06.
# The a100_80gb type exists only on gpunode05, which is draining; the original
# seed-42 Stage 2 runs used an A100-PCIE-40GB, so the generic type matches the
# hardware those results came from.
#
#   sbatch scripts/slurm_stage2_spm_seeds.sh
#   sbatch --array=1-4 scripts/slurm_stage2_spm_seeds.sh   # seeds 43-46 only

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
MODEL="${VEXMLM_MODEL:-checkpoints/vexmlm-stage1-spm}"
OUTPUT_BASE="checkpoints/vexmlm-spm-stage2"
MODE="official"

echo "=== SP-Merge Stage 2 seed=$SEED model=$MODEL node=$(hostname) $(date -Is) ==="
[ -d "$MODEL" ] || { echo "ERROR: $MODEL missing; Stage 1 must finish first" >&2; exit 2; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Record the config hash so the run set can be verified as homogeneous.
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from vexmlm import config as c
print("config_hash:", c.config_hash(c.load_config("configs/base.yaml")))
PY

mkdir -p "$OUTPUT_BASE"

TASKS=("amqa" "tigqa" "tiquad" "masakhaner_amh" "tigrinya_ner" "afrisenti")
TYPES=("qa" "qa" "qa" "ner" "ner" "sentiment")

FAILURES=""
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"; type="${TYPES[$i]}"
    echo ""
    echo "--- [$((i+1))/6] $task (seed $SEED) ---"

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
        2>&1 | tee "logs/spm-stage2-${task}-${SEED}.log"; then
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
echo "=== seed $SEED: all 6 tasks OK $(date -Is) ==="
