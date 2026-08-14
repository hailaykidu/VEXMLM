#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-stage2
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage2_%A_%a.out
#SBATCH --error=logs/stage2_%A_%a.err
#SBATCH --array=0-4
#
# Stage 2: all five task/dataset combinations for one seed, one array task per
# seed. Paper fine-tuning settings come from configs/base.yaml and are not
# overridden here.
#
#   sbatch scripts/slurm_stage2_all.sh
#   VEXMLM_MODEL=checkpoints/vexmlm-stage1 sbatch scripts/slurm_stage2_all.sh

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
MODEL="${VEXMLM_MODEL:-checkpoints/vexmlm-stage1}"

echo "=== Stage 2 seed=$SEED model=$MODEL node=$(hostname) $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 scripts/gpu_probe.py --out "reports/GPU_ENVIRONMENT.md" \
  || { echo "GPU verification FAILED; aborting" >&2; exit 2; }

run () {  # run <label> <script> <args...>
  local label="$1"; shift
  echo ""; echo "--- $label (seed $SEED) ---"
  if python3 "$@"; then
    echo "OK: $label seed=$SEED"
  else
    echo "FAILED: $label seed=$SEED (continuing with remaining tasks)" >&2
    FAILURES="${FAILURES:-}$label(seed$SEED) "
  fi
}

COMMON=(--config configs/base.yaml --model "$MODEL" --seed "$SEED" --mode official)

run "QA/TIGQA"        finetuning/qa/run_qa.py "${COMMON[@]}" \
    --dataset tigqa          --output "checkpoints/qa-tigqa-$SEED"
run "QA/AmQA"         finetuning/qa/run_qa.py "${COMMON[@]}" \
    --dataset amqa           --output "checkpoints/qa-amqa-$SEED"
run "NER/MasakhaNER"  finetuning/ner/run_ner.py "${COMMON[@]}" \
    --dataset masakhaner_amh --output "checkpoints/ner-masakhaner-$SEED"
run "NER/Tigrinya"    finetuning/ner/run_ner.py "${COMMON[@]}" \
    --dataset tigrinya_ner   --output "checkpoints/ner-tigrinya-$SEED"
run "SA/AfriSenti"    finetuning/sentiment/run_sentiment.py "${COMMON[@]}" \
    --dataset afrisenti --dataset-config amh \
    --output "checkpoints/sa-afrisenti-$SEED"

echo ""
if [[ -n "${FAILURES:-}" ]]; then
  echo "=== seed $SEED finished with failures: $FAILURES ==="
  exit 1
fi
echo "=== seed $SEED: all 5 tasks OK $(date -Is) ==="
