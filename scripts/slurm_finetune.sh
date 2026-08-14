#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ft
#SBATCH --partition=standby
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_%A_%a.out
#SBATCH --error=logs/ft_%A_%a.err
#SBATCH --array=0-4              # one array task per seed
#
# Stage 2: fine-tuning across the five seeds, one GPU each.
#   sbatch scripts/slurm_finetune.sh qa tigqa
#   sbatch scripts/slurm_finetune.sh ner masakhaner_amh
#   sbatch scripts/slurm_finetune.sh sentiment afrisenti amh

set -euo pipefail
mkdir -p logs
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TASK="${1:?usage: slurm_finetune.sh <qa|ner|sentiment> <dataset> [config]}"
DATASET="${2:?dataset required}"
DS_CONFIG="${3:-}"

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${VEXMLM_MODEL:-checkpoints/vexmlm-stage1}"
OUT="checkpoints/${TASK}-${DATASET}${DS_CONFIG:+-$DS_CONFIG}-${SEED}"

echo "task=$TASK dataset=$DATASET seed=$SEED model=$MODEL"
nvidia-smi || true

ARGS=(--config configs/base.yaml --model "$MODEL" --dataset "$DATASET"
      --seed "$SEED" --output "$OUT")
[[ -n "$DS_CONFIG" ]] && ARGS+=(--dataset-config "$DS_CONFIG")
[[ -d "datasets/raw/$DATASET" ]] && ARGS+=(--local-path "datasets/raw/$DATASET")

case "$TASK" in
  qa)        srun python3 finetuning/qa/run_qa.py "${ARGS[@]}" ;;
  ner)       srun python3 finetuning/ner/run_ner.py "${ARGS[@]}" ;;
  sentiment) srun python3 finetuning/sentiment/run_sentiment.py "${ARGS[@]}" ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
