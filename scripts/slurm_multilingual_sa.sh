#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ml-sa
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpunode08      # Blackwell RTX PRO 6000: no kernels in this torch build
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/ml-sa_%A_%a.out
#SBATCH --error=logs/slurm/ml-sa_%A_%a.err
#SBATCH --array=0-59
#
# Expanded multilingual evaluation — AfriSenti sentiment analysis.
# 12 trainable languages x 5 seeds = 60 array tasks.
#
#   sbatch scripts/slurm_multilingual_sa.sh
#
# Outputs land under results/multilingual_evaluation/sentiment/ and never
# touch the published release artifacts in results/.

set -euo pipefail
mkdir -p logs/slurm

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

source /opt/conda/bin/activate /homes/neumann/teklehaymanot/.conda/envs/Gllm

LANGS=(amh arq ary hau ibo kin pcm por swa tso twi yor)
SEEDS=(42 43 44 45 46)

LANG_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
LANG="${LANGS[$LANG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

MODEL="${VEXMLM_MODEL:-checkpoints/vexmlm-stage1-spm}"
OUT="results/multilingual_evaluation/sentiment/afrisenti-${LANG}-seed${SEED}"

echo "=== expanded multilingual evaluation run (not a paper result) ==="
echo "task=sentiment dataset=afrisenti lang=$LANG seed=$SEED model=$MODEL"
nvidia-smi || true

srun python3 finetuning/sentiment/run_sentiment.py \
  --config configs/base.yaml \
  --model "$MODEL" \
  --dataset afrisenti \
  --dataset-config "$LANG" \
  --seed "$SEED" \
  --output "$OUT" \
  --override finetuning.save_total_limit=1
