#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ml-zs
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpunode08      # Blackwell RTX PRO 6000: no kernels in this torch build
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/ml-zs_%A_%a.out
#SBATCH --error=logs/slurm/ml-zs_%A_%a.err
#SBATCH --array=0-9
#
# Expanded multilingual evaluation — zero-shot sentiment transfer.
#
# AfriSenti ships `orm` (Oromo) and `tir` (Tigrinya) with dev and test splits
# but NO training split (SemEval-2023 Task 12, Subtask C). They are therefore
# evaluated zero-shot: a model fine-tuned on Amharic is applied directly to the
# target-language test set. Amharic is the transfer source because it is the
# closest Ge'ez-script language with training data.
#
# 2 languages x 5 seeds = 10 array tasks, one per source-model seed.
#
#   sbatch scripts/slurm_zeroshot_sa.sh
#
# These runs perform NO training. They require the corresponding Amharic
# fine-tuned checkpoint to exist, so submit this with a dependency on the
# sentiment array.

set -euo pipefail
mkdir -p logs/slurm

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

# Conda environment. Override for your site:
#   export VEXMLM_CONDA_BASE=/opt/conda VEXMLM_CONDA_ENV=/path/to/env
source "${VEXMLM_CONDA_BASE:-/opt/conda}/bin/activate" "${VEXMLM_CONDA_ENV:?set VEXMLM_CONDA_ENV to your conda env path}"

LANGS=(orm tir)
SEEDS=(42 43 44 45 46)

L_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
LANG="${LANGS[$L_IDX]}"
SEED="${SEEDS[$S_IDX]}"

SRC="results/multilingual_evaluation/sentiment/afrisenti-amh-seed${SEED}"
OUT="results/multilingual_evaluation/sentiment_zeroshot/afrisenti-${LANG}-from-amh-seed${SEED}"

echo "=== expanded multilingual evaluation run (not a paper result) ==="
echo "task=sentiment-zeroshot target=$LANG source=amh seed=$SEED"

if [[ ! -d "$SRC" ]]; then
  echo "source checkpoint missing: $SRC" >&2
  exit 2
fi

srun python3 evaluation/run_zeroshot_sentiment.py \
  --model "$SRC" \
  --dataset afrisenti \
  --dataset-config "$LANG" \
  --split test \
  --source-language amh \
  --seed "$SEED" \
  --output "$OUT"
