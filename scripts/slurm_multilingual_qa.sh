#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ml-qa
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpunode08      # Blackwell RTX PRO 6000: no kernels in this torch build
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/ml-qa_%A_%a.out
#SBATCH --error=logs/slurm/ml-qa_%A_%a.err
#SBATCH --array=0-9
#
# Expanded multilingual evaluation — QA, unchanged in scope.
# AmQA (amh) and TIGQA (tir), 2 datasets x 5 seeds = 10 array tasks.
#
# The QA task set is deliberately IDENTICAL to the release: same datasets, same
# seeds, same hyperparameters. These runs are re-executed here only so the
# multilingual report carries QA numbers produced by the same pipeline pass.
# They do not replace the published QA results.
#
#   sbatch scripts/slurm_multilingual_qa.sh

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

DATASETS=(amqa tigqa)
SEEDS=(42 43 44 45 46)

D_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
DATASET="${DATASETS[$D_IDX]}"
SEED="${SEEDS[$S_IDX]}"

MODEL="${VEXMLM_MODEL:-checkpoints/vexmlm-stage1-spm}"
OUT="results/multilingual_evaluation/qa/${DATASET}-seed${SEED}"

echo "=== expanded multilingual evaluation run (not a paper result) ==="
echo "task=qa dataset=$DATASET seed=$SEED model=$MODEL"
nvidia-smi || true

ARGS=(--config configs/base.yaml --model "$MODEL" --dataset "$DATASET"
      --seed "$SEED" --output "$OUT"
      --override finetuning.save_total_limit=1)
[[ -d "datasets/raw/$DATASET" ]] && ARGS+=(--local-path "datasets/raw/$DATASET")

srun python3 finetuning/qa/run_qa.py "${ARGS[@]}"
