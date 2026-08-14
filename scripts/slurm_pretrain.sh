#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-stage1
#SBATCH --partition=standby
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --signal=B:USR1@300      # warn 5 min before preemption
#
# Stage 1: continued MLM pretraining.
# Requeues on preemption; the Trainer resumes from the last checkpoint.
#   sbatch scripts/slurm_pretrain.sh

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
nvidia-smi || true

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

srun python3 pretraining/run_mlm.py \
  --config configs/base.yaml \
  --model checkpoints/vexmlm-expanded \
  --amharic datasets/raw/amharic/*.txt \
  --tigrinya datasets/raw/tigrinya/*.txt \
  --output checkpoints/vexmlm-stage1

echo "done=$(date -Is)"
