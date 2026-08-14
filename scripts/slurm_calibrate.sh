#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-calib
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/calib_%j.out
#SBATCH --error=logs/calib_%j.err
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

# Mandatory GPU gate -- refuses to continue without CUDA.
python3 scripts/gpu_probe.py || { echo "GPU verification FAILED; aborting" >&2; exit 2; }

srun python3 pretraining/run_mlm.py \
  --config configs/base.yaml \
  --model checkpoints/vexmlm-expanded \
  --amharic datasets/processed/amharic.train.txt \
  --tigrinya datasets/processed/tigrinya.train.txt \
  --block-chunk \
  --mode official \
  --output checkpoints/calib-stage1 \
  --max-steps 30 --no-tracking --override pretraining.save_total_limit=1

echo "done=$(date -Is)"
