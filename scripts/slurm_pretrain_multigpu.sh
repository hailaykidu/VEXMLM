#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ddp
#SBATCH --partition=standby
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddp_%j.out
#SBATCH --error=logs/ddp_%j.err
#
# Stage 1 on 4 GPUs via torchrun (DDP).
# Note: per_device_train_batch_size stays 32; the EFFECTIVE batch becomes
# 32 x 4 = 128. To hold the paper's effective batch, either drop
# per_device_train_batch_size to 8 or set gradient_accumulation_steps
# accordingly -- the run log records the effective value either way.

set -euo pipefail
mkdir -p logs
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

nvidia-smi || true
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node=4 \
  pretraining/run_mlm.py \
  --config configs/base.yaml \
  --model checkpoints/vexmlm-expanded \
  --amharic datasets/raw/amharic/*.txt \
  --tigrinya datasets/raw/tigrinya/*.txt \
  --output checkpoints/vexmlm-stage1 \
  --override pretraining.per_device_train_batch_size=8
