#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-gpuprobe
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpuprobe_%j.out
#SBATCH --error=logs/gpuprobe_%j.err
#
# Mandatory GPU verification gate. Must pass before Stage 1.
#   sbatch scripts/slurm_gpu_probe.sh

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"
mkdir -p logs

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} $(date -Is)"
nvidia-smi || true
python3 scripts/gpu_probe.py
