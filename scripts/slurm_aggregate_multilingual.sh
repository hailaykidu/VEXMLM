#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ml-agg
#SBATCH --partition=standby
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/ml-agg_%j.out
#SBATCH --error=logs/slurm/ml-agg_%j.err
#
# Aggregate the expanded multilingual evaluation. Submitted with a dependency
# on the evaluation arrays, so it runs once they finish.
#
#   sbatch --dependency=afterany:<sa>:<ner>:<qa>:<zs> scripts/slurm_aggregate_multilingual.sh

set -euo pipefail
mkdir -p logs/slurm
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# Conda environment. Override for your site:
#   export VEXMLM_CONDA_BASE=/opt/conda VEXMLM_CONDA_ENV=/path/to/env
source "${VEXMLM_CONDA_BASE:-/opt/conda}/bin/activate" "${VEXMLM_CONDA_ENV:?set VEXMLM_CONDA_ENV to your conda env path}"

python3 evaluation/aggregate_multilingual.py \
  --runs results/multilingual_evaluation \
  --out  results/multilingual_evaluation/aggregated
