#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ml-ner
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpunode08      # Blackwell RTX PRO 6000: no kernels in this torch build
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/ml-ner_%A_%a.out
#SBATCH --error=logs/slurm/ml-ner_%A_%a.err
#SBATCH --array=0-109
#
# Expanded multilingual evaluation — NER.
# 22 languages x 5 seeds = 110 array tasks:
#   - 20 MasakhaNER 2.0 languages
#   - Amharic  (MasakhaNER v1, local copy)
#   - Tigrinya (separate resource, local copy)
#
#   sbatch scripts/slurm_multilingual_ner.sh
#
# NOTE: MasakhaNER v1 and v2 are CC BY-NC 4.0 (non-commercial). Any model
# fine-tuned here inherits that restriction.

set -euo pipefail
mkdir -p logs/slurm

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

source /opt/conda/bin/activate /homes/neumann/teklehaymanot/.conda/envs/Gllm

# "dataset:config" pairs — masakhaner2 carries a language config; the two
# local sets do not.
TARGETS=(
  masakhaner2:bam masakhaner2:bbj masakhaner2:ewe masakhaner2:fon
  masakhaner2:hau masakhaner2:ibo masakhaner2:kin masakhaner2:lug
  masakhaner2:luo masakhaner2:mos masakhaner2:nya masakhaner2:pcm
  masakhaner2:sna masakhaner2:swa masakhaner2:tsn masakhaner2:twi
  masakhaner2:wol masakhaner2:xho masakhaner2:yor masakhaner2:zul
  masakhaner_amh: tigrinya_ner:
)
SEEDS=(42 43 44 45 46)

T_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
TARGET="${TARGETS[$T_IDX]}"
SEED="${SEEDS[$S_IDX]}"

DATASET="${TARGET%%:*}"
DS_CONFIG="${TARGET##*:}"

MODEL="${VEXMLM_MODEL:-checkpoints/vexmlm-stage1-spm}"
NAME="${DATASET}${DS_CONFIG:+-$DS_CONFIG}"
OUT="results/multilingual_evaluation/ner/${NAME}-seed${SEED}"

echo "=== expanded multilingual evaluation run (not a paper result) ==="
echo "task=ner dataset=$DATASET config=${DS_CONFIG:-<none>} seed=$SEED model=$MODEL"
nvidia-smi || true

ARGS=(--config configs/base.yaml --model "$MODEL" --dataset "$DATASET"
      --seed "$SEED" --output "$OUT"
      --override finetuning.save_total_limit=1)
[[ -n "$DS_CONFIG" ]] && ARGS+=(--dataset-config "$DS_CONFIG")
[[ -d "datasets/raw/$DATASET" ]] && ARGS+=(--local-path "datasets/raw/$DATASET")

srun python3 finetuning/ner/run_ner.py "${ARGS[@]}"
