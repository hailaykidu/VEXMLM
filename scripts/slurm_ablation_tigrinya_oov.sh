#!/usr/bin/env bash
#SBATCH --job-name=vexmlm-ablation
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --array=0-4
#
# Table 5 ablation: Tigrinya NER OOV accuracy for the four arms, one array task
# per seed (42-46). Each arm is fine-tuned on Tigrinya NER from its own starting
# checkpoint, then scored with evaluation/run_ner_oov.py.
#
# The four arms differ only in the starting checkpoint:
#   arm1 xlmr_baseline        xlm-roberta-base                    (no expansion)
#   arm2 expansion_random     checkpoints/vexmlm-expanded-spm-random
#   arm3 expansion_mean       checkpoints/vexmlm-expanded-spm     (global_mean)
#   arm4 full_vexmlm          checkpoints/vexmlm-stage1-spm       (+ Stage 1 MLM)
#
# arm2 and arm3 are identical apart from embedding initialization, so their
# difference isolates mean-vs-random init. arm3 -> arm4 isolates continued
# pretraining. Fine-tuning hyperparameters come from configs/base.yaml and are
# not overridden, so all arms share one config hash.
#
# The OOV word set is defined by the baseline (xlm-roberta-base) against a FIXED
# reference tokenizer (checkpoints/vexmlm-expanded-spm), never against the arm
# being scored -- otherwise arm1, whose tokenizer IS the baseline, would yield an
# empty OOV set and each arm would be scored on a different word set.
#
#   sbatch scripts/slurm_ablation_tigrinya_oov.sh

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"
mkdir -p logs results/ablation

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"

ARMS=("xlmr_baseline" "expansion_random_init" "expansion_mean_init" "full_vexmlm")
MODELS=("xlm-roberta-base"
        "checkpoints/vexmlm-expanded-spm-random"
        "checkpoints/vexmlm-expanded-spm"
        "checkpoints/vexmlm-stage1-spm")
LABELS=("XLM-R baseline"
        "+ VocabExp (Random Init)"
        "+ VocabExp (Mean Init)"
        "+ Continued Pretraining")

echo "=== Ablation seed=$SEED node=$(hostname) $(date -Is) ==="
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from vexmlm import config as c
print("config_hash:", c.config_hash(c.load_config("configs/base.yaml")))
PY

FAILURES=""
for i in "${!ARMS[@]}"; do
    arm="${ARMS[$i]}"; model="${MODELS[$i]}"; label="${LABELS[$i]}"
    out="checkpoints/ablation/${arm}-tigrinya-seed${SEED}"
    echo ""
    echo "--- [$((i+1))/4] $arm (seed $SEED) from $model ---"

    if [ ! -d "$model" ] && [ "$model" != "xlm-roberta-base" ]; then
        echo "FAILED: $arm — missing checkpoint $model" >&2
        FAILURES="${FAILURES}${arm}(missing) "
        continue
    fi

    if python3 finetuning/ner/run_ner.py \
        --config configs/base.yaml --model "$model" \
        --dataset tigrinya_ner --local-path datasets/raw/tigrinya_ner \
        --seed "$SEED" --mode official --output "$out" \
        2>&1 | tee "logs/ablation-${arm}-${SEED}.log"; then
        echo "OK: fine-tune $arm seed=$SEED"
    else
        echo "FAILED: fine-tune $arm seed=$SEED" >&2
        FAILURES="${FAILURES}${arm}(train) "
        continue
    fi

    if python3 evaluation/run_ner_oov.py \
        --model "$out" --dataset tigrinya_ner \
        --local-path datasets/raw/tigrinya_ner \
        --reference-tokenizer checkpoints/vexmlm-expanded-spm \
        --arm "$arm" --label "$label" --seed "$SEED" \
        --out "results/ablation/${arm}-seed${SEED}.json" \
        2>&1 | tee -a "logs/ablation-${arm}-${SEED}.log"; then
        echo "OK: oov-eval $arm seed=$SEED"
    else
        echo "FAILED: oov-eval $arm seed=$SEED" >&2
        FAILURES="${FAILURES}${arm}(eval) "
    fi
done

echo ""
if [[ -n "$FAILURES" ]]; then
  echo "=== seed $SEED finished with failures: $FAILURES ==="
  exit 1
fi
echo "=== seed $SEED: all 4 arms OK $(date -Is) ==="
