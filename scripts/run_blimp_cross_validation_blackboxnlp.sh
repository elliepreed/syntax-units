#!/usr/bin/env bash
set -euo pipefail

cd /local/scratch/ep757/syntax-units

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home/ep757/anaconda3/envs/syntax-head

export HF_HOME=/local/scratch/ep757/huggingface
export HUGGINGFACE_HUB_CACHE=/local/scratch/ep757/huggingface/hub
export PIP_CACHE_DIR=/local/scratch/ep757/pip-cache
export TMPDIR=/local/scratch/ep757/tmp

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"
mkdir -p english/cross-validation/blackboxnlp
mkdir -p logs

DATASET="blimp"
SAVEDIR="english/cross-validation/blackboxnlp"
PERCENTAGE="1.0"
NUM_FOLDS="2"
MODELS=(
  "deepseek-ai/deepseek-llm-7b-base"
  "google/gemma-3-4b-pt"
  "mistralai/Mistral-7B-v0.3"
)

GPU="${CUDA_VISIBLE_DEVICES:-0}"

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL=$(echo "$MODEL" | sed 's#[/:]#_#g')
  LOG_FILE="logs/cross_validation_${DATASET}_${SAFE_MODEL}.log"

  echo "============================================================"
  echo "Running cross-validation"
  echo "Model: $MODEL"
  echo "Dataset: $DATASET"
  echo "Savedir: $SAVEDIR"
  echo "GPU: $GPU"
  echo "Log: $LOG_FILE"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU" python cross_validation.py \
    --model-name "$MODEL" \
    --dataset "$DATASET" \
    --savedir "$SAVEDIR" \
    --percentage "$PERCENTAGE" \
    --num-folds "$NUM_FOLDS" \
    2>&1 | tee "$LOG_FILE"

  echo "Finished $MODEL"
  echo
done

echo "All BLiMP cross-validation runs finished."
echo "Outputs saved in: $SAVEDIR"
