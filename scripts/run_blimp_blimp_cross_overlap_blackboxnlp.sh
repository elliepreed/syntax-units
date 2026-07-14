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

DATASET_1="blimp"
DATASET_2="blimp"
SAVEDIR="english/cross-overlap/blackboxnlp"
PERCENTAGE="1.0"

mkdir -p "$SAVEDIR"
mkdir -p logs

MODELS=(
  "openai-community/gpt2-xl"
  "meta-llama/Llama-3.2-3B"
  "tiiuae/Falcon3-3B-Base"
  "microsoft/Phi-4-mini-instruct"
  "deepseek-ai/deepseek-llm-7b-base"
  "google/gemma-3-4b-pt"
  "mistralai/Mistral-7B-v0.3"
)

GPU="${CUDA_VISIBLE_DEVICES:-0}"

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL=$(echo "$MODEL" | sed 's#[/:]#_#g')
  LOG_FILE="logs/cross_overlap_${DATASET_1}_${DATASET_2}_${SAFE_MODEL}.log"

  export LOC_CACHE="/local/scratch/ep757/syntax-units/cache_cross_overlap_${DATASET_1}_${DATASET_2}_${SAFE_MODEL}"
  mkdir -p "$LOC_CACHE"

  echo "============================================================"
  echo "Running BLiMP × BLiMP cross-overlap"
  echo "Model: $MODEL"
  echo "Dataset 1: $DATASET_1"
  echo "Dataset 2: $DATASET_2"
  echo "Savedir: $SAVEDIR"
  echo "GPU: $GPU"
  echo "LOC_CACHE: $LOC_CACHE"
  echo "Log: $LOG_FILE"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU" python cross_overlap.py \
    --model-name "$MODEL" \
    --dataset-1 "$DATASET_1" \
    --dataset-2 "$DATASET_2" \
    --savedir "$SAVEDIR" \
    --percentage "$PERCENTAGE" \
    2>&1 | tee "$LOG_FILE"

  echo "Finished $MODEL"
  echo
done

echo "All BLiMP × BLiMP cross-overlap runs finished."
echo "Outputs saved in: $SAVEDIR"
