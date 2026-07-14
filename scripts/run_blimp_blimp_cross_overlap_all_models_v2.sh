#!/usr/bin/env bash
set -uo pipefail

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
PERCENTAGE="1.0"

SAVEDIR="english/cross-overlap/blackboxnlp_all_models_v2"
LOGDIR="logs/cross_overlap_blimp_blimp_all_models_v2"

mkdir -p "$SAVEDIR"
mkdir -p "$LOGDIR"

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

echo "Saving outputs to: $SAVEDIR"
echo "Saving logs to:    $LOGDIR"
echo "Using GPU:         $GPU"
echo

for MODEL in "${MODELS[@]}"; do
  MODEL_BASENAME="${MODEL##*/}"
  SAFE_MODEL=$(echo "$MODEL" | sed 's#[/:]#_#g')

  EXPECTED_OUT="$SAVEDIR/cross-overlap_${DATASET_1}_${DATASET_2}_${MODEL_BASENAME}_${PERCENTAGE}%.csv"
  LOG_FILE="$LOGDIR/${SAFE_MODEL}.log"

  echo "============================================================"
  echo "Model: $MODEL"
  echo "Expected output: $EXPECTED_OUT"
  echo "Log: $LOG_FILE"
  echo "============================================================"

  if [[ -f "$EXPECTED_OUT" ]]; then
    echo "Already exists, skipping: $EXPECTED_OUT"
    echo
    continue
  fi

  export LOC_CACHE="/local/scratch/ep757/syntax-units/cache_cross_overlap_${DATASET_1}_${DATASET_2}_${SAFE_MODEL}_v2"
  mkdir -p "$LOC_CACHE"

  if CUDA_VISIBLE_DEVICES="$GPU" python cross_overlap.py \
      --model-name "$MODEL" \
      --dataset-1 "$DATASET_1" \
      --dataset-2 "$DATASET_2" \
      --savedir "$SAVEDIR" \
      --percentage "$PERCENTAGE" \
      2>&1 | tee "$LOG_FILE"; then

    echo "FINISHED: $MODEL"
  else
    echo "FAILED: $MODEL"
    echo "Check log: $LOG_FILE"
  fi

  echo
done

echo "Done."
echo
echo "Created files:"
find "$SAVEDIR" -maxdepth 1 -name "cross-overlap_${DATASET_1}_${DATASET_2}_*.csv" -printf "%f\n" | sort

echo
echo "Number of CSV outputs:"
find "$SAVEDIR" -maxdepth 1 -name "cross-overlap_${DATASET_1}_${DATASET_2}_*.csv" | wc -l
