#!/usr/bin/env bash
# Run the full LingoQA evaluation on 500 samples.
# Uses 4-bit bitsandbytes NF4 quantisation for T4 GPU compatibility.
#
# Phase 4 from the UniDriveVLA_nuScenes_mini_Colab checklist.
#
# Usage:
#   bash vqa_evaluation/LingoQA/run_lingoqa_mini.sh \
#       [MODEL_PATH] [IMAGE_ROOT] [OUTPUT_DIR]
#
# Defaults:
#   MODEL_PATH  path/to/UniDriveVLA_Nusc_Stage3
#   IMAGE_ROOT  data/LingoQA/images/val
#   OUTPUT_DIR  results/lingoqa

set -euo pipefail

MODEL_PATH="${1:-path/to/UniDriveVLA_Nusc_Stage3}"
IMAGE_ROOT="${2:-data/LingoQA/images/val}"
OUTPUT_DIR="${3:-results/lingoqa}"
PARQUET_PATH="vqa_evaluation/LingoQA/val.parquet"
PRED_CSV="${OUTPUT_DIR}/preds.csv"
NUM_GPUS=1
BATCH_SIZE=4

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo " UniDriveVLA — LingoQA Full Evaluation (500 samples)"
echo "============================================================"
echo " MODEL_PATH  : ${MODEL_PATH}"
echo " IMAGE_ROOT  : ${IMAGE_ROOT}"
echo " OUTPUT_DIR  : ${OUTPUT_DIR}"
echo ""

# ----------------------------------------------------------------
# Step 1: Inference (4-bit quantisation via bitsandbytes)
# ----------------------------------------------------------------
echo "[1/2] Running inference ..."
python vqa_evaluation/LingoQA/infer_qwenvl3.py \
    --model_path   "${MODEL_PATH}" \
    --image_root   "${IMAGE_ROOT}" \
    --parquet_path "${PARQUET_PATH}" \
    --output_path  "${PRED_CSV}" \
    --num_gpus     "${NUM_GPUS}" \
    --batch_size   "${BATCH_SIZE}"

echo ""
echo "[2/2] Evaluating predictions ..."
# ----------------------------------------------------------------
# Step 2: Score predictions against ground truth
# ----------------------------------------------------------------
python vqa_evaluation/LingoQA/evaluate.py \
    --pred_path  "${PRED_CSV}" \
    --gt_path    "${PARQUET_PATH}" \
    --output_dir "${OUTPUT_DIR}/eval"

echo ""
echo "============================================================"
echo " LingoQA evaluation complete."
echo " Results: ${OUTPUT_DIR}/eval/lingoqa_metrics.json"
echo "============================================================"
