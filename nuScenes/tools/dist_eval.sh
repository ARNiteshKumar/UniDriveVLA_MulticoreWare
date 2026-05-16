#!/usr/bin/env bash
# =============================================================================
# Distributed evaluation launcher for UniDriveVLA — nuScenes Mini
#
# Usage:
#   bash tools/dist_eval.sh <config> <checkpoint> <n_gpus> [extra_args...]
#
# Single-GPU example (T4 Colab / local):
#   cd nuScenes
#   bash tools/dist_eval.sh \
#       projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
#       /path/to/checkpoint.pth \
#       1 \
#       --eval bbox map motion planning \
#       --out work_dirs/eval/results.json
#
# Multi-GPU example (8 GPU cluster):
#   bash tools/dist_eval.sh \
#       projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
#       /path/to/checkpoint.pth \
#       8
# =============================================================================

set -euo pipefail

CONFIG="$1"
CHECKPOINT="$2"
GPUS="$3"
shift 3

PORT="${PORT:-29501}"
GPUS_PER_NODE=$(( GPUS < 8 ? GPUS : 8 ))
T=$(date +%m%d%H%M)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Work dir mirrors config path  (configs/ → work_dirs/)
WORK_DIR="${REPO_DIR}/$(echo "${CONFIG%.*}" | sed 's|projects/configs|work_dirs|')"
mkdir -p "${WORK_DIR}/logs"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-}"
export OCCWORLD_VAE_PATH="${OCCWORLD_VAE_PATH:-}"

echo "====================================================="
echo " UniDriveVLA — nuScenes Mini Evaluation"
echo "====================================================="
echo "  Config     : ${CONFIG}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  GPUs       : ${GPUS}"
echo "  Work dir   : ${WORK_DIR}"
echo "  Port       : ${PORT}"
echo "  Extra args : $*"
echo "-----------------------------------------------------"

torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/test.py" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    "$@" \
    2>&1 | tee "${WORK_DIR}/logs/eval_${T}.log"

echo ""
echo "Results saved to: ${WORK_DIR}"
echo "Log: ${WORK_DIR}/logs/eval_${T}.log"
