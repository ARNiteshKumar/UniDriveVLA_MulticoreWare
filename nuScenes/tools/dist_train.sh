#!/usr/bin/env bash
# =============================================================================
# Distributed training launcher for UniDriveVLA — nuScenes Mini
#
# Stage 1 (perception only, no VLM):
#   cd nuScenes
#   bash tools/dist_train.sh \
#       projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
#       1                   # 1 GPU on T4; use 8 for cluster
#
# Stage 2 (VLM integration):
#   export VLM_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct
#   export OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
#   export STAGE1_CHECKPOINT=/path/to/stage1_checkpoint.pth
#   bash tools/dist_train.sh \
#       projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
#       1
# =============================================================================

set -euo pipefail

CONFIG="$1"
GPUS="$2"
shift 2

PORT="${PORT:-29500}"
GPUS_PER_NODE=$(( GPUS < 8 ? GPUS : 8 ))
T=$(date +%m%d%H%M)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORK_DIR="${REPO_DIR}/$(echo "${CONFIG%.*}" | sed 's|projects/configs|work_dirs|')"
mkdir -p "${WORK_DIR}/logs"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-}"
export OCCWORLD_VAE_PATH="${OCCWORLD_VAE_PATH:-}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"

echo "====================================================="
echo " UniDriveVLA — nuScenes Mini Training"
echo "====================================================="
echo "  Config     : ${CONFIG}"
echo "  GPUs       : ${GPUS}"
echo "  Work dir   : ${WORK_DIR}"
echo "  Port       : ${PORT}"
echo "  Extra args : $*"
echo "-----------------------------------------------------"

torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/train.py" \
    "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    "$@" \
    2>&1 | tee "${WORK_DIR}/logs/train_${T}.log"

echo ""
echo "Checkpoints saved to: ${WORK_DIR}"
