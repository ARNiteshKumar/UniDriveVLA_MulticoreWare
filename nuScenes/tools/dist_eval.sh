#!/usr/bin/env bash
# =============================================================================
# Distributed evaluation launcher for UniDriveVLA.
#
# Usage:
#   bash tools/dist_eval.sh <config> <checkpoint> <n_gpus> [extra_args...]
#
# Example (4 GPUs):
#   bash tools/dist_eval.sh \
#       projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
#       work_dirs/mini_stage1/latest.pth \
#       4 \
#       --eval bbox map
# =============================================================================

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3

PORT=${PORT:-29501}

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}
export PYTHONPATH

echo "==> Launching distributed evaluation"
echo "    Config     : ${CONFIG}"
echo "    Checkpoint : ${CHECKPOINT}"
echo "    GPUs       : ${GPUS}"
echo "    Port       : ${PORT}"
echo "    Extra      : $*"
echo ""

python -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "$(dirname "$0")/test.py" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --launcher pytorch \
    "$@"
