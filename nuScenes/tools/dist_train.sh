#!/usr/bin/env bash
# =============================================================================
# Distributed training launcher for UniDriveVLA.
#
# Usage:
#   bash tools/dist_train.sh <config> <n_gpus> [extra_args...]
#
# Example (4 GPUs):
#   bash tools/dist_train.sh \
#       projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
#       4 \
#       --work-dir work_dirs/mini_stage1
# =============================================================================

set -euo pipefail

CONFIG=$1
GPUS=$2
shift 2

PORT=${PORT:-29500}

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}
export PYTHONPATH

echo "==> Launching distributed training"
echo "    Config : ${CONFIG}"
echo "    GPUs   : ${GPUS}"
echo "    Port   : ${PORT}"
echo "    Extra  : $*"
echo ""

python -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "$(dirname "$0")/train.py" \
    "${CONFIG}" \
    --launcher pytorch \
    "$@"
