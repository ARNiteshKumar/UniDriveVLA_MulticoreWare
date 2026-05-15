#!/usr/bin/env bash
# Data preparation script for nuScenes v1.0-mini.
# Produces:
#   data/infos/nuscenes_infos_mini_train.pkl
#   data/infos/nuscenes_infos_mini_val.pkl
#
# Usage (from nuScenes/ directory):
#   bash scripts/create_data_mini.sh [NUSCENES_ROOT] [CANBUS_ROOT]
#
# Defaults: data/nuscenes  data/nuscenes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..":${PYTHONPATH:-}

NUSCENES_ROOT="${1:-data/nuscenes}"
CANBUS_ROOT="${2:-data/nuscenes}"
OUT_DIR="data/infos"

mkdir -p "${OUT_DIR}"

echo "==> Creating nuScenes mini info files ..."
echo "    data_root : ${NUSCENES_ROOT}"
echo "    out_dir   : ${OUT_DIR}"

# Standard nuScenes converter (mini split)
python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path "${NUSCENES_ROOT}" \
    --canbus "${CANBUS_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --extra-tag nuscenes_mini \
    --version v1.0-mini

# VAD-format converter for driving QA co-training annotations
python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path "${NUSCENES_ROOT}" \
    --canbus "${CANBUS_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --extra-tag vad_nuscenes_mini \
    --version v1.0-mini

echo "==> Done.  Info files written to ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
