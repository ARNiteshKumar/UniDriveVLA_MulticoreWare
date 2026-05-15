#!/usr/bin/env bash
# Generate k-means anchor files for detection, map, motion, and planning.
# These are used by the instance query banks inside UnifiedPerceptionDecoder.
#
# Usage (from nuScenes/ directory):
#   bash scripts/kmeans.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..":${PYTHONPATH:-}

ANN_FILE="data/infos/nuscenes_infos_mini_train.pkl"
OUT_DIR="data/kmeans"

mkdir -p "${OUT_DIR}"

echo "==> Computing k-means anchors from ${ANN_FILE} ..."

python tools/kmeans/kmeans_anchor.py \
    --ann-file "${ANN_FILE}" \
    --out-dir "${OUT_DIR}" \
    --det-k 900 \
    --map-k 100 \
    --motion-k 6 \
    --plan-k 6

echo "==> Anchor files written to ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
