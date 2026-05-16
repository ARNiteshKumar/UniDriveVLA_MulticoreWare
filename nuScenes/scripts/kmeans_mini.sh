#!/usr/bin/env bash
# K-means anchor generation for nuScenes v1.0-mini.
# Produces four .npy anchor files used by UnifiedPerceptionDecoder:
#   data/kmeans/kmeans_det_900_mini.npy    — detection query anchors
#   data/kmeans/kmeans_map_100_mini.npy    — map element anchors
#   data/kmeans/kmeans_motion_6_mini.npy   — motion-prediction anchors
#   data/kmeans/kmeans_plan_6_mini.npy     — planning trajectory anchors
#
# Usage (from nuScenes/ directory):
#   bash scripts/kmeans_mini.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..":${PYTHONPATH:-}

INFO_PATH="data/infos/nuscenes_mini_temporal_train.pkl"
OUT_DIR="data/kmeans"

mkdir -p "${OUT_DIR}"

if [ ! -f "${INFO_PATH}" ]; then
    echo "[ERROR] Info file not found: ${INFO_PATH}"
    echo "        Run scripts/create_data_mini.sh first."
    exit 1
fi

echo "==> Computing k-means anchors from ${INFO_PATH} ..."

# Detection anchors — 900 queries for bounding-box prediction
echo "  [1/4] Detection anchors (k=900) ..."
python tools/kmeans/kmeans_generator.py \
    --info-path "${INFO_PATH}" \
    --task det \
    --num-clusters 900 \
    --output "${OUT_DIR}/kmeans_det_900_mini.npy"

# Map anchors — 100 queries for lane/crossing/boundary polylines
echo "  [2/4] Map anchors (k=100) ..."
python tools/kmeans/kmeans_generator.py \
    --info-path "${INFO_PATH}" \
    --task map \
    --num-clusters 100 \
    --output "${OUT_DIR}/kmeans_map_100_mini.npy"

# Motion anchors — 6 trajectory modes (multimodal future predictions)
echo "  [3/4] Motion anchors (k=6) ..."
python tools/kmeans/kmeans_generator.py \
    --info-path "${INFO_PATH}" \
    --task motion \
    --num-clusters 6 \
    --output "${OUT_DIR}/kmeans_motion_6_mini.npy"

# Planning anchors — 6 ego-motion trajectory modes
echo "  [4/4] Planning anchors (k=6) ..."
python tools/kmeans/kmeans_generator.py \
    --info-path "${INFO_PATH}" \
    --task plan \
    --num-clusters 6 \
    --output "${OUT_DIR}/kmeans_plan_6_mini.npy"

echo ""
echo "==> K-means anchors saved to ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"*.npy 2>/dev/null || echo "No .npy files found — check for errors above."
