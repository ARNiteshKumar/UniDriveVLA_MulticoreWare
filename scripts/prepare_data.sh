#!/bin/bash
# ============================================================
# UniDriveVLA — nuScenes Mini Data Preparation
# Usage:  bash scripts/prepare_data.sh [/path/to/nuscenes]
#
# Prerequisites:
#   • nuScenes v1.0-mini downloaded (https://www.nuscenes.org)
#   • CAN bus expansion downloaded (same page)
#   • Environment set up via setup_env.sh
# ============================================================
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUSCENES_ROOT="${1:-}"

# ── locate nuScenes data ──────────────────────────────────────
NUSCENES_LINK="${REPO_ROOT}/nuScenes/data/nuscenes"
if [[ -n "$NUSCENES_ROOT" ]]; then
    echo "==> Linking nuScenes from: $NUSCENES_ROOT"
    mkdir -p "${REPO_ROOT}/nuScenes/data"
    ln -sfn "$NUSCENES_ROOT" "$NUSCENES_LINK"
elif [[ ! -d "$NUSCENES_LINK" ]]; then
    echo "ERROR: nuScenes data not found."
    echo "  Option A: bash scripts/prepare_data.sh /path/to/nuscenes"
    echo "  Option B: ln -s /path/to/nuscenes ${NUSCENES_LINK}"
    exit 1
fi
echo "  nuScenes path: $(readlink -f "$NUSCENES_LINK")"

# ── verify v1.0-mini layout ───────────────────────────────────
for required in "v1.0-mini" "samples" "sweeps" "maps"; do
    if [[ ! -d "${NUSCENES_LINK}/${required}" ]]; then
        echo "ERROR: Missing '${required}' under nuScenes root."
        echo "  Make sure you extracted v1.0-mini.zip and can_bus.zip into the same folder."
        exit 1
    fi
done
echo "  nuScenes directory layout: OK"

# ── run data conversion ───────────────────────────────────────
cd "${REPO_ROOT}/nuScenes"
export PYTHONPATH="${REPO_ROOT}/nuScenes:${REPO_ROOT}:$PYTHONPATH"
mkdir -p data/infos data/kmeans

echo ""
echo "==> Step 1/4: Generate nuscenes_mini temporal info PKLs"
python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes_mini \
    --version v1.0-mini

echo ""
echo "==> Step 2/4: Generate VAD nuscenes_mini info PKLs"
python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag vad_nuscenes_mini \
    --version v1.0-mini

echo ""
echo "==> Step 3/4: Generate K-means anchors (det/map/motion/plan)"
python tools/kmeans/kmeans_generator.py \
    --info-path data/infos/nuscenes_mini_temporal_train.pkl \
    --out-dir data/kmeans/ \
    --version mini

echo ""
echo "==> Step 4/4: Verifying output files"
for f in \
    "data/infos/nuscenes_mini_temporal_train.pkl" \
    "data/infos/nuscenes_mini_temporal_val.pkl" \
    "data/infos/vad_nuscenes_mini_temporal_train.pkl" \
    "data/infos/vad_nuscenes_mini_temporal_val.pkl" \
    "data/kmeans/kmeans_det_900_mini.npy" \
    "data/kmeans/kmeans_map_100_mini.npy" \
    "data/kmeans/kmeans_motion_6_mini.npy" \
    "data/kmeans/kmeans_plan_6_mini.npy"; do
    if [[ -f "$f" ]]; then
        echo "  ✓ $f ($(du -h "$f" | cut -f1))"
    else
        echo "  ✗ MISSING: $f"
    fi
done

cd "$REPO_ROOT"
echo ""
echo "✓ Data preparation complete."
echo "  Next: bash scripts/run_benchmarks.sh"
