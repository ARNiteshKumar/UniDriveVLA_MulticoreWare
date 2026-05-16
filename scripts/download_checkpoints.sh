#!/bin/bash
# ============================================================
# UniDriveVLA — Download model checkpoints & VQA datasets
# Usage:  bash scripts/download_checkpoints.sh [--stage STAGE]
#         STAGE: 1 | 2 | 3 (default: 3 — final stage)
# ============================================================
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-3}"
HF_REPO="xiaomi-research/UniDriveVLA"
CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

# ── model checkpoint ─────────────────────────────────────────
echo "==> Downloading UniDriveVLA nuScenes Stage${STAGE} checkpoint"
echo "    Source: https://huggingface.co/${HF_REPO}"

# huggingface-hub download
pip install --quiet huggingface_hub

python - <<EOF
from huggingface_hub import snapshot_download
import os

repo_id = "${HF_REPO}"
local_dir = "${CKPT_DIR}"

# Stage 3 is the final checkpoint used for evaluation
stage_map = {
    "1": "owl10/UniDriveVLA_Nusc_Base_Stage1",
    "2": "owl10/UniDriveVLA_Nusc_Base_Stage2",
    "3": "owl10/UniDriveVLA_Nusc_Base_Stage3",
}
stage = "${STAGE}"
subfolder = stage_map.get(stage, stage_map["3"])

try:
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(local_dir, f"stage{stage}"),
        ignore_patterns=["*.msgpack", "flax_*"],
    )
    print(f"Downloaded to: {path}")
except Exception as e:
    print(f"ERROR: {e}")
    print("Manual download: https://huggingface.co/${HF_REPO}")
    print(f"  Download the '{subfolder}' directory")
    print(f"  Place in: {local_dir}/stage{stage}/")
EOF

# ── VLM base model (Qwen3-VL-2B) ─────────────────────────────
echo ""
echo "==> Downloading Qwen3-VL-2B-Instruct (VLM base)"
python - <<EOF
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        local_dir="checkpoints/Qwen3-VL-2B-Instruct",
        ignore_patterns=["*.msgpack"],
    )
    print(f"VLM base downloaded to: {path}")
except Exception as e:
    print(f"WARNING: {e}")
    print("Manual: huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir checkpoints/Qwen3-VL-2B-Instruct")
EOF

# ── occvae checkpoint ─────────────────────────────────────────
echo ""
echo "==> Downloading OccWorld VAE checkpoint (occvae_latest.pth)"
python - <<EOF
from huggingface_hub import hf_hub_download
import os
try:
    path = hf_hub_download(
        repo_id="${HF_REPO}",
        filename="occvae_latest.pth",
        local_dir="checkpoints",
    )
    print(f"OccVAE checkpoint: {path}")
except Exception as e:
    print(f"WARNING: {e}")
    print("Manual: place occvae_latest.pth in checkpoints/")
EOF

# ── LingoQA val.parquet ───────────────────────────────────────
echo ""
echo "==> Downloading LingoQA val.parquet"
LINGOQA_DIR="vqa_evaluation/LingoQA"
if [[ ! -f "${LINGOQA_DIR}/val.parquet" ]]; then
    python - <<EOF
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id="wayveai/LingoQA",
        filename="val.parquet",
        repo_type="dataset",
        local_dir="${LINGOQA_DIR}",
    )
    print(f"LingoQA parquet: {path}")
except Exception as e:
    print(f"WARNING: {e}")
    print("Manual: download from https://huggingface.co/datasets/wayveai/LingoQA")
EOF
else
    echo "  LingoQA val.parquet already present"
fi

echo ""
echo "==> Checkpoint paths summary:"
find "$CKPT_DIR" -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null \
    | head -20 || echo "  (none yet)"

echo ""
echo "✓ Done. Export env vars before evaluation:"
echo "  export VLM_PRETRAINED_PATH=${REPO_ROOT}/checkpoints/Qwen3-VL-2B-Instruct"
echo "  export OCCWORLD_VAE_PATH=${REPO_ROOT}/checkpoints/occvae_latest.pth"
echo "  export STAGE3_CKPT=${REPO_ROOT}/checkpoints/stage3/<checkpoint>.pth"
echo ""
echo "  Next: bash scripts/prepare_data.sh"
