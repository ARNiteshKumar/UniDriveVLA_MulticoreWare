#!/bin/bash
# ============================================================
# UniDriveVLA — nuScenes Mini  |  Full Environment Setup
# Run once after cloning:  bash scripts/setup_env.sh
# Tested: Python 3.9, CUDA 12.1/12.8, Ubuntu 20.04/22.04
# ============================================================
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── detect CUDA ──────────────────────────────────────────────
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
if [[ -z "$CUDA_VER" ]]; then
    echo "WARNING: nvcc not found; defaulting to CUDA 12.1"
    CUDA_VER="12.1"
fi
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
if   [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" -ge "8" ]]; then CUDA_TAG="cu128"
elif [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" -ge "4" ]]; then CUDA_TAG="cu124"
elif [[ "$CUDA_MAJOR" == "12" ]];                            then CUDA_TAG="cu121"
elif [[ "$CUDA_MAJOR" == "11" ]];                            then CUDA_TAG="cu118"
else CUDA_TAG="cu121"; fi
echo "==> Detected CUDA ${CUDA_VER}, using PyTorch wheel tag: ${CUDA_TAG}"

# ── PyTorch 2.5.1 ────────────────────────────────────────────
echo "==> Installing PyTorch 2.5.1 + torchvision 0.20.1 (${CUDA_TAG})"
pip install --quiet \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ── transformers 4.57.1 + Qwen3-VL patch ─────────────────────
echo "==> Installing transformers 4.57.1"
pip install --quiet transformers==4.57.1

TRANSFORMERS_DIR=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
if [[ -d "qwenvl3/transformers_replace/models" ]]; then
    echo "==> Applying Qwen3-VL patch to ${TRANSFORMERS_DIR}/models"
    cp -r qwenvl3/transformers_replace/models "${TRANSFORMERS_DIR}/"
else
    echo "WARNING: qwenvl3/transformers_replace/models not found — skipping Qwen3-VL patch"
    echo "         Clone the full UniDriveVLA repo: https://github.com/xiaomi-research/unidrivevla"
fi

# ── mmcv 1.7.2 (pre-built wheel) ─────────────────────────────
echo "==> Installing mmcv-full 1.7.2"
TORCH_VER="2.5"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}/mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl"
pip install --quiet "$MMCV_WHEEL" 2>/dev/null || \
    pip install --quiet mmcv-full==1.7.2 -f "https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}/index.html"

# ── mmdetection3d 1.0.0rc6 (from third_party) ────────────────
if [[ -d "third_party/mmdetection3d-1.0.0rc6" ]]; then
    echo "==> Installing mmdetection3d 1.0.0rc6 from third_party/"
    pip install --quiet -e "third_party/mmdetection3d-1.0.0rc6" --no-build-isolation
else
    echo "==> Installing mmdetection3d from git (fallback)"
    pip install --quiet "mmdet3d @ git+https://github.com/open-mmlab/mmdetection3d.git@v1.0.0rc6"
fi

# ── deep learning extras ──────────────────────────────────────
echo "==> Installing deepspeed, peft, timm, qwen_vl_utils"
pip install --quiet \
    deepspeed==0.14.4 \
    peft \
    timm \
    qwen_vl_utils \
    bitsandbytes \
    accelerate

# ── flash-attention (optional, skip on T4) ───────────────────
echo "==> Attempting flash-attn install (skip failure OK on T4)"
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || \
    echo "   flash-attn not available — will use eager attention fallback"

# ── vllm (for VQA evaluation) ────────────────────────────────
echo "==> Installing vllm + ray for VQA evaluation"
pip install --quiet "vllm>=0.4.0" ray datasets click pyarrow

# ── nuScenes requirements ─────────────────────────────────────
echo "==> Installing nuScenes requirements"
pip install --quiet -r requirements/requirements_nusc.txt

# ── build custom CUDA ops ────────────────────────────────────
if [[ -d "nuScenes/projects/mmdet3d_plugin/ops" ]]; then
    echo "==> Building custom CUDA ops (deformable_aggregation_ext)"
    cd nuScenes/projects/mmdet3d_plugin/ops
    python setup.py build_ext --inplace
    cd "$REPO_ROOT"
else
    echo "WARNING: custom ops dir not found — evaluation may fail"
fi

echo ""
echo "✓ Environment setup complete."
echo "  Next: bash scripts/download_checkpoints.sh"
