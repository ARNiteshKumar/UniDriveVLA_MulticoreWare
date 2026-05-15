# Installation Guide

This guide covers setting up UniDriveVLA for nuScenes mini on a Linux machine with CUDA 12.1 and at least one GPU.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| OS          | Ubuntu 20.04 / 22.04 |
| Python      | 3.10    |
| CUDA        | 12.1    |
| GPU memory  | ≥ 24 GB (A100/RTX 3090+) |
| Conda       | Latest  |

## Step 1 — Create Conda Environment

```bash
conda create -n unidrivevla python=3.10 -y
conda activate unidrivevla
```

## Step 2 — Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1 True
```

## Step 3 — Install Base Requirements

```bash
cd /path/to/UniDriveVLA_MulticoreWare
pip install -r requirements/requirements_base.txt
```

Key packages installed:
- `transformers==4.57.1` — HuggingFace transformers (Qwen-VL)
- `peft==0.13.2` — LoRA adapter support
- `deepspeed==0.14.4` — ZeRO-1 for Stage 2 training
- `flash-attn==2.7.4.post1` — FlashAttention-2 for efficient VLM attention
- `timm==1.0.3` — Image model utilities

> **Note**: `flash-attn` requires CUDA headers to compile. Make sure `nvcc` is on `$PATH`.

## Step 4 — Install nuScenes Requirements

```bash
pip install -r requirements/requirements_nusc.txt
```

Key packages installed:
- `nuscenes-devkit>=1.1.11` — official nuScenes Python library
- `pyquaternion>=0.9.9` — quaternion math for 3D rotations
- `shapely>=2.0` — polygon geometry for map evaluation

## Step 5 — Install mmdet3d

```bash
pip install mmdet==2.28.0
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.5/index.html
pip install mmdet3d==1.4.0
```

> If the pre-built `mmcv-full` wheel is unavailable for your CUDA version, build from source:
> ```bash
> MMCV_WITH_OPS=1 pip install mmcv-full==1.7.2
> ```

## Step 6 — Install the Project Package

```bash
cd nuScenes
pip install -e .
```

This registers `projects.mmdet3d_plugin` as an importable package so that the
custom dataset, detector, and heads are discovered by mmdet3d's build system.

## Step 7 — Verify Installation

```bash
python -c "
import torch
from projects.mmdet3d_plugin.unidrivevla.detectors.unidrivevla import UniDriveVLA
from projects.mmdet3d_plugin.unidrivevla.dense_heads.unified_perception_decoder import UnifiedPerceptionDecoder
print('UniDriveVLA imports OK')
"
```

## Optional — Flash-Attention Pre-build

If `flash-attn` compilation takes too long, you can use pre-built wheels:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Or use the `FLASH_ATTENTION_SKIP_CUDA_BUILD=1` flag if your GPU supports xFormers instead.

## Common Issues

| Error | Fix |
|-------|-----|
| `ImportError: libGL.so.1` | `apt-get install libgl1-mesa-glx` |
| `mmcv/_ext not found` | Rebuild mmcv with `MMCV_WITH_OPS=1` |
| `flash-attn CUDA version mismatch` | Ensure `nvcc --version` matches PyTorch CUDA |
| `No module named 'nuscenes'` | `pip install nuscenes-devkit` |
