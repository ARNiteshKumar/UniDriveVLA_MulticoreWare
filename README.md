# UniDriveVLA — nuScenes Mini Adaptation

A reproduction of the [UniDriveVLA](https://github.com/xiaomi-research/unidrivevla) autonomous driving framework adapted for the **nuScenes mini** dataset (`v1.0-mini`, 10 scenes, ~323 samples). This repository is intended for rapid development, testing, and CI evaluation without requiring the full nuScenes dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [nuScenes Mini Notes](#nuscenes-mini-notes)
9. [Citation](#citation)

---

## Project Overview

UniDriveVLA unifies autonomous driving perception and language understanding into a single framework by coupling a **BEV-based perception backbone** (BEVFormer-style) with a **Vision-Language Model (VLM)**. The framework supports:

- **3D Object Detection** (10 nuScenes classes)
- **Online Map Prediction** (3 map element classes)
- **Ego Motion / Planning** (waypoint prediction)
- **Driving VQA** (DriveLM, DriveBench, LingoQA benchmarks)
- **Closed-loop Evaluation** (Bench2Drive / CARLA)

This repository adapts the original full-dataset config to the compact **nuScenes mini** split, making it possible to:

- Run data preparation in minutes
- Complete a training stage in hours on a single GPU
- Execute all evaluation pipelines in CI without large storage

---

## Architecture

```
Camera Images (6 cams, 450×800)
        │
        ▼
  Image Backbone (ResNet-50)
        │
        ▼
  BEV Encoder (BEVFormer-tiny style)
  - 3 encoder layers
  - BEV grid: 50×50
  - Single-scale features (C5)
        │
        ▼
  UnifiedPerceptionDecoder
  ┌─────────────────────────────────┐
  │  Detection head (10 classes)    │
  │  Map head (3 classes)           │
  │  Ego-status head               │
  │  Motion / Planning head         │
  └─────────────────────────────────┘
        │               │
        ▼               ▼
   Stage 1         Stage 2
  (Perception)    (+ VLM Integration)
                   Qwen-VL / InternVL
                        │
                        ▼
                  Driving QA Output
```

### UnifiedPerceptionDecoder

`UnifiedPerceptionDecoder` is the core module that processes BEV features and produces multi-task outputs:

- `forward_stage1`: runs perception-only pass through transformer decoder layers
- `forward_stage2`: merges VLM token embeddings into perception features
- `loss`: computes combined multi-task loss (detection + map + planning)
- `post_process`: converts raw outputs to nuScenes-format predictions

### VLM Integration

The VLM (Qwen-VL or similar) receives:
1. Front-camera image crops
2. BEV tokens projected to language space
3. Task prompt (question / instruction)

Its output tokens are injected back into `forward_stage2` to condition the planning and QA heads.

---

## Installation

See [docs/installation.md](docs/installation.md) for the full step-by-step guide.

```bash
# 1. Clone repository
git clone <this-repo>
cd UniDriveVLA_MulticoreWare

# 2. Create conda environment
conda create -n unidrivevla python=3.10 -y
conda activate unidrivevla

# 3. Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Install base requirements
pip install -r requirements/requirements_base.txt

# 5. Install nuScenes requirements
pip install -r requirements/requirements_nusc.txt

# 6. Install mmdet3d (v1.4)
pip install mmdet==2.28.0 mmcv-full==1.7.2
pip install mmdet3d==1.4.0

# 7. Install project package
cd nuScenes
pip install -e .
```

---

## Data Preparation

See [docs/data_preparation.md](docs/data_preparation.md) for details.

```
data/
└── nuscenes/
    ├── v1.0-mini/
    │   ├── scene.json
    │   ├── sample.json
    │   └── ...
    ├── samples/
    └── sweeps/
```

Download nuScenes mini from [https://www.nuscenes.org/download](https://www.nuscenes.org/download) (Free account required, ~4 GB).

```bash
# Create info pkl files for mini
cd nuScenes
bash scripts/create_data_mini.sh
```

This creates:
- `data/infos/nuscenes_infos_mini_train.pkl`
- `data/infos/nuscenes_infos_mini_val.pkl`

---

## Training

### Stage 1 — Perception Only

```bash
cd nuScenes

# Single GPU
python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --work-dir work_dirs/mini_stage1

# Multi-GPU (e.g., 4 GPUs)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    4 \
    --work-dir work_dirs/mini_stage1
```

### Stage 2 — Perception + VLM

```bash
cd nuScenes

python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --work-dir work_dirs/mini_stage2 \
    --load-from work_dirs/mini_stage1/latest.pth
```

---

## Evaluation

### nuScenes Detection / Mapping / Planning

```bash
cd nuScenes

# Single GPU evaluation
python tools/test.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    work_dirs/mini_stage1/latest.pth \
    --eval bbox map

# Multi-GPU evaluation
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    work_dirs/mini_stage1/latest.pth \
    4
```

Full nuScenes metric suite:

```bash
python tools/evaluation/nuscenes_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --eval-set mini_val
```

### Planning Metrics (L2, Collision Rate)

```bash
python tools/evaluation/planning_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --config projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py
```

### DriveLM Evaluation

```bash
python vqa_evaluation/DriveLM/eval_drivelm.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --data-root data/drivelm \
    --split val \
    --output-dir work_dirs/drivelm_results
```

### DriveBench Evaluation

```bash
python vqa_evaluation/DriveBench/eval_drivebench.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --data-root data/drivebench \
    --output-dir work_dirs/drivebench_results
```

### LingoQA Evaluation

```bash
python vqa_evaluation/LingoQA/eval_lingoqa.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --data-root data/lingoqa \
    --output-dir work_dirs/lingoqa_results
```

### Bench2Drive (Requires CARLA)

See [docs/bench2drive_setup.md](docs/bench2drive_setup.md). This requires a local CARLA 0.9.15 installation and cannot run in standard CI.

```bash
# Example (after CARLA setup)
python tools/bench2drive_eval.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --carla-host localhost \
    --carla-port 2000
```

---

## Results

> **Note**: The table below shows placeholder metric names. Actual values are populated after running evaluation on the nuScenes mini validation set.

### Detection (Stage 1, nuScenes mini val)

| Model | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|-------|-----|-----|------|------|------|------|------|
| UniDriveVLA-mini (Stage1) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Online Mapping (Stage 1, nuScenes mini val)

| Model | mIoU (divider) | mIoU (ped_crossing) | mIoU (boundary) | mIoU (avg) |
|-------|---------------|---------------------|-----------------|------------|
| UniDriveVLA-mini (Stage1) | TBD | TBD | TBD | TBD |

### Planning (Stage 1, nuScenes mini val)

| Model | L2 (1s) | L2 (2s) | L2 (3s) | Col. Rate (1s) | Col. Rate (2s) | Col. Rate (3s) |
|-------|---------|---------|---------|----------------|----------------|----------------|
| UniDriveVLA-mini (Stage1) | TBD | TBD | TBD | TBD | TBD | TBD |

### VQA (Stage 2)

| Model | DriveLM Acc | DriveLM BLEU-4 | DriveLM CIDEr | LingoQA Score | DriveBench Score |
|-------|-------------|----------------|---------------|---------------|------------------|
| UniDriveVLA-mini (Stage2) | TBD | TBD | TBD | TBD | TBD |

---

## nuScenes Mini Notes

The nuScenes mini split differs from the full dataset in several ways that affect this configuration:

| Parameter | Full nuScenes | nuScenes Mini |
|-----------|--------------|---------------|
| Version string | `v1.0` | `v1.0-mini` |
| Scenes (train/val) | 700/150 | 7/3 |
| Samples | ~28,000 | ~323 |
| Image size | 960×544 | 800×450 |
| BEV grid | 200×200 | 50×50 |
| Queue length | 4 | 2 |
| Batch size / GPU | 2–4 | 1 |
| BEVFormer enc. layers | 6 | 3 |
| Feature scales | Multi-scale | Single (C5) |
| Stage 1 epochs | 24 | 24 |
| Stage 2 epochs | 15 | 15 |
| Info file (train) | `nuscenes_infos_train.pkl` | `nuscenes_infos_mini_train.pkl` |
| Info file (val) | `nuscenes_infos_val.pkl` | `nuscenes_infos_mini_val.pkl` |

Because the mini split has only ~323 samples, evaluation metrics will have high variance. Results should be interpreted as a sanity check rather than a performance benchmark.

---

## Citation

If you use this work, please cite the original UniDriveVLA paper:

```bibtex
@article{unidrivevla2024,
  title={UniDriveVLA: Unified Autonomous Driving with Vision-Language-Action Model},
  author={Xiaomi Research},
  journal={arXiv},
  year={2024}
}
```

Please also cite the nuScenes dataset:

```bibtex
@article{caesar2020nuscenes,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and others},
  journal={CVPR},
  year={2020}
}
```
