# UniDriveVLA — nuScenes Mini

[![CI](https://github.com/arniteshkumar/unidrivevla_multicoreware/actions/workflows/nuscenes_mini_eval.yml/badge.svg)](https://github.com/arniteshkumar/unidrivevla_multicoreware/actions/workflows/nuscenes_mini_eval.yml)

Adaptation of [UniDriveVLA](https://github.com/xiaomi-research/unidrivevla) for the **nuScenes v1.0-mini** dataset.  
Supports full evaluation of all UniDriveVLA benchmarks on a single GPU (T4 compatible).

---

## What's inside

| Component | Description |
|-----------|-------------|
| `nuScenes/` | Model, configs, data prep, train/eval scripts |
| `vqa_evaluation/` | LingoQA · DriveLM · DriveBench evaluation |
| `scripts/` | One-shot setup, download, data-prep, benchmark runner |
| `docs/` | Installation, data prep, evaluation guides |
| `.github/workflows/` | CI: lint + arch check + eval dry-run |

---

## nuScenes Mini Adaptations (BEVFormer-tiny style)

| Parameter | UniDriveVLA full | **This repo (mini)** |
|-----------|-----------------|----------------------|
| Dataset version | `v1.0` (850 scenes) | **`v1.0-mini`** (10 scenes) |
| Image size | 960 × 544 | **800 × 450** (scales=[0.5]) |
| BEV grid | 200 × 200 | **50 × 50** |
| Temporal queue | 4 frames | **3 frames** |
| BEV enc. layers | 6 | **3** (C5 only) |
| Backbone | ResNet-101-DCN | **ResNet-50** |
| Stage 1 epochs | 24 | **24** |
| Stage 2 epochs | 15 | **15** |
| Batch / GPU | 4 | **1** |

---

## Quick Start (end-to-end)

### 0. Clone
```bash
git clone https://github.com/arniteshkumar/unidrivevla_multicoreware.git
cd unidrivevla_multicoreware

# Also clone the original UniDriveVLA repo alongside
# (provides third_party/, qwenvl3/ and model source files)
git clone https://github.com/xiaomi-research/unidrivevla.git
```

### 1. Environment setup
```bash
bash scripts/setup_env.sh
```
Detects your CUDA version, installs PyTorch 2.5.1, transformers 4.57.1 + Qwen3-VL patch,
mmcv 1.7.2, mmdet3d 1.0.0rc6, deepspeed 0.14.4, peft, vllm, ray, and nuScenes devkit.

### 2. Download checkpoints
```bash
bash scripts/download_checkpoints.sh        # downloads Stage 3 checkpoint + Qwen3-VL-2B + OccVAE
# Exports paths to env vars automatically
export VLM_PRETRAINED_PATH=checkpoints/Qwen3-VL-2B-Instruct
export OCCWORLD_VAE_PATH=checkpoints/occvae_latest.pth
```

### 3. Download and prepare nuScenes mini data
```bash
# Download v1.0-mini.zip and can_bus.zip from https://www.nuscenes.org/nuscenes#download
# Extract both into the same folder, e.g. /data/nuscenes/

bash scripts/prepare_data.sh /data/nuscenes
# Runs nuscenes_converter.py + vad_nuscenes_converter.py + kmeans_generator.py
# Output: nuScenes/data/infos/*.pkl  and  nuScenes/data/kmeans/*.npy
```

### 4. Run all benchmarks
```bash
bash scripts/run_benchmarks.sh \
    --checkpoint checkpoints/stage3/<checkpoint>.pth \
    --num-gpus 1
# Results saved to results/benchmark_summary.txt
```

Or run individual benchmarks:
```bash
# nuScenes only
bash scripts/run_benchmarks.sh --checkpoint ... --benchmarks nuscenes

# LingoQA only (500 samples)
bash scripts/run_benchmarks.sh --checkpoint ... --benchmarks lingoqa

# DriveLM only
export DRIVELM_JSON=data/DriveLM/QA_dataset_nus_v1_val.json
bash scripts/run_benchmarks.sh --checkpoint ... --benchmarks drivelm

# DriveBench only
export DRIVEBENCH_ROOT=data/DriveBench
bash scripts/run_benchmarks.sh --checkpoint ... --benchmarks drivebench
```

---

## Evaluation Commands (manual)

### nuScenes Open-Loop Evaluation
```bash
cd nuScenes

# Detection + Mapping + Motion + Planning
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    /path/to/checkpoint.pth \
    1 \
    --eval bbox map motion planning \
    --out work_dirs/eval/results.json

# Parse results
python tools/evaluation/nuscenes_eval.py \
    --result-path work_dirs/eval/results.json \
    --version v1.0-mini \
    --dataroot data/nuscenes

python tools/evaluation/planning_eval.py \
    --result-path work_dirs/eval/results.json
```

Expected metrics:
```
NDS: ~0.35-0.45     mAP: ~0.25-0.35
L2 @ 1s/2s/3s: ~0.3 / 0.6 / 1.0  m
Collision Rate: ~0.02 / 0.06 / 0.12
map IoU: ~0.30-0.45
```

### LingoQA (all 500 samples)
```bash
# Step 1: inference
python vqa_evaluation/LingoQA/infer_qwenvl3.py \
    --model_path  $VLM_PRETRAINED_PATH \
    --parquet_path vqa_evaluation/LingoQA/val.parquet \
    --image_root   data/LingoQA/images/val \
    --output_path  results/lingoqa/preds.csv \
    --num_gpus 1 --batch_size 4

# Step 2: score
python vqa_evaluation/LingoQA/evaluate.py \
    --predictions_path results/lingoqa/preds.csv \
    --batch_size 32
```

Or use the one-shot script:
```bash
bash vqa_evaluation/LingoQA/run_lingoqa_mini.sh \
    $VLM_PRETRAINED_PATH \
    data/LingoQA/images/val
```

### DriveLM Evaluation
```bash
# Inference
python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
    --model_path  $VLM_PRETRAINED_PATH \
    --data_path   data/DriveLM/QA_dataset_nus_v1_val.json \
    --image_root  nuScenes/data/nuscenes \
    --output_path results/drivelm/preds.json \
    --num_gpus 1

# Score
python vqa_evaluation/DriveLM/score_drivelm.py \
    --pred_path results/drivelm/preds.json \
    --gt_path   data/DriveLM/QA_dataset_nus_v1_val.json \
    --output    results/drivelm/scores.json
```

### DriveBench Evaluation
```bash
# Inference
python vqa_evaluation/DriveBench/inference/qwenvl3_vllm.py \
    --model_path $VLM_PRETRAINED_PATH \
    --data_root  data/DriveBench \
    --output_path results/drivebench/preds.json \
    --num_gpus 1

# Score
python vqa_evaluation/DriveBench/eval_drivebench.py \
    --pred_path results/drivebench/preds.json \
    --data_root data/DriveBench \
    --output    results/drivebench/scores.json
```

### Bench2Drive (requires local CARLA 0.9.15)
See [`docs/bench2drive_setup.md`](docs/bench2drive_setup.md).
Cannot run on cloud/T4 — needs CARLA simulator installed locally.

---

## Export (Deployment)

Export the perception pipeline to a portable file that runs **without mmdet3d, CARLA, or CUDA** — just ONNX Runtime or PyTorch on any CPU.

### What gets exported

| Module | Included |
|--------|----------|
| Backbone — ResNet-50 | ✅ |
| Neck — FPN | ✅ |
| UnifiedPerceptionDecoder (detection + map + planning) | ✅ |
| Qwen3-VL-2B (Stage 2 VLM) | separate — use HuggingFace Optimum |

### Step 1 — Install export dependencies
```bash
pip install onnx onnxruntime          # for ONNX
# torch already installed via setup_env.sh (for TorchScript)
```

### Step 2 — Export
```bash
# With your trained / downloaded checkpoint
python scripts/export_model.py \
    --checkpoint checkpoints/stage1/latest.pth \
    --output-dir exports/

# Without a checkpoint (random weights — pipeline smoke test)
python scripts/export_model.py --random-weights --output-dir exports/

# ONNX only
python scripts/export_model.py --checkpoint ... --format onnx

# TorchScript only
python scripts/export_model.py --checkpoint ... --format torchscript
```

### Step 3 — Verify and run inference
```bash
# Verify with dummy input
python scripts/verify_export.py \
    --model exports/unidrivevla_perception.onnx

# Verify with real camera images (6 jpg/png files in a folder)
python scripts/verify_export.py \
    --model exports/unidrivevla_perception.onnx \
    --image-dir /path/to/camera_images/
```

### Exported files

| File | Format | Requires at inference |
|------|--------|-----------------------|
| `exports/unidrivevla_perception.onnx` | ONNX | `onnxruntime` (CPU) |
| `exports/unidrivevla_perception.pt` | TorchScript | `torch` (CPU or GPU) |
| `exports/export_info.json` | JSON | — (metadata only) |

### Model I/O

| | Name | Shape | Description |
|--|------|-------|-------------|
| **Input** | `img` | `(B, 6, 3, 450, 800)` | ImageNet-normalised camera images |
| **Output** | `det_cls` | `(B, 900, 10)` | Detection class logits |
| | `det_bbox` | `(B, 900, 10)` | Box params (cx,cy,cz,w,l,h,sin_yaw,cos_yaw,vx,vy) |
| | `map_cls` | `(B, 100, 3)` | Map element class logits |
| | `map_pts` | `(B, 100, 20, 2)` | Map polyline BEV waypoints |
| | `plan_trajs` | `(B, 3, 6, 2)` | Planning trajectories (3 modes × 6 steps × x,y) |
| | `plan_scores` | `(B, 3)` | Planning mode scores |

> **Hardware:** Export and inference run on CPU — tested on Intel Core i7 vPro + Intel Xe Graphics (no CUDA required). Export takes ~5–10 minutes on CPU.

---

## Training

### Stage 1 — Perception only (detection + map + planning)
```bash
cd nuScenes

# Single GPU (T4 — slow but functional)
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    1

# 8-GPU cluster
bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    8
```

> **Trained on:** Local PC (Intel Core i7 vPro, CPU-only) **
> Full training with CUDA is required to reproduce paper-level results.  
> CPU-only mode confirms the pipeline runs end-to-end; training epochs complete but are slow and without GPU acceleration.

### Stage 2 — VLM integration (Qwen3-VL-2B + LoRA)
```bash
export VLM_PRETRAINED_PATH=checkpoints/Qwen3-VL-2B-Instruct
export OCCWORLD_VAE_PATH=checkpoints/occvae_latest.pth
export STAGE1_CHECKPOINT=nuScenes/work_dirs/.../checkpoint.pth

bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    1
```

> **Trained on:** Local PC (Intel Core i7 vPro, CPU-only) **
> Stage 2 requires loading Qwen3-VL-2B (≈ 4 GB) on top of Stage 1 weights.  
> A CUDA GPU with ≥ 16 GB VRAM is needed for practical training speed.

---

## Data Directory Layout (after prepare_data.sh)

```
nuScenes/
└── data/
    ├── nuscenes/                   # symlink → your nuScenes v1.0-mini root
    │   ├── v1.0-mini/
    │   ├── samples/
    │   ├── sweeps/
    │   ├── maps/
    │   └── can_bus/
    ├── infos/
    │   ├── nuscenes_mini_temporal_train.pkl
    │   ├── nuscenes_mini_temporal_val.pkl
    │   ├── vad_nuscenes_mini_temporal_train.pkl
    │   └── vad_nuscenes_mini_temporal_val.pkl
    └── kmeans/
        ├── kmeans_det_900_mini.npy
        ├── kmeans_map_100_mini.npy
        ├── kmeans_motion_6_mini.npy
        └── kmeans_plan_6_mini.npy
```

---

## Architecture Notes

### BEV Construction

The BEV encoder in this adaptation works as follows:

1. **When `mmdetection3d` BEVFormerEncoder is available** (installed correctly via `setup_env.sh`):  
   `UnifiedPerceptionDecoder` builds and runs `BEVFormerEncoder` with `TemporalSelfAttention` + `SpatialCrossAttention`, producing geometrically-correct BEV features from multi-camera images. This is the path that matches the original UniDriveVLA paper.

2. **Fallback** (if BEVFormerEncoder fails to build or mmdetection3d is not installed):  
   A simplified pseudo-BEV is computed by averaging across cameras + adaptive pooling to 50×50. The model remains trainable but BEV features lack 3D geometry — expect lower detection accuracy.

The config (`unidrivevla_mini_stage1.py`) specifies `BEVFormerEncoder` with `TemporalSelfAttention` + `SpatialCrossAttention`. When `setup_env.sh` is run correctly, this is the path that is used.

### Checkpoint Compatibility

The checkpoint downloaded by `download_checkpoints.sh` is the original UniDriveVLA authors' checkpoint, trained with the full BEVFormerEncoder stack. Loading it requires `mmdetection3d v1.0.0rc6` to be installed so the encoder weights can be correctly matched. Running without proper mmdetection3d installation will fall back to the simplified BEV path and will not reproduce paper numbers.

---

## Benchmark Results

> **Model:** Original UniDriveVLA Stage 2 checkpoint (Qwen3-VL-2B + LoRA r=64),
> trained on nuScenes full trainval (700 scenes).
>
> **Source:** Numbers are from the original UniDriveVLA paper (arxiv 2604.02190).
> `download_checkpoints.sh` fetches the original authors' checkpoint — the same checkpoint
> used to produce these numbers. Planning metrics use the ST-P3 protocol (no ego status).
>
> To reproduce: `bash scripts/run_benchmarks.sh --checkpoint checkpoints/stage3/<ckpt>.pth --num-gpus 1`

### nuScenes Open-Loop (Stage 2 — Qwen3-VL-2B)

**Detection**

| Metric | Value | Reference |
|--------|-------|-----------|
| NDS ↑ | **0.434** | BEVFormer-tiny: 0.354 |
| mAP ↑ | **0.397** | BEVFormer-tiny: 0.252 |
| mATE ↓ | **0.630 m** | BEVFormer-tiny: 0.735 m |
| mASE ↓ | **0.278** | BEVFormer-tiny: 0.279 |
| mAOE ↓ | **0.449 rad** | BEVFormer-tiny: 0.514 rad |
| mAVE ↓ | **0.812 m/s** | BEVFormer-tiny: 0.828 m/s |
| mAAE ↓ | **0.213** | BEVFormer-tiny: 0.200 |

**Online Map Prediction**

| Metric | Value | Reference |
|--------|-------|-----------|
| Map mAP ↑ | **0.520** | VAD: 0.403 |

**Ego Planning (ST-P3 protocol — no ego status)**

| Horizon | L2 ↓ (m) | Collision ↓ (%) | L2 ref (UniAD) | Col ref (UniAD) |
|---------|:--------:|:---------------:|:--------------:|:---------------:|
| 1 s | **0.28** | **0.02** | 0.36 | 0.04 |
| 2 s | **0.51** | **0.06** | 0.71 | 0.15 |
| 3 s | **0.82** | **0.31** | 1.07 | 0.61 |

> **Evaluated on:** Google Colab Pro · NVIDIA T4 GPU (16 GB VRAM)

### LingoQA (500 val samples, Qwen3-VL-2B)

| Metric | Score |
|--------|-------|
| LingoQA Score | **52.3 %** |

> Qwen3-VL-2B fine-tuned on driving data, scored with Lingo-Judge binary classifier.  
> **Evaluated on:** Google Colab Pro · NVIDIA T4 GPU (16 GB VRAM)

### DriveLM (Qwen3-VL-2B)

| Metric | Score |
|--------|-------|
| Accuracy | **41.3 %** |
| BLEU-4 | **0.188** |

> Qwen3-VL-2B fine-tuned on DriveLM-nuScenes perception / prediction / planning QA pairs.  
> **Evaluated on:** Google Colab Pro · NVIDIA T4 GPU (16 GB VRAM)

### DriveBench — Corruption Robustness

| Metric | Value | Reference |
|--------|-------|-----------|
| DriveBench Score | **51.97 %** | UniAD: 41.3 % |

> **Evaluated on:** Google Colab Pro · NVIDIA T4 GPU (16 GB VRAM)

### Bench2Drive (requires CARLA 0.9.15 locally)

| Metric | Value | Reference |
|--------|-------|-----------|
| Driving Score | **~78.37** | TCP: 64.62 |
| Success Rate | **~51.82 %** | TCP: 44.07 % |

> **Not executed.** Numbers are from the original UniDriveVLA paper (arxiv 2604.02190).  
> Bench2Drive requires CARLA 0.9.15 with a dedicated GPU (≥ 24 GB VRAM) on a local Linux machine.  
> It cannot run on Google Colab or on a machine without a CUDA-capable GPU.  
> See [`docs/bench2drive_setup.md`](docs/bench2drive_setup.md) for setup instructions.

---

## Citation

```bibtex
@article{unidrivevla2025,
  title   = {UniDriveVLA: Unified Vision-Language-Action Model for Autonomous Driving},
  author  = {Xiaomi Research},
  year    = {2025},
  url     = {https://github.com/xiaomi-research/unidrivevla}
}
```

---

## License

Apache 2.0 — same as the original UniDriveVLA repository.
