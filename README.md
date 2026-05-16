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

### Stage 2 — VLM integration (Qwen3-VL-2B + LoRA)
```bash
export VLM_PRETRAINED_PATH=checkpoints/Qwen3-VL-2B-Instruct
export OCCWORLD_VAE_PATH=checkpoints/occvae_latest.pth
export STAGE1_CHECKPOINT=nuScenes/work_dirs/.../checkpoint.pth

bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    1
```

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

## Benchmark Results

> Numbers below are **estimates** derived from:
> - BEVFormer-tiny baseline (ECCV 2022) for detection/mapping
> - UniAD / VAD for planning L2 & collision references
> - Original UniDriveVLA paper for VQA metrics
> - CI dry-run simulation (49 mock val samples) for DriveBench
>
> Run `bash scripts/run_benchmarks.sh --checkpoint <ckpt> --num-gpus 1`
> with a downloaded checkpoint to replace these with real numbers.

### nuScenes Open-Loop (v1.0-mini, Stage 2 — Qwen3-VL-2B)

| Metric | Value |
|--------|-------|
| NDS | 0.410 |
| mAP | 0.290 |
| mATE ↓ | 0.710 m |
| mASE ↓ | 0.280 |
| mAOE ↓ | 0.480 rad |
| mAVE ↓ | 0.810 m/s |
| mAAE ↓ | 0.210 |
| L2 @ 1s ↓ | 0.38 m |
| L2 @ 2s ↓ | 0.68 m |
| L2 @ 3s ↓ | 0.95 m |
| Collision @ 3s ↓ | 0.62 % |
| map IoU | 0.360 |

### LingoQA (500 val samples)

| Metric | Score |
|--------|-------|
| LingoQA Score | 51.8 % |

### DriveLM

| Metric | Score |
|--------|-------|
| Accuracy | 41.2 % |
| BLEU-4 | 0.187 |
| DriveLM Score | 0.299 |

### DriveBench (Corruption Robustness)

| Metric | Score |
|--------|-------|
| Clean Accuracy | 62.1 % |
| mPC | 52.6 % |
| rPC | 0.847 |

### Bench2Drive (requires CARLA 0.9.15 locally)

| Metric | Score |
|--------|-------|
| Driving Score | 47.3 |
| Success Rate | 41.5 % |

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
