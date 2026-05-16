# Training and Evaluation on nuScenes Mini

## Overview

UniDriveVLA uses a two-stage training scheme:

| Stage | Task | Epochs | Approx. time on T4 (mini) |
|-------|------|:------:|:-------------------------:|
| Stage 1 | Perception (detection + map + planning) | 24 | ~3–8 h |
| Stage 2 | Perception + Qwen3-VL-2B (LoRA) | 15 | ~4–12 h |

With nuScenes mini (~282 train samples), one epoch of Stage 1 takes roughly:
- **T4 GPU (16 GB)**: ~4–7 min (batch=1, FP32/mixed)
- **A100 (40 GB)**: ~1–3 min

## Prerequisites

1. Conda environment with PyTorch 2.5.1, mmcv-full 1.7.2, mmdet3d 1.4.0
2. nuScenes mini info files generated:
   ```bash
   cd nuScenes
   bash scripts/create_data_mini.sh
   ```
3. For Stage 2: `VLM_PRETRAINED_PATH` env var pointing to `Qwen/Qwen3-VL-2B-Instruct`

---

## Stage 1 Training

### Single GPU

```bash
cd nuScenes

python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --work-dir work_dirs/mini_stage1 \
    --seed 42
```

### Multi-GPU (4 GPUs)

```bash
cd nuScenes

PORT=29500 bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    4 \
    --work-dir work_dirs/mini_stage1
```

### Checkpoints

Checkpoints are saved every 4 epochs to `work_dirs/mini_stage1/epoch_*.pth`.
The latest checkpoint is symlinked as `latest.pth`.

### Monitoring

```bash
tensorboard --logdir work_dirs/mini_stage1
```

### Resuming from a crash

```bash
python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --work-dir work_dirs/mini_stage1 \
    --resume-from work_dirs/mini_stage1/latest.pth
```

### Config overrides

```bash
python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --cfg-options \
        data.samples_per_gpu=1 \
        optimizer.lr=1e-4 \
        total_epochs=12
```

---

## Stage 2 Training (requires Stage 1 checkpoint)

Stage 2 freezes the image backbone, FPN neck, and BEV encoder.
Only the VLM LoRA adapter (`r=64, alpha=128`) and the BEV↔VLM projection layers are trained.

```bash
cd nuScenes

export VLM_PRETRAINED_PATH="Qwen/Qwen3-VL-2B-Instruct"
export STAGE1_CHECKPOINT="work_dirs/mini_stage1/latest.pth"

python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --work-dir work_dirs/mini_stage2
```

### Multi-GPU with DeepSpeed ZeRO-1 (recommended)

```bash
deepspeed tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --deepspeed nuScenes/zero_configs/adam_zero1_bf16.json \
    --work-dir work_dirs/mini_stage2
```

> Note: `attn_implementation='eager'` is set in the config because Flash Attention 2
> requires Ampere architecture (A100/RTX 3000+). T4 GPUs (Turing) are not supported.

### 8×A100 (full scale, recommended for paper reproduction)

```bash
PORT=29500 bash tools/dist_train.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    8 \
    --work-dir work_dirs/mini_stage2_8gpu
```

---

## Evaluation

### nuScenes Detection + Mapping

```bash
cd nuScenes

# Single GPU
python tools/test.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    work_dirs/mini_stage1/latest.pth \
    --eval bbox map \
    --json-out work_dirs/mini_stage1/results.json

# Multi-GPU (1 GPU for mini)
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    work_dirs/mini_stage1/latest.pth \
    1 \
    --eval bbox map \
    --json-out work_dirs/mini_stage1/results.json
```

### Full nuScenes Metric Suite (NDS, mAP, mATE, mASE, mAOE, mAVE, mAAE)

```bash
python tools/evaluation/nuscenes_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --dataroot    data/nuscenes \
    --version     v1.0-mini \
    --eval-set    mini_val \
    --eval-detection \
    --eval-map \
    --output-dir  work_dirs/mini_stage1/eval_results
```

Expected output format:
```
Evaluating Detection:
  NDS   : 0.XXX
  mAP   : 0.XXX
  mATE  : 0.XXX
  mASE  : 0.XXX
  mAOE  : 0.XXX
  mAVE  : 0.XXX
  mAAE  : 0.XXX
```

### Planning Metrics (L2, Collision Rate)

```bash
python tools/evaluation/planning_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --output-dir  work_dirs/mini_stage1/planning_results
```

Expected output format:
```
Planning Evaluation:
  L2 @ 1s : X.XXX m
  L2 @ 2s : X.XXX m
  L2 @ 3s : X.XXX m
  CR @ 1s : X.XXX %
  CR @ 2s : X.XXX %
  CR @ 3s : X.XXX %
```

---

## VQA Evaluation (Stage 2)

### LingoQA (500 samples)

```bash
# Full pipeline (inference + scoring)
bash vqa_evaluation/LingoQA/run_lingoqa_mini.sh \
    work_dirs/mini_stage2/latest.pth \
    data/LingoQA/images/val

# Or separately:
python vqa_evaluation/LingoQA/infer_qwenvl3.py \
    --model_path   work_dirs/mini_stage2/latest.pth \
    --image_root   data/LingoQA/images/val \
    --parquet_path vqa_evaluation/LingoQA/val.parquet \
    --output_path  results/lingoqa_preds.csv

python vqa_evaluation/LingoQA/evaluate.py \
    --pred_path  results/lingoqa_preds.csv \
    --gt_path    vqa_evaluation/LingoQA/val.parquet \
    --output_dir results/lingoqa_eval
```

### DriveLM

```bash
# Inference
python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
    --model_path work_dirs/mini_stage2/latest.pth \
    --data_path  data/drivelm/v1_1_val_nus_q_only.json \
    --image_root data/nuscenes \
    --output_dir results/drivelm

# Full metrics
python vqa_evaluation/DriveLM/eval_drivelm.py \
    --predictions results/drivelm/drivelm_preds.json \
    --answers     data/drivelm/v1_1_val_nus_a.json \
    --output-dir  results/drivelm
```

### DriveBench

```bash
python vqa_evaluation/DriveBench/eval_drivebench.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint   work_dirs/mini_stage2/latest.pth \
    --data-root    data/drivebench \
    --output-dir   results/drivebench
```

---

## Expected Metric Ranges (nuScenes mini val)

Because the mini split has only ~81 validation samples, all metrics have
**high variance** between runs. The table below shows approximate expected ranges
for a model trained on mini (not guaranteed performance):

| Metric | Expected Range | Notes |
|--------|:--------------:|-------|
| NDS    | 0.20 – 0.45    | High variance; mini val has 81 samples |
| mAP    | 0.10 – 0.35    | |
| map mIoU | 0.10 – 0.30  | |
| L2 @ 1s | 0.3 – 1.0 m  | |
| L2 @ 2s | 0.6 – 2.0 m  | |
| L2 @ 3s | 1.0 – 3.5 m  | |
| LingoQA Score | 0.35 – 0.60 | Token F1 approx. |
| DriveLM Acc   | 0.40 – 0.65 | |

These ranges are not benchmarks — they reflect the variance expected when training
on 282 samples and evaluating on 81 samples.

---

## Results Interpretation

The evaluation scripts produce JSON metric files:

```
work_dirs/mini_stage1/eval_results/
├── nuscenes_metrics.json    # NDS, mAP, per-class AP
├── map_metrics.json         # per-class mIoU, avg mIoU
└── planning_metrics.json    # L2, collision rate

results/
├── lingoqa_eval/lingoqa_metrics.json
├── drivelm/drivelm_metrics.json
└── drivebench/drivebench_metrics.json
```

See [docs/evaluation_results.md](evaluation_results.md) for structured tables and metric definitions.
