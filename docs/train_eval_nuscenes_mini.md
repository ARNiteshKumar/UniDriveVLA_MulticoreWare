# Training and Evaluation on nuScenes Mini

## Overview

UniDriveVLA uses a two-stage training scheme:

| Stage | Task | Duration (mini) |
|-------|------|-----------------|
| Stage 1 | Perception only (detection + map + planning) | 24 epochs |
| Stage 2 | Perception + VLM (adds QA heads) | 15 epochs |

On nuScenes mini (282 train samples), one epoch takes roughly:
- **Stage 1**: ~2–5 min on a single A100
- **Stage 2**: ~3–8 min on a single A100 (VLM forward pass is heavier)

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

## Stage 2 Training (requires Stage 1 checkpoint)

```bash
cd nuScenes

python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --work-dir work_dirs/mini_stage2 \
    --load-from work_dirs/mini_stage1/latest.pth
```

Stage 2 freezes the image backbone, FPN neck, and BEV encoder.
Only the VLM LoRA adapter and the BEV↔VLM projection layers are trained.

For Stage 2 with DeepSpeed ZeRO-1 (recommended for 7B VLM):

```bash
deepspeed tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --deepspeed nuScenes/zero_configs/adam_zero1_bf16.json \
    --work-dir work_dirs/mini_stage2 \
    --load-from work_dirs/mini_stage1/latest.pth
```

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

# Multi-GPU
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    work_dirs/mini_stage1/latest.pth \
    4 \
    --eval bbox map \
    --json-out work_dirs/mini_stage1/results.json
```

### Full nuScenes Metric Suite

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

### Planning Metrics

```bash
python tools/evaluation/planning_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --config projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --output-dir work_dirs/mini_stage1/planning_results
```

## Expected Metrics (nuScenes mini val)

Because the mini split has only 81 validation samples, all metrics will have
**high variance**. The table below shows approximate expected ranges, not
guaranteed performance:

| Metric | Expected Range |
|--------|---------------|
| NDS    | 0.20 – 0.45   |
| mAP    | 0.10 – 0.35   |
| map_mIoU | 0.10 – 0.30 |
| plan_L2_1s | 0.3 – 1.0 m |
| plan_L2_2s | 0.6 – 2.0 m |
| plan_L2_3s | 1.0 – 3.5 m |

## Config overrides at command line

```bash
python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --cfg-options \
        data.samples_per_gpu=2 \
        optimizer.lr=1e-4 \
        total_epochs=12
```

## Resuming from a crash

```bash
python tools/train.py \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    --work-dir work_dirs/mini_stage1 \
    --resume-from work_dirs/mini_stage1/latest.pth
```
