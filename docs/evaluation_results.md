# Evaluation Results

> This file is a placeholder for benchmark results.
> Populate this file by running the evaluation scripts after training.

## nuScenes Mini — Detection (Stage 1)

Run:
```bash
python tools/evaluation/nuscenes_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --dataroot data/nuscenes --version v1.0-mini --eval-set mini_val \
    --eval-detection --output-dir work_dirs/mini_stage1/eval_results
```

| Model | Checkpoint | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|-------|-----------|-----|-----|------|------|------|------|------|
| UniDriveVLA-mini (S1) | epoch_24.pth | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Per-Class AP

| Class | AP |
|-------|----|
| car | TBD |
| truck | TBD |
| construction_vehicle | TBD |
| bus | TBD |
| trailer | TBD |
| barrier | TBD |
| motorcycle | TBD |
| bicycle | TBD |
| pedestrian | TBD |
| traffic_cone | TBD |

---

## nuScenes Mini — Online Mapping (Stage 1)

| Model | mIoU (divider) | mIoU (ped_crossing) | mIoU (boundary) | mIoU (avg) |
|-------|---------------|---------------------|-----------------|------------|
| UniDriveVLA-mini (S1) | TBD | TBD | TBD | TBD |

---

## nuScenes Mini — Planning (Stage 1)

Run:
```bash
python tools/evaluation/planning_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --output-dir work_dirs/mini_stage1/planning_results
```

| Model | L2 (1s) | L2 (2s) | L2 (3s) | Col. Rate (1s) | Col. Rate (2s) | Col. Rate (3s) |
|-------|---------|---------|---------|----------------|----------------|----------------|
| UniDriveVLA-mini (S1) | TBD | TBD | TBD | TBD | TBD | TBD |

---

## VQA — DriveLM (Stage 2)

Run:
```bash
python vqa_evaluation/DriveLM/eval_drivelm.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --data-root data/drivelm --split val \
    --output-dir work_dirs/drivelm_results
```

| Model | Accuracy | BLEU-4 | CIDEr | METEOR |
|-------|---------|--------|-------|--------|
| UniDriveVLA-mini (S2) | TBD | TBD | TBD | TBD |

---

## VQA — LingoQA (Stage 2)

Run:
```bash
python vqa_evaluation/LingoQA/eval_lingoqa.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --data-root data/lingoqa \
    --output-dir work_dirs/lingoqa_results
```

| Model | LingoQA Score |
|-------|--------------|
| UniDriveVLA-mini (S2) | TBD |

---

## VQA — DriveBench (Stage 2)

Run:
```bash
python vqa_evaluation/DriveBench/eval_drivebench.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint work_dirs/mini_stage2/latest.pth \
    --data-root data/drivebench \
    --output-dir work_dirs/drivebench_results
```

| Model | DriveBench Score | Corruption Robustness |
|-------|-----------------|----------------------|
| UniDriveVLA-mini (S2) | TBD | TBD |

---

## Bench2Drive — Closed-Loop (Stage 2)

> Requires local CARLA 0.9.15. See [docs/bench2drive_setup.md](bench2drive_setup.md).

| Model | DS | RC (%) | IS | Ped. Col. | Veh. Col. |
|-------|----|--------|----|-----------|-----------|
| UniDriveVLA-mini (S2) | TBD | TBD | TBD | TBD | TBD |
