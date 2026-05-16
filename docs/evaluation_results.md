# Evaluation Results

Structured metric tables for all UniDriveVLA benchmarks.
Run the corresponding evaluation commands to populate with actual values.

---

## nuScenes Mini — 3D Object Detection (Stage 1)

**Metric definitions:**
- **NDS** (nuScenes Detection Score): weighted combination of mAP, mATE, mASE, mAOE, mAVE, mAAE
- **mAP**: mean Average Precision (IoU-based matching over class set)
- **mATE**: mean Average Translation Error (m, L2 distance of box centres)
- **mASE**: mean Average Scale Error (1 − IoU of boxes matched by centre)
- **mAOE**: mean Average Orientation Error (rad, yaw angle error)
- **mAVE**: mean Average Velocity Error (m/s)
- **mAAE**: mean Average Attribute Error

**Command:**
```bash
python tools/evaluation/nuscenes_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --dataroot data/nuscenes --version v1.0-mini --eval-set mini_val \
    --eval-detection --output-dir work_dirs/mini_stage1/eval_results
```

| Model | Checkpoint | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|-------|-----------|:---:|:---:|:----:|:----:|:----:|:----:|:----:|
| UniDriveVLA-mini S1 (T4, 24ep) | epoch_24.pth | — | — | — | — | — | — | — |

### Per-Class AP

| Class | AP |
|-------|----|
| car | — |
| truck | — |
| construction_vehicle | — |
| bus | — |
| trailer | — |
| barrier | — |
| motorcycle | — |
| bicycle | — |
| pedestrian | — |
| traffic_cone | — |

---

## nuScenes Mini — Online Map Prediction (Stage 1)

**Metric definitions:**
- **mIoU**: mean Intersection over Union for each map element class (divider, ped_crossing, boundary)

**Command:**
```bash
python tools/evaluation/nuscenes_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --dataroot data/nuscenes --version v1.0-mini --eval-set mini_val \
    --eval-map --output-dir work_dirs/mini_stage1/eval_results
```

| Model | mIoU (divider) | mIoU (ped_crossing) | mIoU (boundary) | mIoU (avg) |
|-------|:--------------:|:-------------------:|:---------------:|:----------:|
| UniDriveVLA-mini S1 | — | — | — | — |

---

## nuScenes Mini — Ego Motion / Planning (Stage 1)

**Metric definitions:**
- **L2 @ Xs**: L2 distance between predicted and ground-truth ego trajectory at horizon X seconds
- **CR @ Xs** (Collision Rate): fraction of predicted trajectories that intersect annotated obstacles at X seconds
- Lower is better for all planning metrics.

**Command:**
```bash
python tools/evaluation/planning_eval.py \
    --result-path work_dirs/mini_stage1/results.json \
    --output-dir  work_dirs/mini_stage1/planning_results
```

| Model | L2 @ 1s | L2 @ 2s | L2 @ 3s | CR @ 1s | CR @ 2s | CR @ 3s |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| UniDriveVLA-mini S1 | — | — | — | — | — | — |

---

## DriveLM (Stage 2)

DriveLM tests multi-camera scene understanding and reasoning through structured QA pairs
(perception, prediction, and planning categories).

**Metric definitions:**
- **Accuracy**: exact-match accuracy for multiple-choice / short-answer questions
- **BLEU-4**: 4-gram precision score (corpus level)
- **CIDEr**: consensus-based image description evaluation metric
- **METEOR**: unigram recall against references (if nltk available)
- **DriveLM score**: official weighted combination of Accuracy, BLEU-4, CIDEr

**Command:**
```bash
python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
    --model_path work_dirs/mini_stage2/latest.pth \
    --data_path  data/drivelm/v1_1_val_nus_q_only.json \
    --image_root data/nuscenes \
    --output_dir results/drivelm

python vqa_evaluation/DriveLM/eval_drivelm.py \
    --predictions results/drivelm/drivelm_preds.json \
    --answers     data/drivelm/v1_1_val_nus_a.json \
    --output-dir  results/drivelm
```

| Model | Accuracy | BLEU-4 | CIDEr | METEOR | DriveLM Score |
|-------|:--------:|:------:|:-----:|:------:|:-------------:|
| UniDriveVLA-mini S2 | — | — | — | — | — |

### Per-Category Breakdown

| Category | Accuracy |
|----------|:--------:|
| Perception | — |
| Prediction | — |
| Planning   | — |

---

## LingoQA (Stage 2) — 500 Samples

LingoQA evaluates language-grounded scene understanding for autonomous driving.
Uses a learned judge model (LingoJudge) for binary correctness scoring.

**Metric definitions:**
- **LingoQA Score**: percentage of judge-evaluated correct answers (official metric)
- Sub-categories: Action, Spatial, Attribute, Prediction, Planning

**Command:**
```bash
bash vqa_evaluation/LingoQA/run_lingoqa_mini.sh \
    work_dirs/mini_stage2/latest.pth \
    data/LingoQA/images/val
```

| Model | LingoQA Score | Action | Spatial | Attribute | Prediction | Planning |
|-------|:-------------:|:------:|:-------:|:---------:|:----------:|:--------:|
| UniDriveVLA-mini S2 | — % | — | — | — | — | — |

---

## DriveBench (Stage 2) — Corruption Robustness

DriveBench tests model robustness by evaluating on nuScenes data corrupted with
various sensor artifacts at 3 severity levels.

**Metric definitions:**
- **Accuracy**: exact-match accuracy on each corruption type/severity
- **mPC** (mean Performance under Corruptions): mean accuracy across all corruption types
- **rPC** (relative PC): mPC / clean_accuracy — measures degradation relative to clean performance
- Higher is better for all robustness metrics.

**Command:**
```bash
python vqa_evaluation/DriveBench/eval_drivebench.py \
    --model-config projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    --checkpoint   work_dirs/mini_stage2/latest.pth \
    --data-root    data/drivebench \
    --output-dir   results/drivebench
```

| Model | Clean Acc | mPC | rPC |
|-------|:---------:|:---:|:---:|
| UniDriveVLA-mini S2 | — | — | — |

### Per-Corruption Accuracy

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|------------|:----------:|:----------:|:----------:|
| CameraBlur | — | — | — |
| CameraFog | — | — | — |
| CameraRain | — | — | — |
| CameraSnow | — | — | — |
| CameraContrast | — | — | — |
| CameraBrightness | — | — | — |
| CameraSaturation | — | — | — |
| CameraJpeg | — | — | — |

---

## Bench2Drive — Closed-Loop (Stage 2)

> **Requires CARLA 0.9.15 installed locally.**
> See [bench2drive_setup.md](bench2drive_setup.md) for setup instructions.
> This cannot run in standard CI.

**Metric definitions:**
- **DS** (Driving Score): primary metric; weighted combination of RC and IS
- **RC** (Route Completion %): percentage of route distance completed
- **IS** (Infraction Score): multiplicative penalty for infractions (collisions, red lights, etc.)
- **SR** (Success Rate): fraction of routes completed without critical infractions
- **MATA** (Multi-Ability Test Aggregation): aggregated score over scenario sub-categories

| Model | DS | SR (%) | RC (%) | IS | MATA |
|-------|----|:------:|:------:|:--:|:----:|
| UniDriveVLA-mini S2 | — | — | — | — | — |

---

## Notes on nuScenes Mini Variance

The mini validation split contains only ~81 samples (3 scenes). This means:

1. Detection/mapping metrics will vary significantly (±0.05–0.10 NDS) between runs
2. A single missed or wrongly-predicted scene can shift mAP by >5 points
3. Results should be treated as sanity checks, not performance benchmarks
4. For reliable benchmarking, retrain and evaluate on the full nuScenes dataset

To reproduce results on the full dataset, change:
- `version = 'v1.0'`
- `ann_file_train/val` to the full temporal PKL files
- `samples_per_gpu = 2` (or higher, depending on GPU memory)
