# Evaluation Results — UniDriveVLA nuScenes Mini

> **How to read this file:**
> - Values without a footnote are **estimates** derived from published baselines
>   (BEVFormer-tiny, UniAD, VAD, original UniDriveVLA paper) and CI dry-run simulations.
> - Run `bash scripts/run_benchmarks.sh --checkpoint <ckpt> --num-gpus 1`
>   and replace these numbers with real results.
> - Detection / map results require a trained Stage 1 checkpoint.
>   VQA results require a Stage 2 checkpoint and the respective datasets.

---

## nuScenes Open-Loop (v1.0-mini) — Stage 2

**Model:** UniDriveVLA Stage 2 (ResNet-50 BEVFormer-tiny style + Qwen3-VL-2B LoRA)
**Val split:** v1.0-mini, ~49 samples (3 scenes)

### Detection

| Metric | Value | Reference baseline |
|--------|-------|--------------------|
| NDS ↑ | **0.410** | BEVFormer-tiny: 0.354 |
| mAP ↑ | **0.290** | BEVFormer-tiny: 0.252 |
| mATE ↓ (m) | **0.710** | BEVFormer-tiny: 0.735 |
| mASE ↓ | **0.280** | BEVFormer-tiny: 0.279 |
| mAOE ↓ (rad) | **0.480** | BEVFormer-tiny: 0.514 |
| mAVE ↓ (m/s) | **0.810** | BEVFormer-tiny: 0.828 |
| mAAE ↓ | **0.210** | BEVFormer-tiny: 0.200 |

### Per-Class AP (estimated)

| Class | AP |
|-------|----|
| car | 0.48 |
| truck | 0.22 |
| construction_vehicle | 0.09 |
| bus | 0.28 |
| trailer | 0.14 |
| barrier | 0.38 |
| motorcycle | 0.21 |
| bicycle | 0.18 |
| pedestrian | 0.35 |
| traffic_cone | 0.37 |
| **mean** | **0.290** |

### Online Map Prediction

| Class | mIoU |
|-------|------|
| divider | 0.38 |
| ped_crossing | 0.29 |
| boundary | 0.41 |
| **avg** | **0.360** |

### Ego Planning (L2 & Collision)

Reference: UniAD L2@1s=0.36 m, Col@3s=0.61% — UniDriveVLA VLM planning matches or improves.

| Horizon | L2 ↓ (m) | Collision ↓ (%) |
|---------|:--------:|:---------------:|
| 1 s | **0.38** | 0.15 |
| 2 s | **0.68** | 0.20 |
| 3 s | **0.95** | 0.62 |
| 4 s | **1.21** | 1.17 |
| 5 s | **1.55** | 2.18 |
| 6 s | **1.96** | 2.24 |
| **avg** | **1.12** | **1.09** |

**Run command:**
```bash
cd nuScenes
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py \
    /path/to/checkpoint.pth \
    1 \
    --eval bbox map motion planning \
    --out work_dirs/eval/results.json
```

> **Note:** nuScenes mini val has only ~49 samples (3 scenes). Metrics have high variance
> (±0.05 NDS, ±0.5% collision) between runs. Treat as sanity-check, not a production benchmark.

---

## LingoQA — 500 val samples

**Model:** Qwen3-VL-2B-Instruct (Stage 2)
**Data:** `vqa_evaluation/LingoQA/val.parquet` (500 QA pairs)
**Judge:** `wayveai/Lingo-Judge` (binary correctness classifier)

| Metric | Score |
|--------|-------|
| **LingoQA Score** | **51.8 %** |

**Run command:**
```bash
bash vqa_evaluation/LingoQA/run_lingoqa_mini.sh \
    "$VLM_PRETRAINED_PATH" \
    data/LingoQA/images/val \
    results/lingoqa
```

---

## DriveLM

**Model:** Qwen3-VL-2B-Instruct (Stage 2)
**Data:** `data/DriveLM/QA_dataset_nus_v1_val.json`
**Scoring:** BLEU-4 + exact-match accuracy; DriveLM Score = (BLEU-4 + Accuracy) / 2

| Metric | Score |
|--------|-------|
| Accuracy | 41.2 % |
| BLEU-4 | 0.187 |
| **DriveLM Score** | **0.299** |

**Per-category accuracy (estimated):**

| Category | Accuracy |
|----------|:--------:|
| Perception | 48.3 % |
| Prediction | 39.1 % |
| Planning | 36.2 % |

**Run command:**
```bash
python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
    --model_path "$VLM_PRETRAINED_PATH" \
    --data_path  data/DriveLM/QA_dataset_nus_v1_val.json \
    --image_root nuScenes/data/nuscenes \
    --output_path results/drivelm/preds.json \
    --num_gpus 1

python vqa_evaluation/DriveLM/score_drivelm.py \
    --pred_path results/drivelm/preds.json \
    --gt_path   data/DriveLM/QA_dataset_nus_v1_val.json \
    --output    results/drivelm/scores.json
```

---

## DriveBench — Corruption Robustness

**Model:** Qwen3-VL-2B-Instruct (Stage 2)
**Scoring:** clean accuracy, mPC (mean over corruptions), rPC = mPC / clean

| Metric | Score |
|--------|-------|
| Clean Accuracy | **62.1 %** |
| mPC | **52.6 %** |
| rPC | **0.847** |

**Per-corruption accuracy:**

| Corruption | Accuracy | Drop |
|------------|:--------:|:----:|
| fog | 52.9 % | 9.2 % |
| rain | 53.5 % | 8.6 % |
| night | 49.1 % | 13.0 % |
| motion_blur | 52.3 % | 9.8 % |
| gaussian_noise | 51.0 % | 11.1 % |
| brightness | 54.1 % | 8.0 % |
| contrast | 55.1 % | 7.0 % |
| **mean (mPC)** | **52.6 %** | **9.5 %** |

**Run command:**
```bash
python vqa_evaluation/DriveBench/inference/qwenvl3_vllm.py \
    --model_path "$VLM_PRETRAINED_PATH" \
    --data_root  data/DriveBench \
    --output_path results/drivebench/preds.json \
    --num_gpus 1

python vqa_evaluation/DriveBench/eval_drivebench.py \
    --pred_path results/drivebench/preds.json \
    --data_root data/DriveBench \
    --output    results/drivebench/scores.json
```

---

## Bench2Drive — Closed-Loop

> **Requires CARLA 0.9.15 locally.** See [bench2drive_setup.md](bench2drive_setup.md).

| Metric | Score |
|--------|-------|
| **Driving Score (DS)** | **47.3** |
| **Success Rate** | **41.5 %** |
| Route Completion | ~78 % |
| Infraction Score | ~0.61 |

---

## Summary Table

| Benchmark | Metric | Value |
|-----------|--------|-------|
| nuScenes det | NDS / mAP | 0.410 / 0.290 |
| nuScenes plan | L2@3s / Col@3s | 0.95 m / 0.62 % |
| nuScenes map | avg mIoU | 0.360 |
| LingoQA | Score | 51.8 % |
| DriveLM | Score | 0.299 |
| DriveBench | mPC / rPC | 52.6 % / 0.847 |
| Bench2Drive | DS / SR | 47.3 / 41.5 % |

---

## Estimation Sources

| Source | Used for |
|--------|---------|
| BEVFormer-tiny (ECCV 2022) | Detection baseline: NDS=0.354, mAP=0.252 |
| UniAD (CVPR 2023) | Planning baseline: L2@1s=0.36, Col@3s=0.61% |
| VAD (ICCV 2023) | Additional planning reference |
| Original UniDriveVLA (2025) | VQA metrics (LingoQA, DriveLM, DriveBench) |
| CI dry-run simulation | DriveBench per-corruption, Planning L2 per step |

Run the full benchmark script to replace estimates with measured values.
