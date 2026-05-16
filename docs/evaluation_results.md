# Evaluation Results — UniDriveVLA nuScenes Mini

> Results marked **[CI dry-run]** are generated from mock data in GitHub Actions
> (no GPU, no checkpoint). Replace with real numbers after running
> `bash scripts/run_benchmarks.sh` locally with a downloaded checkpoint.

---

## nuScenes Mini — 3D Object Detection (Stage 1)

**Metric definitions:**
- **NDS**: nuScenes Detection Score (weighted combination of mAP and errors)
- **mAP**: mean Average Precision over 10 classes
- **mATE / mASE / mAOE / mAVE / mAAE**: translation / scale / orientation / velocity / attribute errors

**Run locally:**
```bash
cd nuScenes
bash tools/dist_eval.sh \
    projects/configs/UniDriveVLA/unidrivevla_mini_stage1.py \
    /path/to/epoch_24.pth \
    1 \
    --eval bbox
```

| Model | NDS | mAP | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ |
|-------|:---:|:---:|:-----:|:-----:|:-----:|:-----:|:-----:|
| UniDriveVLA-mini Stage 1 | — | — | — | — | — | — | — |

> **Note:** nuScenes mini val has only ~49 samples (3 scenes); metrics have high variance (±0.05–0.10 NDS). Use full trainval for reliable benchmarking.

---

## nuScenes Mini — Online Map Prediction (Stage 1)

| Model | mIoU divider | mIoU ped_crossing | mIoU boundary | mIoU avg |
|-------|:------------:|:-----------------:|:-------------:|:--------:|
| UniDriveVLA-mini Stage 1 | — | — | — | — |

---

## nuScenes Mini — Ego Planning (Stage 1 & 2)

**Metrics:** L2 displacement (m) and Collision Rate (%) per time horizon.
Lower is better.

**Run locally:**
```bash
python nuScenes/tools/evaluation/planning_eval.py \
    --result-path work_dirs/.../results.json \
    --gt-path     nuScenes/data/infos/nuscenes_mini_temporal_val.pkl \
    --output-dir  results/planning
```

**CI dry-run results** (mock data, 49 val samples, no checkpoint):

| Step | L2 (m) | Collision (%) |
|------|:------:|:-------------:|
| t = 1 s | 0.384 | 0.15 |
| t = 2 s | 0.664 | 0.20 |
| t = 3 s | 0.931 | 1.22 |
| t = 4 s | 1.213 | 1.17 |
| t = 5 s | 1.554 | 2.18 |
| t = 6 s | 1.956 | 2.24 |
| **avg** | **1.117** | **1.19** |

> These are **simulated** numbers from the CI dry-run. Run with a real checkpoint to obtain actual results.

---

## DriveLM (Stage 2)

DriveLM tests multi-camera scene understanding via structured QA (perception, prediction, planning).

**Run locally:**
```bash
python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
    --model_path "$VLM_PRETRAINED_PATH" \
    --data_path  data/DriveLM/QA_dataset_nus_v1_val.json \
    --image_root nuScenes/data/nuscenes \
    --output_path results/drivelm/preds.json

python vqa_evaluation/DriveLM/score_drivelm.py \
    --pred_path results/drivelm/preds.json \
    --gt_path   data/DriveLM/QA_dataset_nus_v1_val.json \
    --output    results/drivelm/scores.json
```

| Model | Accuracy | BLEU-4 | DriveLM Score |
|-------|:--------:|:------:|:-------------:|
| UniDriveVLA-mini Stage 2 | — | — | — |

**Expected range** (from original UniDriveVLA paper, full dataset): Accuracy ~41%, BLEU-4 ~0.19, DriveLM ~0.30

---

## LingoQA (Stage 2) — 500 val samples

LingoQA evaluates language-grounded driving scene understanding using a learned judge model.

**Run locally:**
```bash
bash vqa_evaluation/LingoQA/run_lingoqa_mini.sh \
    "$VLM_PRETRAINED_PATH" \
    data/LingoQA/images/val \
    results/lingoqa
```

| Model | LingoQA Score |
|-------|:-------------:|
| UniDriveVLA-mini Stage 2 | — |

**Expected range** (from original paper): ~52–55%

---

## DriveBench (Stage 2) — Corruption Robustness

**Metrics:** clean accuracy, mPC (mean Performance under Corruption), rPC (mPC / clean). Higher is better.

**Run locally:**
```bash
python vqa_evaluation/DriveBench/inference/qwenvl3_vllm.py \
    --model_path "$VLM_PRETRAINED_PATH" \
    --data_root  data/DriveBench \
    --output_path results/drivebench/preds.json

python vqa_evaluation/DriveBench/eval_drivebench.py \
    --pred_path results/drivebench/preds.json \
    --data_root data/DriveBench \
    --output    results/drivebench/scores.json
```

**CI dry-run results** (mock data, no checkpoint):

| Model | Clean Acc | mPC | rPC |
|-------|:---------:|:---:|:---:|
| UniDriveVLA-mini Stage 2 [CI dry-run] | 62.1% | 52.0% | 0.838 |

**Per-corruption accuracy (CI dry-run):**

| Corruption | Accuracy | Drop |
|------------|:--------:|:----:|
| fog | 51.6% | 10.5% |
| rain | 53.0% | 9.1% |
| night | 48.7% | 13.4% |
| motion_blur | 51.6% | 10.5% |
| gaussian_noise | 48.9% | 13.2% |
| brightness | 54.9% | 7.2% |
| contrast | 55.7% | 6.4% |

> These are **simulated** from CI dry-run mock data. Run locally for real results.

---

## Bench2Drive — Closed-Loop (Stage 2)

> **Requires CARLA 0.9.15 locally.** See [bench2drive_setup.md](bench2drive_setup.md).

| Model | DS | SR (%) | RC (%) | IS |
|-------|----|:------:|:------:|:--:|
| UniDriveVLA-mini Stage 2 | — | — | — | — |

---

## How to Get Real Results

```bash
# 1. Set up environment
bash scripts/setup_env.sh

# 2. Download checkpoints
bash scripts/download_checkpoints.sh

# 3. Prepare nuScenes mini data
bash scripts/prepare_data.sh /path/to/nuscenes

# 4. Run all benchmarks
bash scripts/run_benchmarks.sh \
    --checkpoint checkpoints/stage3/mp_rank_00_model_states.pt \
    --num-gpus 1

# Results saved to results/benchmark_summary.txt
```

Populate this file by replacing `—` values with numbers from `results/benchmark_summary.txt`.
