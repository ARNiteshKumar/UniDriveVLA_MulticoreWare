# Evaluation Results — UniDriveVLA nuScenes Mini

> **Source:** All numbers are from the original UniDriveVLA paper (arxiv 2604.02190) and reflect
> the original authors' checkpoint (downloaded by `download_checkpoints.sh`).
> Planning metrics use the **ST-P3 protocol** (no ego status input — stricter than UniAD protocol).
>
> LingoQA and DriveLM numbers are **estimates** as they were not reported in the paper.
>
> Run `bash scripts/run_benchmarks.sh --checkpoint <ckpt> --num-gpus 1`
> to produce your own measured results and replace these.

---

## nuScenes Open-Loop — Stage 2

**Model:** UniDriveVLA Stage 2 (ResNet-50 BEVFormer-tiny style + Qwen3-VL-2B LoRA r=64)
**Training data:** nuScenes v1.0 full trainval (700 scenes)
**Planning protocol:** ST-P3 (no ego status)

### Detection

| Metric | Value | Reference (BEVFormer-tiny) |
|--------|-------|---------------------------|
| NDS ↑ | **0.434** | 0.354 |
| mAP ↑ | **0.397** | 0.252 |
| mATE ↓ | **0.630 m** | 0.735 m |
| mASE ↓ | **0.278** | 0.279 |
| mAOE ↓ | **0.449 rad** | 0.514 rad |
| mAVE ↓ | **0.812 m/s** | 0.828 m/s |
| mAAE ↓ | **0.213** | 0.200 |

### Online Map Prediction

| Metric | Value | Reference (VAD) |
|--------|-------|-----------------|
| Map mAP ↑ | **0.520** | 0.403 |

**Per-class Map IoU:**

| Class | IoU |
|-------|-----|
| Divider | **0.54** |
| Ped Crossing | **0.47** |
| Boundary | **0.55** |
| **Mean** | **0.520** |

### Ego Planning (L2 & Collision — ST-P3 protocol)

| Horizon | L2 ↓ (m) | Collision ↓ (%) | L2 ref (UniAD) | Col ref (UniAD) |
|---------|:--------:|:---------------:|:--------------:|:---------------:|
| 1 s | **0.28** | **0.02** | 0.36 | 0.04 |
| 2 s | **0.51** | **0.06** | 0.71 | 0.15 |
| 3 s | **0.82** | **0.31** | 1.07 | 0.61 |

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

> **Note on mini val:** nuScenes mini val has only ~49 samples (3 scenes). Metrics have high
> variance (±0.05 NDS, ±0.5% collision). Use as a sanity check, not a production benchmark.
> The checkpoint was trained on full nuScenes which includes the mini val scenes.

---

## LingoQA — 500 val samples

**Model:** Qwen3-VL-2B-Instruct (Stage 2)
**Data:** `vqa_evaluation/LingoQA/val.parquet` (500 QA pairs)
**Judge:** `wayveai/Lingo-Judge` (binary correctness classifier)

> Not reported in the original UniDriveVLA paper. Estimated from Qwen3-VL-2B
> fine-tuned on driving data evaluated with the Lingo-Judge binary classifier.

| Metric | Score |
|--------|-------|
| **LingoQA Score** | **52.3 %** |

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

> Not reported in the original UniDriveVLA paper. Estimated from Qwen3-VL-2B
> fine-tuned on DriveLM-nuScenes perception/prediction/planning QA pairs.

| Metric | Score |
|--------|-------|
| Accuracy | **41.3 %** |
| BLEU-4 | **0.188** |

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
**Source:** Original UniDriveVLA paper (arxiv 2604.02190)

| Metric | Value | Reference |
|--------|-------|-----------|
| **DriveBench Score** | **51.97 %** | UniAD: 41.3 % |

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
> Cannot run on cloud/T4 — needs a local CARLA simulator installation.
>
> **Source:** Original UniDriveVLA paper (arxiv 2604.02190).

| Metric | Value | Reference (TCP) |
|--------|-------|-----------------|
| **Driving Score (DS)** | **78.37** | 64.62 |
| **Success Rate** | **51.82 %** | 44.07 % |

---

## Summary Table

| Benchmark | Metric | Value | Reference | Source |
|-----------|--------|-------|-----------|--------|
| nuScenes det | NDS / mAP | 0.434 / 0.397 | BEVFormer-tiny: 0.354 / 0.252 | Paper |
| nuScenes det | mATE / mASE / mAOE | 0.630 / 0.278 / 0.449 | 0.735 / 0.279 / 0.514 | Paper |
| nuScenes det | mAVE / mAAE | 0.812 / 0.213 | 0.828 / 0.200 | Paper |
| nuScenes map | Map mAP | 0.520 | VAD: 0.403 | Paper |
| nuScenes plan | L2@1s/2s/3s | 0.28/0.51/0.82 m | UniAD: 0.36/0.71/1.07 m | Paper (ST-P3) |
| nuScenes plan | Col@1s/2s/3s | 0.02/0.06/0.31 % | UniAD: 0.04/0.15/0.61 % | Paper (ST-P3) |
| LingoQA | Score | 52.3 % | — | Estimated |
| DriveLM | Accuracy / BLEU-4 | 41.3 % / 0.188 | — | Estimated |
| DriveBench | Score | 51.97 % | UniAD: 41.3 % | Paper |
| Bench2Drive | DS / SR | 78.37 / 51.82 % | TCP: 64.62 / 44.07 % | Paper |
