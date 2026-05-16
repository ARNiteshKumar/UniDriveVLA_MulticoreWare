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

| Metric | Value | Reference baseline |
|--------|-------|--------------------|
| NDS ↑ | **0.434** | BEVFormer-tiny: 0.354 |
| mAP ↑ | **0.397** | BEVFormer-tiny: 0.252 |

### Online Map Prediction

| Metric | Value |
|--------|-------|
| Map mAP ↑ | **0.520** |

### Ego Planning (L2 & Collision — ST-P3 protocol)

| Horizon | L2 ↓ (m) | Collision ↓ (%) |
|---------|:--------:|:---------------:|
| 1 s | **0.28** | **0.02** |
| 2 s | **0.51** | **0.06** |
| 3 s | **0.82** | **0.31** |

**Reference (UniAD):** L2@1s=0.36 m, L2@3s=1.07 m, Col@3s=0.61 %

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

> **Note:** LingoQA scores are not reported in the original UniDriveVLA paper.
> The score below is an estimate from Qwen3-VL-2B zero-shot performance.

| Metric | Score |
|--------|-------|
| **LingoQA Score** | **~52 %** |

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

> **Note:** DriveLM scores are not reported in the original UniDriveVLA paper.
> Scores below are estimates.

| Metric | Score |
|--------|-------|
| Accuracy | ~41 % |
| BLEU-4 | ~0.19 |

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

| Metric | Score |
|--------|-------|
| **DriveBench Score** | **51.97 %** |

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

| Metric | Score |
|--------|-------|
| **Driving Score (DS)** | **78.37** |
| **Success Rate** | **51.82 %** |

---

## Summary Table

| Benchmark | Metric | Value | Source |
|-----------|--------|-------|--------|
| nuScenes det | NDS / mAP | 0.434 / 0.397 | Paper |
| nuScenes map | Map mAP | 0.520 | Paper |
| nuScenes plan | L2@3s / Col@3s | 0.82 m / 0.31 % | Paper (ST-P3) |
| LingoQA | Score | ~52 % | Estimate |
| DriveLM | Accuracy / BLEU-4 | ~41 % / ~0.19 | Estimate |
| DriveBench | Score | 51.97 % | Paper |
| Bench2Drive | DS / SR | 78.37 / 51.82 % | Paper |
