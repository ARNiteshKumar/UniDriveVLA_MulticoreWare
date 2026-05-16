#!/bin/bash
# ============================================================
# UniDriveVLA — nuScenes Mini  |  Master Benchmark Runner
#
# Usage:
#   bash scripts/run_benchmarks.sh [OPTIONS]
#
# Options:
#   --checkpoint PATH   Path to Stage 3 checkpoint .pth
#   --vlm-path PATH     Path to Qwen3-VL-2B-Instruct (default: env VLM_PRETRAINED_PATH)
#   --vae-path PATH     Path to occvae_latest.pth     (default: env OCCWORLD_VAE_PATH)
#   --num-gpus N        Number of GPUs (default: 1)
#   --benchmarks LIST   Comma-separated: nuscenes,lingoqa,drivelm,drivebench (default: all)
#   --output-dir DIR    Where to save results (default: results/)
#
# Example (single T4 GPU):
#   export VLM_PRETRAINED_PATH=checkpoints/Qwen3-VL-2B-Instruct
#   export OCCWORLD_VAE_PATH=checkpoints/occvae_latest.pth
#   bash scripts/run_benchmarks.sh \
#       --checkpoint checkpoints/stage3/mp_rank_00_model_states.pt \
#       --num-gpus 1
# ============================================================
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── defaults ──────────────────────────────────────────────────
CHECKPOINT="${STAGE3_CKPT:-}"
VLM_PATH="${VLM_PRETRAINED_PATH:-checkpoints/Qwen3-VL-2B-Instruct}"
VAE_PATH="${OCCWORLD_VAE_PATH:-checkpoints/occvae_latest.pth}"
NUM_GPUS=1
BENCHMARKS="nuscenes,lingoqa,drivelm,drivebench"
OUTPUT_DIR="results"

# ── parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --vlm-path)   VLM_PATH="$2";   shift 2;;
        --vae-path)   VAE_PATH="$2";   shift 2;;
        --num-gpus)   NUM_GPUS="$2";   shift 2;;
        --benchmarks) BENCHMARKS="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ── validate ──────────────────────────────────────────────────
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required."
    echo "  Run: bash scripts/download_checkpoints.sh"
    exit 1
fi
if [[ ! -f "$CHECKPOINT" && ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi
export VLM_PRETRAINED_PATH="$VLM_PATH"
export OCCWORLD_VAE_PATH="$VAE_PATH"
export PYTHONPATH="${REPO_ROOT}/nuScenes:${REPO_ROOT}:$PYTHONPATH"

echo "=============================================="
echo " UniDriveVLA Benchmark Suite — nuScenes Mini"
echo "=============================================="
echo "  Checkpoint : $CHECKPOINT"
echo "  VLM path   : $VLM_PATH"
echo "  VAE path   : $VAE_PATH"
echo "  GPUs       : $NUM_GPUS"
echo "  Benchmarks : $BENCHMARKS"
echo "  Output dir : $OUTPUT_DIR"
echo "----------------------------------------------"

SUMMARY_FILE="${OUTPUT_DIR}/benchmark_summary.txt"
echo "UniDriveVLA Benchmark Results — $(date)" > "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"

run_if_enabled() {
    local name="$1"
    echo "$BENCHMARKS" | grep -qw "$name"
}

# ── 1. nuScenes Open-Loop Evaluation ─────────────────────────
if run_if_enabled "nuscenes"; then
    echo ""
    echo "==> [1/4] nuScenes Open-Loop Evaluation (det + map + planning)"
    CONFIG="nuScenes/projects/configs/UniDriveVLA/unidrivevla_mini_stage2.py"
    NUSCENES_OUT="${OUTPUT_DIR}/nuscenes"
    mkdir -p "$NUSCENES_OUT"

    bash nuScenes/tools/dist_eval.sh \
        "$CONFIG" \
        "$CHECKPOINT" \
        "$NUM_GPUS" \
        --out "${NUSCENES_OUT}/results.json" \
        --eval bbox map motion planning 2>&1 | tee "${NUSCENES_OUT}/eval.log"

    # Parse and summarise
    python nuScenes/tools/evaluation/nuscenes_eval.py \
        --result-path "${NUSCENES_OUT}/results.json" \
        --output-dir  "${NUSCENES_OUT}" \
        --version v1.0-mini \
        --dataroot nuScenes/data/nuscenes 2>&1 | tee -a "${NUSCENES_OUT}/eval.log"

    python nuScenes/tools/evaluation/planning_eval.py \
        --result-path "${NUSCENES_OUT}/results.json" \
        --output-dir  "${NUSCENES_OUT}" 2>&1 | tee -a "${NUSCENES_OUT}/eval.log"

    echo "" >> "$SUMMARY_FILE"
    echo "--- nuScenes ---" >> "$SUMMARY_FILE"
    grep -E "(NDS|mAP|L2|Collision|map IoU)" "${NUSCENES_OUT}/eval.log" 2>/dev/null \
        >> "$SUMMARY_FILE" || echo "  (see ${NUSCENES_OUT}/eval.log)" >> "$SUMMARY_FILE"
fi

# ── 2. LingoQA (all 500 samples) ─────────────────────────────
if run_if_enabled "lingoqa"; then
    echo ""
    echo "==> [2/4] LingoQA Evaluation (500 samples)"
    LINGOQA_OUT="${OUTPUT_DIR}/lingoqa"
    mkdir -p "$LINGOQA_OUT"
    PARQUET="vqa_evaluation/LingoQA/val.parquet"

    if [[ ! -f "$PARQUET" ]]; then
        echo "ERROR: LingoQA val.parquet not found at $PARQUET"
        echo "  Run: bash scripts/download_checkpoints.sh"
    else
        python vqa_evaluation/LingoQA/infer_qwenvl3.py \
            --model_path    "$VLM_PATH" \
            --parquet_path  "$PARQUET" \
            --image_root    "data/LingoQA/images/val" \
            --output_path   "${LINGOQA_OUT}/preds.csv" \
            --num_gpus      "$NUM_GPUS" \
            --batch_size    4 2>&1 | tee "${LINGOQA_OUT}/infer.log"

        python vqa_evaluation/LingoQA/evaluate.py \
            --predictions_path "${LINGOQA_OUT}/preds.csv" \
            --batch_size 32 2>&1 | tee "${LINGOQA_OUT}/score.log"

        echo "" >> "$SUMMARY_FILE"
        echo "--- LingoQA ---" >> "$SUMMARY_FILE"
        grep -E "(LingoQA|score|accuracy)" "${LINGOQA_OUT}/score.log" 2>/dev/null \
            >> "$SUMMARY_FILE" || echo "  (see ${LINGOQA_OUT}/score.log)" >> "$SUMMARY_FILE"
    fi
fi

# ── 3. DriveLM Evaluation ─────────────────────────────────────
if run_if_enabled "drivelm"; then
    echo ""
    echo "==> [3/4] DriveLM Evaluation"
    DRIVELM_OUT="${OUTPUT_DIR}/drivelm"
    mkdir -p "$DRIVELM_OUT"
    DRIVELM_JSON="${DRIVELM_JSON:-data/DriveLM/QA_dataset_nus_v1_val.json}"

    if [[ ! -f "$DRIVELM_JSON" ]]; then
        echo "WARNING: DriveLM dataset not found at $DRIVELM_JSON"
        echo "  Download from: https://github.com/OpenDriveLab/DriveLM"
        echo "  Set env: export DRIVELM_JSON=/path/to/QA_dataset_nus_v1_val.json"
    else
        python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
            --model_path  "$VLM_PATH" \
            --data_path   "$DRIVELM_JSON" \
            --image_root  "nuScenes/data/nuscenes" \
            --output_path "${DRIVELM_OUT}/preds.json" \
            --num_gpus    "$NUM_GPUS" 2>&1 | tee "${DRIVELM_OUT}/infer.log"

        python vqa_evaluation/DriveLM/score_drivelm.py \
            --pred_path "${DRIVELM_OUT}/preds.json" \
            --gt_path   "$DRIVELM_JSON" \
            --output    "${DRIVELM_OUT}/scores.json" 2>&1 | tee "${DRIVELM_OUT}/score.log"

        echo "" >> "$SUMMARY_FILE"
        echo "--- DriveLM ---" >> "$SUMMARY_FILE"
        grep -E "(Accuracy|BLEU|CIDEr|DriveLM)" "${DRIVELM_OUT}/score.log" 2>/dev/null \
            >> "$SUMMARY_FILE" || echo "  (see ${DRIVELM_OUT}/score.log)" >> "$SUMMARY_FILE"
    fi
fi

# ── 4. DriveBench Evaluation ──────────────────────────────────
if run_if_enabled "drivebench"; then
    echo ""
    echo "==> [4/4] DriveBench Robustness Evaluation"
    DRIVEBENCH_OUT="${OUTPUT_DIR}/drivebench"
    mkdir -p "$DRIVEBENCH_OUT"
    DRIVEBENCH_ROOT="${DRIVEBENCH_ROOT:-data/DriveBench}"

    if [[ ! -d "$DRIVEBENCH_ROOT" ]]; then
        echo "WARNING: DriveBench dataset not found at $DRIVEBENCH_ROOT"
        echo "  Download from: https://github.com/drive-bench/toolkit"
        echo "  Set env: export DRIVEBENCH_ROOT=/path/to/drivebench"
    else
        python vqa_evaluation/DriveBench/inference/qwenvl3_vllm.py \
            --model_path  "$VLM_PATH" \
            --data_root   "$DRIVEBENCH_ROOT" \
            --output_path "${DRIVEBENCH_OUT}/preds.json" \
            --num_gpus    "$NUM_GPUS" 2>&1 | tee "${DRIVEBENCH_OUT}/infer.log"

        python vqa_evaluation/DriveBench/eval_drivebench.py \
            --pred_path "${DRIVEBENCH_OUT}/preds.json" \
            --data_root "$DRIVEBENCH_ROOT" \
            --output    "${DRIVEBENCH_OUT}/scores.json" 2>&1 | tee "${DRIVEBENCH_OUT}/score.log"

        echo "" >> "$SUMMARY_FILE"
        echo "--- DriveBench ---" >> "$SUMMARY_FILE"
        grep -E "(mPC|rPC|accuracy|Corruption)" "${DRIVEBENCH_OUT}/score.log" 2>/dev/null \
            >> "$SUMMARY_FILE" || echo "  (see ${DRIVEBENCH_OUT}/score.log)" >> "$SUMMARY_FILE"
    fi
fi

echo ""
echo "=============================================="
echo " BENCHMARK COMPLETE"
echo " Summary saved to: $SUMMARY_FILE"
echo "=============================================="
cat "$SUMMARY_FILE"
