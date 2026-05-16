"""LingoQA inference using Qwen3-VL via vLLM + Ray.

Sourced/adapted from:
  https://github.com/xiaomi-research/unidrivevla/blob/main/vqa_evaluation/LingoQA/infer_qwenvl3.py

Usage (single T4 GPU):
    python vqa_evaluation/LingoQA/infer_qwenvl3.py \
        --model_path  checkpoints/Qwen3-VL-2B-Instruct \
        --parquet_path vqa_evaluation/LingoQA/val.parquet \
        --image_root   data/LingoQA/images/val \
        --output_path  results/lingoqa/preds.csv \
        --num_gpus 1 --batch_size 4
"""
import os
import argparse
import pandas as pd
from PIL import Image
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ray
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "Generalist Autonomous Driving Agent\n"
    "Role: You are an advanced multimodal AI brain for an autonomous vehicle. "
    "Generate a safe 3-second trajectory (6 waypoints, 0.5 s interval): [(x1,y1), ..., (x6,y6)]. "
    "For QA/Reasoning: provide clear, step-by-step answers grounded in visual evidence. "
    "Always prioritise safety."
)

MODEL_PATH        = "/path/to/UniDriveVLA_nuScenes_hf"
IMAGE_ROOT        = "/path/to/LingoQA/images/val"
LINGOQA_TEST_PATH = "val.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   default=MODEL_PATH)
    p.add_argument("--parquet_path", default=LINGOQA_TEST_PATH)
    p.add_argument("--image_root",   default=IMAGE_ROOT)
    p.add_argument("--output_path",  default="lingoqa_preds.csv")
    p.add_argument("--num_gpus",     type=int,   default=1)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--max_model_len",  type=int, default=30000)
    return p.parse_args()


def load_image(path):
    return Image.open(path).convert("RGB")


def load_images_for_segment(image_root, segment_id, num_frames=5):
    seg_dir = os.path.join(image_root, segment_id)
    paths = [os.path.join(seg_dir, f"{i}.jpg") for i in range(num_frames)]
    with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as exe:
        imgs = list(exe.map(load_image, paths))
    return imgs


class VLLMPredictor:
    def __init__(self, model_path, gpu_memory_utilization, max_model_len):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        self.sampling = SamplingParams(temperature=0.01, top_p=0.001, max_tokens=256)

    def __call__(self, batch):
        inputs_all, valid_idx = [], []
        for i, (prompt, images) in enumerate(zip(batch["prompt"], batch["image_inputs"])):
            if prompt and images is not None:
                inputs_all.append({"prompt": prompt, "multi_modal_data": {"image": images}})
                valid_idx.append(i)

        answers = [""] * len(batch["prompt"])
        if inputs_all:
            outputs = self.llm.generate(inputs_all, sampling_params=self.sampling)
            for idx, out in zip(valid_idx, outputs):
                try:
                    answers[idx] = out.outputs[0].text
                except Exception:
                    answers[idx] = ""
        batch["answer"] = answers
        return batch


def _preprocess_block(df_block, *, image_root, model_path):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    rows = []
    for _, row in df_block.iterrows():
        qid, seg, question = row["question_id"], row["segment_id"], row["question"]
        try:
            images = load_images_for_segment(image_root, seg)
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                           + [{"type": "text",  "text":  question}],
            }]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            imgs_proc, _ = process_vision_info(
                messages,
                image_patch_size=processor.image_processor.patch_size,
            )
        except Exception:
            prompt, imgs_proc = "", None

        rows.append({"question_id": qid, "segment_id": seg,
                     "question": question, "prompt": prompt, "image_inputs": imgs_proc})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    df = pd.read_parquet(args.parquet_path)[["question_id", "segment_id", "question"]]
    print(f"Loaded {len(df)} LingoQA samples")

    ds = ray.data.from_pandas(df).repartition(max(args.num_gpus * 4, 1))

    ds_pre = ds.map_batches(
        _preprocess_block,
        fn_kwargs={"image_root": args.image_root, "model_path": args.model_path},
        batch_format="pandas",
        batch_size=args.batch_size,
    )
    ds_out = ds_pre.map_batches(
        VLLMPredictor,
        fn_constructor_kwargs={
            "model_path": args.model_path,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        },
        batch_format="pandas",
        batch_size=args.batch_size,
        concurrency=args.num_gpus,
        num_gpus=1,
    )

    final_df = ds_out.to_pandas()[["question_id", "segment_id", "question", "answer"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    final_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved → {args.output_path}  ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
