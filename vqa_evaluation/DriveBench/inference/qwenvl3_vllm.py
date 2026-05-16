"""DriveBench robustness inference using Qwen3-VL via vLLM.

Sourced/adapted from:
  https://github.com/xiaomi-research/unidrivevla/blob/main/vqa_evaluation/DriveBench/inference/qwenvl3_vllm.py

DriveBench applies image corruptions to nuScenes-like data to test VQA robustness.
Dataset root structure expected:
  data_root/
    clean/              ← clean images
    corruption_<name>/  ← corrupted images per corruption type
    questions.json      ← [{id, question, image_paths, gt_answer, corruption_type}]

Usage:
    python vqa_evaluation/DriveBench/inference/qwenvl3_vllm.py \
        --model_path  checkpoints/Qwen3-VL-2B-Instruct \
        --data_root   data/DriveBench \
        --output_path results/drivebench/preds.json \
        --num_gpus 1
"""
import os, json, argparse
import pandas as pd
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ray
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You are a driving scene understanding AI. Answer the question about the provided "
    "driving images concisely and accurately."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True)
    p.add_argument("--data_root",    required=True)
    p.add_argument("--output_path",  default="drivebench_preds.json")
    p.add_argument("--num_gpus",     type=int, default=1)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--max_model_len",  type=int, default=16000)
    return p.parse_args()


def load_questions(data_root):
    qpath = os.path.join(data_root, "questions.json")
    if not os.path.exists(qpath):
        raise FileNotFoundError(
            f"questions.json not found at {qpath}.\n"
            "Download DriveBench from: https://github.com/drive-bench/toolkit"
        )
    with open(qpath) as f:
        return json.load(f)


class VLLMWorker:
    def __init__(self, model_path, gpu_memory_utilization, max_model_len):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        self.sampling = SamplingParams(temperature=0.01, top_p=0.001, max_tokens=128)

    def __call__(self, batch):
        inputs_all, valid_idx = [], []
        for i, (prompt, images) in enumerate(zip(batch["prompt"], batch["images"])):
            if prompt and images:
                inputs_all.append({"prompt": prompt, "multi_modal_data": {"image": images}})
                valid_idx.append(i)

        answers = [""] * len(batch["prompt"])
        if inputs_all:
            outs = self.llm.generate(inputs_all, sampling_params=self.sampling)
            for idx, out in zip(valid_idx, outs):
                try:
                    answers[idx] = out.outputs[0].text
                except Exception:
                    answers[idx] = ""
        batch["prediction"] = answers
        return batch


def preprocess_block(df_block, *, model_path):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    rows = []
    for _, row in df_block.iterrows():
        try:
            imgs_content = []
            loaded = []
            for path in (row["image_paths"] or []):
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB")
                    loaded.append(img)
                    imgs_content.append({"type": "image", "image": img})
            messages = [{
                "role": "system", "content": SYSTEM_PROMPT,
            }, {
                "role": "user",
                "content": imgs_content + [{"type": "text", "text": row["question"]}],
            }]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            imgs_proc, _ = process_vision_info(
                messages, image_patch_size=processor.image_processor.patch_size
            )
        except Exception:
            prompt, imgs_proc = "", []

        rows.append({
            "id":              row["id"],
            "question":        row["question"],
            "gt_answer":       row["gt_answer"],
            "corruption_type": row.get("corruption_type", "clean"),
            "prompt":          prompt,
            "images":          imgs_proc,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    questions = load_questions(args.data_root)
    print(f"DriveBench: {len(questions)} questions across corruption types")

    df = pd.DataFrame(questions)
    ds = ray.data.from_pandas(df).repartition(max(args.num_gpus * 4, 1))

    ds_pre = ds.map_batches(
        preprocess_block,
        fn_kwargs={"model_path": args.model_path},
        batch_format="pandas",
        batch_size=args.batch_size,
    )
    ds_out = ds_pre.map_batches(
        VLLMWorker,
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

    final_df = ds_out.to_pandas()
    results  = final_df[["id", "question", "gt_answer", "corruption_type", "prediction"]].to_dict(
        orient="records"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DriveBench predictions saved → {args.output_path}  ({len(results)} items)")


if __name__ == "__main__":
    main()
