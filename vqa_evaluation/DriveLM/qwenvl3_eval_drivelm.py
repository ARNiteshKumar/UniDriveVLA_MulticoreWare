"""DriveLM evaluation using Qwen3-VL via vLLM + Ray.

Sourced/adapted from:
  https://github.com/xiaomi-research/unidrivevla/blob/main/vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py

The DriveLM dataset is a JSON with structure:
  { "scene_token": { "key_frames": { "sample_token": {
        "QA": {"perception": [...], "prediction": [...], "planning": [...]},
        "image_paths": {"CAM_FRONT": ..., "CAM_FRONT_LEFT": ..., ...}
  }}}}

Usage:
    python vqa_evaluation/DriveLM/qwenvl3_eval_drivelm.py \
        --model_path  checkpoints/Qwen3-VL-2B-Instruct \
        --data_path   data/DriveLM/QA_dataset_nus_v1_val.json \
        --image_root  nuScenes/data/nuscenes \
        --output_path results/drivelm/preds.json \
        --num_gpus 1
"""
import os, json, argparse
import pandas as pd
from PIL import Image
from typing import List, Dict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ray
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "Generalist Autonomous Driving Agent. Answer questions about the driving scene "
    "using the provided multi-camera images. Be concise and accurate."
)

CAM_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]

VIEW_LABELS = {
    "CAM_FRONT":       "<FRONT_VIEW>",
    "CAM_FRONT_LEFT":  "<FRONT_LEFT_VIEW>",
    "CAM_FRONT_RIGHT": "<FRONT_RIGHT_VIEW>",
    "CAM_BACK":        "<BACK_VIEW>",
    "CAM_BACK_LEFT":   "<BACK_LEFT_VIEW>",
    "CAM_BACK_RIGHT":  "<BACK_RIGHT_VIEW>",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True)
    p.add_argument("--data_path",    required=True, help="DriveLM QA JSON")
    p.add_argument("--image_root",   required=True, help="nuScenes data root")
    p.add_argument("--output_path",  default="drivelm_preds.json")
    p.add_argument("--num_gpus",     type=int, default=1)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--max_model_len",  type=int, default=16000)
    return p.parse_args()


def flatten_qa(data: dict, image_root: str) -> List[Dict]:
    """Flatten hierarchical DriveLM JSON into a list of (item_id, question, image_paths)."""
    items = []
    for scene_token, scene in data.items():
        for sample_token, frame in scene.get("key_frames", {}).items():
            image_paths = frame.get("image_paths", {})
            full_paths = {
                cam: os.path.join(image_root, path.lstrip("/"))
                for cam, path in image_paths.items()
            }
            for qa_type, qa_list in frame.get("QA", {}).items():
                for i, qa in enumerate(qa_list):
                    q = qa.get("Q", "")
                    a = qa.get("A", "")
                    items.append({
                        "item_id":     f"{sample_token}_{qa_type}_{i}",
                        "sample_token": sample_token,
                        "qa_type":     qa_type,
                        "question":    q,
                        "gt_answer":   a,
                        "image_paths": full_paths,
                    })
    return items


class VLLMWorker:
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


def preprocess_block(df_block: pd.DataFrame, *, model_path: str) -> pd.DataFrame:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    rows = []
    for _, row in df_block.iterrows():
        try:
            # Load cameras in order
            loaded_imgs = []
            cam_content = []
            for cam in CAM_ORDER:
                path = row["image_paths"].get(cam, "")
                if path and os.path.exists(path):
                    img = Image.open(path).convert("RGB")
                    loaded_imgs.append(img)
                    cam_content.append({"type": "image", "image": img})

            messages = [{
                "role": "system",
                "content": SYSTEM_PROMPT,
            }, {
                "role": "user",
                "content": cam_content + [{"type": "text", "text": row["question"]}],
            }]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            imgs_proc, _ = process_vision_info(
                messages, image_patch_size=processor.image_processor.patch_size
            )
        except Exception as e:
            prompt, imgs_proc = "", []

        rows.append({
            "item_id":     row["item_id"],
            "sample_token": row["sample_token"],
            "qa_type":     row["qa_type"],
            "question":    row["question"],
            "gt_answer":   row["gt_answer"],
            "prompt":      prompt,
            "images":      imgs_proc,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"Loading DriveLM JSON: {args.data_path}")
    with open(args.data_path) as f:
        data = json.load(f)

    items = flatten_qa(data, args.image_root)
    print(f"Total QA pairs: {len(items)}")

    df = pd.DataFrame(items)
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
    results = final_df[["item_id", "sample_token", "qa_type",
                         "question", "gt_answer", "prediction"]].to_dict(orient="records")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved → {args.output_path}  ({len(results)} items)")


if __name__ == "__main__":
    main()
