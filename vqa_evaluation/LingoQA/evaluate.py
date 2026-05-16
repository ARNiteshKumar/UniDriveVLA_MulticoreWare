"""LingoQA evaluation using LingoJudge.

Sourced/adapted from:
  https://github.com/xiaomi-research/unidrivevla/blob/main/vqa_evaluation/LingoQA/evaluate.py

Usage:
    python vqa_evaluation/LingoQA/evaluate.py \
        --predictions_path results/lingoqa/preds.csv \
        --batch_size 32
"""
import os, sys, click, torch
import pandas as pd
from datasets import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from judge import LingoJudge
from constants import LINGO_JUDGE, Keys, LINGOQA_TEST


def evaluate_question(batch, metric, device):
    questions   = batch[Keys.question]
    references  = batch[Keys.references]
    predictions = batch[Keys.prediction]

    scores = metric.compute(
        questions=questions,
        references=references,
        predictions=predictions,
    ).to(device)

    batch[Keys.score]       = scores.tolist()
    batch[Keys.probability] = torch.sigmoid(scores).tolist()
    batch[Keys.correct]     = (torch.sigmoid(scores) > 0.5).tolist()
    return batch


def select_correct(example):
    return example[Keys.correct]


@click.command()
@click.option("--predictions_path", required=True, help="CSV from infer_qwenvl3.py")
@click.option("--batch_size",       default=32,    help="Batch size for judge")
@click.option("--parquet_path",     default=LINGOQA_TEST, help="val.parquet reference file")
def evaluate(predictions_path, batch_size, parquet_path):
    # Resolve parquet_path relative to this script if not absolute
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.join(os.path.dirname(__file__), parquet_path)

    print(f"Loading references from: {parquet_path}")
    references_df = pd.read_parquet(parquet_path)

    print(f"Loading predictions from: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    predictions_df = predictions_df.rename(columns={"answer": Keys.prediction})

    # Merge on question_id
    merged = references_df.merge(
        predictions_df[["question_id", Keys.prediction]],
        on="question_id",
        how="inner",
    )
    print(f"Matched {len(merged)} / {len(references_df)} questions")
    if len(merged) == 0:
        print("ERROR: No matching question_ids found. Check prediction CSV columns.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LingoJudge ({LINGO_JUDGE}) on {device}")
    metric = LingoJudge(pretrained_model=LINGO_JUDGE).to(device)

    dataset = Dataset.from_pandas(merged)
    dataset = dataset.map(
        evaluate_question,
        fn_kwargs={"metric": metric, "device": device},
        batched=True,
        batch_size=batch_size,
    )

    correct_count = sum(dataset[Keys.correct])
    total_count   = len(dataset)
    lingoqa_score = correct_count / total_count * 100.0

    print("\n" + "=" * 50)
    print(f"  LingoQA Score : {lingoqa_score:.2f}%")
    print(f"  Correct       : {correct_count} / {total_count}")
    print("=" * 50)

    # Per question-type breakdown (if 'type' column present)
    if "type" in dataset.column_names:
        df_out = dataset.to_pandas()
        for qtype, grp in df_out.groupby("type"):
            acc = grp[Keys.correct].mean() * 100
            print(f"  {qtype:<25} : {acc:.1f}%  (n={len(grp)})")

    return lingoqa_score


if __name__ == "__main__":
    evaluate()
