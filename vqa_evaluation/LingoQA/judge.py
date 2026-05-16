"""LingoJudge — textual classifier for LingoQA benchmark.

Sourced from: https://github.com/xiaomi-research/unidrivevla/blob/main/vqa_evaluation/LingoQA/judge.py
"""
import torch
from torch import nn
from tqdm import trange
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constants import LINGO_JUDGE, Keys


class LingoJudge(nn.Module):
    """Evaluates truthfulness of an answer on the LingoQA benchmark."""

    def __init__(self, pretrained_model: str = LINGO_JUDGE):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model
        ).eval()

    @torch.inference_mode()
    def forward(self, question: str, references: List[str], prediction: str):
        device = next(self.parameters()).device
        texts = [
            f"{self.tokenizer.cls_token}\nQuestion: {question}\nAnswer: {a_gt}\nStudent: {prediction}"
            for a_gt in references
        ]
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = self.model(**encoded)
        return output.logits.squeeze(-1)

    def compute(
        self,
        questions: List[str],
        references: List[List[str]],
        predictions: List[str],
    ) -> torch.Tensor:
        max_scores = []
        for idx, question in enumerate(questions):
            refs_proc = [self.preprocess(r) for r in references[idx]]
            pred_proc = self.preprocess(predictions[idx])
            scores = self.forward(question, refs_proc, pred_proc)
            max_scores.append(max(scores))
        return torch.tensor(max_scores)

    def preprocess(self, string: str) -> str:
        return str(string).lower().strip()
