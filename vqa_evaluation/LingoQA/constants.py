from enum import Enum

LINGOQA_TEST = "val.parquet"
LINGO_JUDGE = "wayveai/Lingo-Judge"

class Keys(str, Enum):
    question_id = "question_id"
    segment_id  = "segment_id"
    question    = "question"
    answer      = "answer"
    references  = "references"
    prediction  = "prediction"
    max_score   = "max_score"
    score       = "score"
    probability = "probability"
    correct     = "correct"
