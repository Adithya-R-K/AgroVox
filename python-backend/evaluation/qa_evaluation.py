"""
AgroVox - Evaluation: Question Answering (Module 5)

Evaluates retrieval accuracy of the TF-IDF QA engine: for each question in
data/qa_dataset.csv, we paraphrase it slightly (simulate a real farmer not
typing the exact catalog wording) and check whether the QA engine still
retrieves the correct original answer as its #1 or top-3 result
(Top-1 / Top-3 accuracy), plus mean reciprocal rank (MRR).

Run: python evaluation/qa_evaluation.py
"""
import os
import sys
import json
import re
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.qa.qa_engine import TFIDFQAEngine

random.seed(config.RANDOM_SEED)

PARAPHRASE_PREFIXES = [
    "Can you tell me {q}",
    "I want to know {q}",
    "Please explain {q}",
    "{q}",
    "Quick question - {q}",
]


def paraphrase(question: str) -> str:
    q = question[0].lower() + question[1:] if question else question
    q = q.rstrip("?")
    template = random.choice(PARAPHRASE_PREFIXES)
    return (template.format(q=q) + "?").strip()


def main():
    qa = TFIDFQAEngine()
    df = qa.df

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    n = len(df)

    for i, row in df.iterrows():
        query = paraphrase(row["question"])
        result = qa.answer(query, top_k=3)
        candidates = [result["matched_question"]] + [a["question"] for a in result["alternatives"]]
        rank = None
        for r, cand in enumerate(candidates, start=1):
            if cand == row["question"]:
                rank = r
                break
        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    metrics = {
        "num_questions": n,
        "top1_accuracy": hits_at_1 / n,
        "top3_accuracy": hits_at_3 / n,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / n,
    }
    print(json.dumps(metrics, indent=2))
    out_path = os.path.join(config.EVAL_RESULTS_DIR, "qa_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
