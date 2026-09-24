"""
AgroVox - Evaluation: Named Entity Recognition (Module 4)

Computes entity-level Precision / Recall / F1 (exact character-span +
label match) of NEREngine against the hand-annotated held-out sentences in
data/ner_data.json ("eval_sentences"), per the spec's Phase 2 requirement.

Run: python evaluation/ner_metrics.py
"""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ner.ner_engine import NEREngine, _load_ner_data


def spans_match(a, b):
    return a["start"] == b["start"] and a["end"] == b["end"] and a["label"] == b["label"]


def evaluate():
    engine = NEREngine()
    data = _load_ner_data()
    eval_sentences = data["eval_sentences"]

    tp = fp = fn = 0
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for item in eval_sentences:
        text = item["text"]
        gold = item["entities"]
        pred = engine.extract(text)

        matched_gold = set()
        for p in pred:
            hit = False
            for i, g in enumerate(gold):
                if i in matched_gold:
                    continue
                if spans_match(p, g):
                    hit = True
                    matched_gold.add(i)
                    break
            if hit:
                tp += 1
                per_label[p["label"]]["tp"] += 1
            else:
                fp += 1
                per_label[p["label"]]["fp"] += 1
        for i, g in enumerate(gold):
            if i not in matched_gold:
                fn += 1
                per_label[g["label"]]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_label_metrics = {}
    for label, counts in per_label.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label_metrics[label] = {"precision": p, "recall": r, "f1": f, **counts}

    results = {
        "overall": {"precision": precision, "recall": recall, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn},
        "per_label": per_label_metrics,
        "num_eval_sentences": len(eval_sentences),
    }
    return results


def main():
    results = evaluate()
    print(json.dumps(results, indent=2))
    out_path = os.path.join(config.EVAL_RESULTS_DIR, "ner_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
