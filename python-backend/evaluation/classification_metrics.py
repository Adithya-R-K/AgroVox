"""
AgroVox - Evaluation: Text Classification (Module 3)

Trains + compares TF-IDF+LogReg / TF-IDF+SVM / TF-IDF+NaiveBayes on a
held-out test split of data/intents.csv, saves the best model, and writes:
  evaluation/results/classification_report.json
  evaluation/results/confusion_matrix.png
  evaluation/results/classifier_comparison.json  (also written by classifier.py)

Run: python evaluation/classification_metrics.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.classification.classifier import train_and_compare, save_best_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def main():
    result = train_and_compare()
    print("Model comparison (held-out test split):")
    print(result["comparison"])
    save_best_model(result)

    best_name = result["best_model_name"]
    best_model = result["models"][best_name]
    y_test = result["y_test"]
    X_test_vec = result["X_test_vec"]
    preds = best_model.predict(X_test_vec)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    report_path = os.path.join(config.EVAL_RESULTS_DIR, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump({"best_model": best_name, "report": report}, f, indent=2)
    print(f"Saved classification report -> {report_path}")

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title(f"AgroVox Intent Classifier Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(config.EVAL_RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix -> {cm_path}")


if __name__ == "__main__":
    main()
