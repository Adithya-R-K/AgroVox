"""
AgroVox - End-to-end training/build script.

Runs the full offline build pipeline in order:
  1. Generate knowledge base + intents + NER + QA datasets (if missing)
  2. Train & compare classical text classifiers, save the best one
  3. Build the NER EntityRuler pipeline (verifies it loads)
  4. Build the QA TF-IDF retrieval index
  5. Run all evaluation scripts and print a consolidated summary

Run:  python train_classifier.py
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import config


def _run(cmd, cwd=BASE_DIR):
    print(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_datasets():
    if not os.path.exists(config.CROPS_CSV):
        _run([sys.executable, "data/generate_knowledge_base.py"])
    if not os.path.exists(config.INTENTS_CSV):
        _run([sys.executable, "data/generate_intents.py"])
    if not os.path.exists(config.NER_JSON):
        _run([sys.executable, "data/generate_ner_data.py"])
    if not os.path.exists(config.QA_CSV):
        _run([sys.executable, "data/generate_qa_dataset.py"])


def main():
    print("AgroVox - Full build & training pipeline")
    ensure_datasets()

    _run([sys.executable, "evaluation/classification_metrics.py"])
    _run([sys.executable, "evaluation/ner_metrics.py"])
    _run([sys.executable, "evaluation/translation_metrics.py"])
    _run([sys.executable, "evaluation/qa_evaluation.py"])
    _run([sys.executable, "evaluation/speech_evaluation.py"])

    print("\nAll models trained/verified and evaluation reports written to "
          f"{config.EVAL_RESULTS_DIR}\nYou can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
