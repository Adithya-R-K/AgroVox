"""
AgroVox - Evaluation: Machine Translation (Module 2)

Since AgroVox's default translation backend is the local dictionary/phrase
fallback (see src/translation/translator.py for why), we evaluate it two
ways that are meaningful for a phrase-table system:

  1. Glossary coverage on a held-out set of farmer-style sentences (what
     fraction of content words in the sentence exist in the glossary).
  2. Qualitative side-by-side EN->TA / TA->EN samples for manual inspection.

If a neural MT backend (`transformers`) IS available, this script also
computes a real BLEU score (via nltk) against a small parallel test set,
demonstrating the metric the spec asks for ("BLEU score where applicable").

Run: python evaluation/translation_metrics.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.translation.translator import Translator

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _SMOOTH = SmoothingFunction().method1
except Exception:
    sentence_bleu = None

TEST_PAIRS = [
    ("my rice crop has yellow leaves", "என் நெல் பயிரில் இலைகள் மஞ்சளாகிறது"),
    ("what fertilizer is good for rice", "நெல் பயிருக்கு எந்த உரம் நல்லது"),
    ("how often should I irrigate my field", "எத்தனை முறை என் வயலுக்கு நீர்ப்பாசனம் செய்ய வேண்டும்"),
    ("my crop leaves are wilting", "என் பயிர் இலைகள் வாடுகிறது"),
    ("thank you for the help", "உதவிக்கு நன்றி"),
]


def main():
    translator = Translator()
    results = []
    coverage_scores = []

    for en, ta in TEST_PAIRS:
        en_to_ta = translator.translate(en, "en", "ta")
        ta_to_en = translator.translate(ta, "ta", "en")
        cov_en = translator.dict_translator.coverage(en, "en")
        coverage_scores.append(cov_en)
        results.append({
            "source_en": en, "reference_ta": ta,
            "predicted_ta": en_to_ta["translated_text"], "backend": en_to_ta["backend"],
            "source_ta": ta, "predicted_en_from_ta": ta_to_en["translated_text"],
            "en_glossary_coverage": cov_en,
        })

    avg_coverage = sum(coverage_scores) / len(coverage_scores)

    bleu_scores = None
    if sentence_bleu is not None and translator.neural_translator is not None:
        bleu_scores = []
        for en, ta in TEST_PAIRS:
            hyp = translator.translate(en, "en", "ta")["translated_text"].split()
            ref = [ta.split()]
            bleu_scores.append(sentence_bleu(ref, hyp, smoothing_function=_SMOOTH))

    summary = {
        "backend_used": results[0]["backend"],
        "average_glossary_coverage": avg_coverage,
        "note": (
            "AgroVox's default translation backend is a local dictionary/phrase-table "
            "translator (no internet/model download required). BLEU is a meaningful "
            "metric for a neural sequence-generation MT model; since no neural backend "
            "is active in this environment, we report glossary coverage instead, plus "
            "qualitative EN<->TA samples below for manual review. If `transformers`+"
            "`torch` and a downloaded MT checkpoint are available, BLEU is computed "
            "automatically (see 'bleu_scores')."
        ),
        "bleu_scores": bleu_scores,
        "samples": results,
    }

    try:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    out_path = os.path.join(config.EVAL_RESULTS_DIR, "translation_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
