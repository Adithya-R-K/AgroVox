"""
AgroVox - Evaluation: Speech Processing (Module 1)

Speech recognition is evaluated via Word Error Rate (WER) between an ASR
transcript and a ground-truth reference. Since no audio corpus / ASR model
is available offline in this environment (see src/speech/speech_engine.py
fallback documentation), this script demonstrates the WER computation on a
synthetic transcript-vs-reference pair set and clearly reports whether a
real ASR backend is active.

Run: python evaluation/speech_evaluation.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.speech.speech_engine import SpeechEngine, word_error_rate

# (reference transcript, simulated ASR hypothesis) pairs illustrating the
# WER metric definition. Replace with real (audio -> whisper transcript)
# pairs once a microphone + ASR backend is available.
DEMO_PAIRS = [
    ("my rice crop has yellow leaves", "my rice crop has yellow leaf"),
    ("what fertilizer is good for tomato", "what fertilizer is good for tomatoes"),
    ("how often should i irrigate my field", "how often should i irrigate my field"),
    ("my cotton plant is wilting", "my cotton plants wilting"),
]


def main():
    engine = SpeechEngine()
    wers = [word_error_rate(ref, hyp) for ref, hyp in DEMO_PAIRS]
    avg_wer = sum(wers) / len(wers)

    result = {
        "asr_backend_available": engine.asr_available(),
        "tts_backend_available": engine.tts_available(),
        "note": (
            "No microphone/audio corpus is available in this build environment, so WER "
            "is demonstrated here on illustrative (reference, hypothesis) transcript "
            "pairs rather than real recorded audio. When `openai-whisper` is installed "
            "and real audio is captured via the Streamlit Voice Assistant tab, this "
            "script's DEMO_PAIRS list should be replaced by (ground_truth_transcript, "
            "whisper_output) pairs collected from actual farmer recordings."
        ),
        "pairs": [{"reference": r, "hypothesis": h, "wer": w}
                  for (r, h), w in zip(DEMO_PAIRS, wers)],
        "average_wer": avg_wer,
    }
    print(json.dumps(result, indent=2))
    out_path = os.path.join(config.EVAL_RESULTS_DIR, "speech_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
