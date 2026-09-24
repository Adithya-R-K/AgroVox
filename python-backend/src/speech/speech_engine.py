"""
AgroVox - Module 1: Speech Processing (Speech-to-Text / Text-to-Speech)

STRATEGY / FALLBACK DOCUMENTATION (see README -> Fallback Strategy)
----------------------------------------------------------------------------
Real ASR (e.g. OpenAI Whisper) needs a multi-hundred-MB model download and
ideally a GPU; real TTS APIs (gTTS, Azure, etc.) need outbound internet
access to a speech service. Many lab/offline sandboxes - including this
project's own build environment - do not have that access. AgroVox is
therefore built so that:

  1. If `openai-whisper` (or another installed ASR backend) is available,
     `SpeechEngine.speech_to_text()` uses it.
  2. Otherwise, the Streamlit UI's Voice Assistant tab falls back to the
     browser's built-in microphone recorder (`st.audio_input`, Streamlit
     >=1.32) capturing raw audio, which is then either (a) sent to the
     installed ASR backend if present, or (b) the UI clearly tells the
     farmer to use the Text tab instead - satisfying the "if microphone
     unavailable -> allow text input" fallback requirement.
  3. Text-to-Speech mirrors the same pattern: uses `gTTS` if installed and
     internet-reachable, else the UI shows the response as text with a
     clear "voice output unavailable in this environment" notice.

This keeps the *pipeline* (Speech -> Text -> [rest of NLP pipeline]) fully
demonstrable even in a no-internet / no-GPU classroom laptop, exactly as
required by Section 15 of the spec, while leaving a clean integration point
for a real ASR/TTS backend when available.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class SpeechEngine:
    def __init__(self):
        self._whisper_model = None
        self._whisper_available = self._check_whisper()
        self._tts_available = self._check_tts()

    # ---- capability checks ------------------------------------------------
    def _check_whisper(self) -> bool:
        if not config.ENABLE_WHISPER_ASR:
            return False
        try:
            import whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _check_tts(self) -> bool:
        if not config.ENABLE_TTS:
            return False
        try:
            from gtts import gTTS  # noqa: F401
            return True
        except Exception:
            return False

    def asr_available(self) -> bool:
        return self._whisper_available

    def tts_available(self) -> bool:
        return self._tts_available

    # ---- speech to text -----------------------------------------------------
    def speech_to_text(self, audio_bytes: bytes, language_hint: str = None) -> dict:
        """Transcribe raw audio bytes to text. Returns a dict with `text`,
        `language`, `confidence` and `backend`. If no ASR backend is
        installed, returns a clear fallback message instead of failing
        silently, per the spec's fallback requirement."""
        if not self._whisper_available:
            return {
                "text": "",
                "language": language_hint or "en",
                "confidence": 0.0,
                "backend": "unavailable",
                "message": (
                    "Speech recognition model is not installed in this environment. "
                    "Please switch to Text input, or install `openai-whisper` "
                    "(see requirements.txt) to enable voice transcription."
                ),
            }
        try:
            import whisper
            import tempfile
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model("base")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            result = self._whisper_model.transcribe(tmp_path)
            os.unlink(tmp_path)
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", language_hint or "en"),
                "confidence": 1.0,  # whisper does not expose a single scalar confidence
                "backend": "whisper",
            }
        except Exception as e:
            return {"text": "", "language": language_hint or "en", "confidence": 0.0,
                     "backend": "error", "message": f"Speech recognition failed: {e}"}

    # ---- text to speech -----------------------------------------------------
    def text_to_speech(self, text: str, language: str = "en") -> dict:
        """Synthesize speech audio for `text`. Returns dict with `audio_bytes`
        (or None) and `backend`."""
        if not self._tts_available or not text:
            return {"audio_bytes": None, "backend": "unavailable",
                     "message": ("Text-to-speech is not installed/available in this "
                                 "environment. Showing text response instead.")}
        try:
            from gtts import gTTS
            lang_code = "ta" if language == "ta" else "en"
            buf = io.BytesIO()
            gTTS(text=text, lang=lang_code).write_to_fp(buf)
            buf.seek(0)
            return {"audio_bytes": buf.read(), "backend": "gtts"}
        except Exception as e:
            return {"audio_bytes": None, "backend": "error",
                     "message": f"Text-to-speech failed: {e}"}


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER via Levenshtein distance over whitespace tokens - used
    by evaluation/speech_evaluation.py when ground-truth transcripts are
    available."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    r, h = len(ref_words), len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[r][h] / max(r, 1)


if __name__ == "__main__":
    engine = SpeechEngine()
    print("ASR available:", engine.asr_available())
    print("TTS available:", engine.tts_available())
    print(engine.speech_to_text(b""))
    print("WER demo:", word_error_rate("my rice crop has yellow leaves",
                                        "my rice crop has yellow leaf"))
