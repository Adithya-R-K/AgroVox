"""
AgroVox - Module 2: Machine Translation (Tamil <-> English)

STRATEGY / FALLBACK DOCUMENTATION (see also README.md -> Fallback Strategy)
----------------------------------------------------------------------------
A production AgroVox would call a neural multilingual MT model (e.g. a
Hugging Face NLLB/IndicTrans2 checkpoint) via `HFNeuralTranslator`, defined
below. That path needs internet access to download model weights and,
ideally, a GPU. Many student/lab environments (and this project's own build
sandbox) do NOT have that internet access, so AgroVox ships a fully local,
zero-download `DictionaryTranslator` fallback that is used by default:

  * It is trained/built directly from data/translations.csv (a curated
    agricultural Tamil<->English glossary/phrase table).
  * It performs phrase-table + word-level substitution with simple
    Tamil/English function-word handling, which is sufficient to keep the
    six-concept pipeline fully runnable end-to-end offline.
  * `Translator.translate()` automatically tries the neural backend first
    (if `transformers`+`torch` are installed AND a model can be loaded) and
    transparently falls back to the dictionary backend otherwise - the
    caller does not need to know which backend served the request; the
    returned dict includes `backend` for transparency in the UI.

This mirrors exactly the graceful-fallback requirement in the project
specification (Section 15).
"""
import os
import csv
import re
import sys
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class DictionaryTranslator:
    """Local, offline, phrase-table based Tamil<->English translator."""

    def __init__(self, glossary_csv: str = config.TRANSLATIONS_CSV):
        self.en_to_ta = {}
        self.ta_to_en = {}
        self._load(glossary_csv)

    def _load(self, path):
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                en = row["term_en"].strip().lower()
                ta = row["term_ta"].strip()
                if en and ta:
                    self.en_to_ta[en] = ta
                    self.ta_to_en[ta] = en
        # max phrase length (in whitespace-tokens) present in the glossary,
        # used to bound the greedy sliding-window phrase matcher below
        self._max_phrase_len = max(
            [len(p.split()) for p in list(self.en_to_ta) + list(self.ta_to_en)] or [1])

    def _phrase_translate(self, text: str, vocab: dict, lowercase_keys: bool) -> str:
        """Whitespace-tokenized, longest-phrase-first greedy substitution.

        NOTE ON CORRECTNESS: an earlier version of this translator used naive
        substring replacement on the raw string (`phrase in result` / regex
        sub), which had two real bugs, both fixed here:
          1. English substring bug: replacing the short word "is" would also
             corrupt words that merely *contain* "is", e.g. "this" ->
             "th<ta-word-for-is>" or "disease" being mangled mid-word.
          2. Tamil grapheme-splitting bug: Tamil vowel signs/virama are
             Unicode *combining marks*, which are not part of `\\w` in
             Python's regex engine, so any `\\b`-based word-boundary
             approach silently split Tamil letters apart from their
             combining marks (e.g. "பயிர்" broken into "பய" + "ர்").
        Operating on whitespace-delimited tokens (never `\\w+`/`\\b` on the
        Tamil side) avoids both problems and matches whole words/phrases
        exactly, only falling back to passthrough for genuinely unknown
        tokens - out-of-vocabulary text is left as-is rather than partially
        mangled.
        """
        tokens = text.split()
        n = len(tokens)
        output = []
        i = 0
        while i < n:
            matched = False
            max_len = min(self._max_phrase_len, n - i)
            for length in range(max_len, 0, -1):
                window = tokens[i:i + length]
                clean_window = [w.strip(string.punctuation) for w in window]
                if not all(clean_window):
                    continue
                key = " ".join(w.lower() for w in clean_window) if lowercase_keys \
                    else " ".join(clean_window)
                if key in vocab:
                    trailing = "".join(ch for ch in window[-1] if ch in string.punctuation)
                    output.append(vocab[key] + trailing)
                    i += length
                    matched = True
                    break
            if not matched:
                output.append(tokens[i])
                i += 1
        return " ".join(output)

    def translate_en_to_ta(self, text: str) -> str:
        return self._phrase_translate(text, self.en_to_ta, lowercase_keys=True)

    def translate_ta_to_en(self, text: str) -> str:
        return self._phrase_translate(text, self.ta_to_en, lowercase_keys=False)

    def coverage(self, text: str, lang: str) -> float:
        """Fraction of content words in `text` that exist in the glossary -
        used for a simple translation-quality proxy metric."""
        vocab = self.en_to_ta if lang == "en" else self.ta_to_en
        words = re.findall(r"[\w\u0B80-\u0BFF]+", text.lower())
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in vocab or any(w in k for k in vocab))
        return hits / len(words)


class HFNeuralTranslator:
    """Optional neural MT backend (requires `transformers`+`torch` and
    internet access to download a multilingual checkpoint on first use).
    Not used unless explicitly enabled AND the dependencies/model are
    available - see Translator.translate().
    """

    _MODEL_NAME = "facebook/nllb-200-distilled-600M"  # example; swap for IndicTrans2 if preferred

    def __init__(self):
        self._pipe = None
        self._available = False
        try:
            from transformers import pipeline  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline("translation", model=self._MODEL_NAME)

    def translate(self, text: str, src: str, tgt: str) -> str:
        self._load()
        result = self._pipe(text, src_lang=src, tgt_lang=tgt)
        return result[0]["translation_text"]


class Translator:
    """Unified translation facade used by the rest of the AgroVox pipeline."""

    def __init__(self, use_neural_if_available: bool = None):
        # Respect the AGROVOX_ENABLE_NEURAL_MT env flag (see config.py /
        # .env.example) by default, so a heavy neural checkpoint is never
        # downloaded/loaded unless a deployment has explicitly opted in -
        # `use_neural_if_available` can still override this for tests.
        if use_neural_if_available is None:
            use_neural_if_available = config.ENABLE_NEURAL_MT
        self.dict_translator = DictionaryTranslator()
        self.neural_translator = None
        if use_neural_if_available:
            try:
                nt = HFNeuralTranslator()
                if nt.is_available():
                    self.neural_translator = nt
            except Exception:
                self.neural_translator = None

    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        if not text or source_lang == target_lang:
            return {"translated_text": text, "backend": "identity",
                     "source_lang": source_lang, "target_lang": target_lang}

        if self.neural_translator is not None:
            try:
                translated = self.neural_translator.translate(text, source_lang, target_lang)
                return {"translated_text": translated, "backend": "neural-mt",
                         "source_lang": source_lang, "target_lang": target_lang}
            except Exception:
                pass  # fall through to dictionary backend

        if source_lang == "ta" and target_lang == "en":
            translated = self.dict_translator.translate_ta_to_en(text)
        elif source_lang == "en" and target_lang == "ta":
            translated = self.dict_translator.translate_en_to_ta(text)
        else:
            translated = text

        return {"translated_text": translated, "backend": "dictionary-fallback",
                 "source_lang": source_lang, "target_lang": target_lang}


if __name__ == "__main__":
    t = Translator()
    print(t.translate("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது", "ta", "en"))
    print(t.translate("what fertilizer is good for rice", "en", "ta"))
