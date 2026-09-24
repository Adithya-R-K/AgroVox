"""Unit tests: dictionary-based Tamil<->English translator.

These specifically guard against the two real bugs found and fixed during
the production audit:
  1. Substring corruption (e.g. "this" being mangled because it contains "is").
  2. Tamil grapheme-cluster splitting (combining marks separated from base
     letters by a naive \\b/\\w-based approach).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.translation.translator import DictionaryTranslator, Translator


def test_no_substring_corruption_on_short_words():
    t = DictionaryTranslator()
    out = t.translate_en_to_ta("this is a test")
    assert "this" in out, "word 'this' must not be mangled just because it contains 'is'"


def test_no_substring_corruption_disease_word():
    t = DictionaryTranslator()
    out = t.translate_en_to_ta("my crop has a disease")
    # "disease" contains "is" - must not be corrupted into a mixed-script mess
    assert "நோய்" in out  # "disease" correctly translated as a whole word
    assert "isஏeasee" not in out and "dஆகிறது" not in out


def test_tamil_graphemes_not_split():
    t = DictionaryTranslator()
    out = t.translate_ta_to_en("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது")
    # the untranslated compound word must survive completely intact,
    # not be split into fragments missing their vowel signs/virama
    assert "நெற்பயிரில்" in out


def test_known_phrase_translation():
    t = DictionaryTranslator()
    out = t.translate_en_to_ta("my crop leaves are turning yellow")
    assert "பயிர்" in out or "இலை" in out  # at least some known terms translated


def test_translator_facade_identity_when_same_language():
    tr = Translator()
    result = tr.translate("hello", "en", "en")
    assert result["translated_text"] == "hello"
    assert result["backend"] == "identity"


def test_translator_facade_roundtrip_backend_reported():
    tr = Translator()
    result = tr.translate("rice", "en", "ta")
    assert result["backend"] in ("dictionary-fallback", "neural-mt")
    assert result["translated_text"]


def test_empty_string_does_not_crash():
    tr = Translator()
    result = tr.translate("", "en", "ta")
    assert result["translated_text"] == ""
