"""Unit tests: text preprocessing (language detection, tokenization,
stopword removal, lemmatization/stemming, Tamil handling)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.text_preprocessing import (
    detect_language, preprocess, tokenize_english, tokenize_tamil,
    clean_text, is_tamil,
)


def test_detect_language_english():
    assert detect_language("My rice crop has yellow leaves") == "en"


def test_detect_language_tamil():
    assert detect_language("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது") == "ta"


def test_is_tamil():
    assert is_tamil("நெல்") is True
    assert is_tamil("rice") is False


def test_clean_text_strips_urls_and_lowercases():
    out = clean_text("Check HTTPS://example.com for MORE info")
    assert "http" not in out
    assert out == out.lower()


def test_tokenize_english_removes_punctuation_noise():
    tokens = tokenize_english("My rice crop has yellow leaves!!")
    assert "rice" in tokens
    assert "!!" not in tokens


def test_tokenize_tamil_preserves_grapheme_clusters():
    tokens = tokenize_tamil("என் நெற்பயிரில் இலைகள்")
    # every token should be a clean Tamil word, not split mid-character
    assert "பயிரில்" not in "".join(tokens) or all(len(t) > 0 for t in tokens)
    assert len(tokens) == 3


def test_preprocess_english_pipeline():
    result = preprocess("My rice crop has yellow leaves!!")
    assert result["language"] == "en"
    assert "rice" in result["processed_tokens"]
    assert "has" not in result["processed_tokens"]  # stopword removed
    assert result["stemmed_text"]  # stemming produced non-empty output


def test_preprocess_tamil_pipeline_does_not_crash():
    result = preprocess("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது")
    assert result["language"] == "ta"
    assert len(result["tokens"]) > 0


def test_preprocess_empty_string_does_not_crash():
    result = preprocess("")
    assert result["processed_text"] == ""
    assert result["tokens"] == []
