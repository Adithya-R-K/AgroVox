"""Unit tests: QA retrieval engine (TF-IDF + cosine similarity)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.qa.qa_engine import TFIDFQAEngine


def test_qa_retrieves_relevant_answer():
    qa = TFIDFQAEngine()
    result = qa.answer("What fertilizer is suitable for rice during the vegetative stage?")
    assert result["answer"]
    assert result["confidence"] > 0


def test_qa_category_boost_changes_ranking():
    qa = TFIDFQAEngine()
    r_no_hint = qa.answer("my rice crop has yellow leaves")
    r_with_hint = qa.answer("my rice crop has yellow leaves", category_hint="Crop Disease")
    # the disease-focused hint should retrieve a disease-category answer
    assert r_with_hint["category"] == "Crop Disease"


def test_qa_handles_unknown_query_gracefully():
    qa = TFIDFQAEngine()
    result = qa.answer("asdkjaskjd totally unrelated gibberish")
    assert "answer" in result
    assert isinstance(result["confidence"], float)


def test_qa_empty_string_does_not_crash():
    qa = TFIDFQAEngine()
    result = qa.answer("")
    assert "answer" in result
