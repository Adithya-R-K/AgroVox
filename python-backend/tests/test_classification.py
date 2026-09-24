"""Unit tests: intent classifier (trained TF-IDF + classical ML models)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.classification.classifier import IntentClassifier


def test_classifier_loads_if_trained():
    clf = IntentClassifier()
    if not clf.is_ready():
        import pytest
        pytest.skip("Model not yet trained - run train_classifier.py first")
    assert clf.model is not None


def test_predict_fertilizer_intent():
    clf = IntentClassifier()
    if not clf.is_ready():
        import pytest
        pytest.skip("Model not yet trained - run train_classifier.py first")
    result = clf.predict("What fertilizer is suitable for rice during the vegetative stage?")
    assert result["intent"] in config.INTENT_LABELS
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_returns_valid_label_for_any_text():
    clf = IntentClassifier()
    if not clf.is_ready():
        import pytest
        pytest.skip("Model not yet trained - run train_classifier.py first")
    result = clf.predict("asdkjaskjd random gibberish text")
    assert result["intent"] in config.INTENT_LABELS


def test_predict_empty_string_does_not_crash():
    clf = IntentClassifier()
    if not clf.is_ready():
        import pytest
        pytest.skip("Model not yet trained - run train_classifier.py first")
    result = clf.predict("")
    assert "intent" in result
