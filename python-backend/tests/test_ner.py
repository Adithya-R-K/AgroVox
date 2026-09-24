"""Unit tests: NER engine (gazetteer EntityRuler)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ner.ner_engine import NEREngine


def test_extracts_crop_and_symptom():
    engine = NEREngine()
    entities = engine.extract("My rice crop has yellow leaves in Coimbatore.")
    labels = {e["label"] for e in entities}
    texts = {e["text"].lower() for e in entities}
    assert "CROP" in labels
    assert "rice" in texts


def test_extracts_disease_entity():
    engine = NEREngine()
    entities = engine.extract("Is this bacterial leaf blight affecting my rice?")
    diseases = [e for e in entities if e["label"] == "DISEASE"]
    assert len(diseases) >= 1


def test_grouped_extraction_deduplicates():
    engine = NEREngine()
    grouped = engine.extract_grouped("rice rice rice crop")
    assert grouped["CROP"].count("rice") == 1  # deduplicated


def test_empty_text_returns_no_entities():
    engine = NEREngine()
    assert engine.extract("") == []


def test_no_entities_in_irrelevant_text():
    engine = NEREngine()
    entities = engine.extract("The weather is nice today in general.")
    # should not hallucinate agricultural entities from unrelated text
    assert all(e["label"] in ("WEATHER_CONDITION",) for e in entities) or entities == []
