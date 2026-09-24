"""Integration test: the full 6-concept AgroVox pipeline end-to-end,
including multi-turn context handling and edge cases surfaced during the
production audit (empty input, very long input, mixed-language input)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import config

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
config.DB_PATH = _tmp_db.name

from src.chatbot.chatbot_engine import AgroVoxPipeline, ChatbotSession
from utils import db

db.init_db()
_pipeline = AgroVoxPipeline()


def test_end_to_end_english_query():
    session = ChatbotSession(_pipeline, farmer_id="itest_farmer_1")
    result = session.ask("My rice crop has yellow leaves in Coimbatore.")
    assert result["intent"] == "Crop Disease"
    assert "rice" in [v.lower() for v in result["entities"].get("CROP", [])]
    assert result["answer"]


def test_multi_turn_context_carries_topic():
    session = ChatbotSession(_pipeline, farmer_id="itest_farmer_2")
    r1 = session.ask("My rice crop has yellow leaves.")
    r2 = session.ask("Vegetative stage.")
    # the short follow-up must stay on the previous topic (Crop Disease),
    # not jump to whatever the classifier guesses for two isolated words
    assert r2["effective_intent"] == "Crop Disease"
    assert r2["context_snapshot"]["growth_stage"] == "vegetative"
    assert r2["context_snapshot"]["CROP"] == "rice"


def test_end_to_end_tamil_query():
    session = ChatbotSession(_pipeline, farmer_id="itest_farmer_3")
    result = session.ask("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?")
    assert result["detected_language"] == "ta"
    assert result["translated_input_en"]
    assert result["answer"]  # answer translated back to Tamil


def test_empty_input_does_not_crash():
    session = ChatbotSession(_pipeline, farmer_id="itest_farmer_4")
    result = session.ask("")
    assert "answer" in result


def test_very_long_input_does_not_crash():
    session = ChatbotSession(_pipeline, farmer_id="itest_farmer_5")
    long_text = ("rice disease yellow leaves " * 100).strip()
    result = session.ask(long_text)
    assert "answer" in result


def test_sql_injection_style_text_is_safe():
    session = ChatbotSession(_pipeline, farmer_id="itest_farmer_6")
    result = session.ask("'; DROP TABLE queries; --")
    assert "answer" in result
    # DB must still be intact and queryable afterward
    analytics = db.get_analytics()
    assert analytics["total_queries"] >= 1


def teardown_module(module):
    try:
        os.unlink(_tmp_db.name)
    except OSError:
        pass
