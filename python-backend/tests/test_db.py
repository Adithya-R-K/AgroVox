"""Unit tests: SQLite persistence layer.

Uses a temporary DB file (monkeypatched via config.DB_PATH) so tests never
touch the real agrovox.db used by the running app.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Redirect DB_PATH to a temp file BEFORE importing utils.db, since db.py
# reads config.DB_PATH at call-time via get_connection() (not cached at
# import time), so this monkeypatch is safe.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
config.DB_PATH = _tmp_db.name

from utils import db


def setup_module(module):
    db.init_db()


def test_upsert_and_get_farmer():
    db.upsert_farmer("test_farmer_1", name="Test Farmer", preferred_language="ta",
                      location="Coimbatore", crops_cultivated="Rice")
    farmer = db.get_farmer("test_farmer_1")
    assert farmer is not None
    assert farmer["name"] == "Test Farmer"
    assert farmer["preferred_language"] == "ta"


def test_get_farmer_unknown_returns_none():
    assert db.get_farmer("nonexistent_farmer_xyz") is None


def test_log_query_and_feedback():
    db.upsert_farmer("test_farmer_2")
    db.start_conversation("conv_1", "test_farmer_2")
    qid = db.log_query("conv_1", "test_farmer_2", "My rice crop has yellow leaves", "en",
                        None, "Crop Disease", 0.9, {"CROP": ["rice"]},
                        "Yellowing can indicate nutrient issues.", 0.8)
    assert isinstance(qid, int)
    db.log_feedback(qid, "helpful")

    history = db.get_conversation_history("conv_1")
    assert len(history) == 1
    assert history[0]["intent"] == "Crop Disease"


def test_get_all_queries_for_farmer_across_conversations():
    db.upsert_farmer("test_farmer_3")
    db.start_conversation("conv_a", "test_farmer_3")
    db.start_conversation("conv_b", "test_farmer_3")
    db.log_query("conv_a", "test_farmer_3", "q1", "en", None, "Soil", 0.5, {}, "a1", 0.5)
    db.log_query("conv_b", "test_farmer_3", "q2", "en", None, "Soil", 0.5, {}, "a2", 0.5)
    all_queries = db.get_all_queries_for_farmer("test_farmer_3")
    assert len(all_queries) == 2


def test_analytics_does_not_crash_with_no_data():
    analytics = db.get_analytics()
    assert "total_queries" in analytics
    assert analytics["total_queries"] >= 0


def teardown_module(module):
    try:
        os.unlink(_tmp_db.name)
    except OSError:
        pass
