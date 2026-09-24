"""
AgroVox - SQLite persistence layer.

Stores farmer profiles, conversation history, per-query NLP pipeline
results (language, intent, entities), and thumbs-up/down feedback, backing
the Farmer Profile page and the NLP Analytics dashboard.
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS farmers (
    farmer_id TEXT PRIMARY KEY,
    name TEXT,
    preferred_language TEXT DEFAULT 'en',
    location TEXT,
    crops_cultivated TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    farmer_id TEXT,
    started_at TEXT
);

CREATE TABLE IF NOT EXISTS queries (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    farmer_id TEXT,
    raw_text TEXT,
    language TEXT,
    translated_text TEXT,
    intent TEXT,
    intent_confidence REAL,
    entities_json TEXT,
    answer TEXT,
    answer_confidence REAL,
    input_mode TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER,
    rating TEXT,
    created_at TEXT,
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def upsert_farmer(farmer_id: str, name: str = None, preferred_language: str = "en",
                   location: str = None, crops_cultivated: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT farmer_id FROM farmers WHERE farmer_id = ?", (farmer_id,))
    if cur.fetchone():
        cur.execute(
            "UPDATE farmers SET name=?, preferred_language=?, location=?, crops_cultivated=? "
            "WHERE farmer_id=?",
            (name, preferred_language, location, crops_cultivated, farmer_id))
    else:
        cur.execute(
            "INSERT INTO farmers (farmer_id, name, preferred_language, location, "
            "crops_cultivated, created_at) VALUES (?,?,?,?,?,?)",
            (farmer_id, name, preferred_language, location, crops_cultivated, _now()))
    conn.commit()
    conn.close()


def get_farmer(farmer_id: str):
    """Look up a farmer profile by ID. Returns a dict or None if not found -
    used by the sidebar's 'Resume with existing Farmer ID' flow so a farmer
    can pick up their profile/history in a new browser session instead of
    always starting a brand-new anonymous profile."""
    if not farmer_id:
        return None
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM farmers WHERE farmer_id = ?", (farmer_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_queries_for_farmer(farmer_id: str, limit: int = 200):
    """All queries across every conversation for this farmer (not just the
    current browser session's conversation_id) - powers the Farmer Profile
    tab's full history view."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM queries WHERE farmer_id=? ORDER BY query_id DESC LIMIT ?",
                (farmer_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def start_conversation(conversation_id: str, farmer_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO conversations (conversation_id, farmer_id, started_at) "
                "VALUES (?,?,?)", (conversation_id, farmer_id, _now()))
    conn.commit()
    conn.close()


def log_query(conversation_id: str, farmer_id: str, raw_text: str, language: str,
              translated_text: str, intent: str, intent_confidence: float,
              entities: dict, answer: str, answer_confidence: float,
              input_mode: str = "text") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO queries (conversation_id, farmer_id, raw_text, language, translated_text, "
        "intent, intent_confidence, entities_json, answer, answer_confidence, input_mode, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (conversation_id, farmer_id, raw_text, language, translated_text, intent,
         intent_confidence, json.dumps(entities, ensure_ascii=False), answer,
         answer_confidence, input_mode, _now()))
    query_id = cur.lastrowid
    conn.commit()
    conn.close()
    return query_id


def log_feedback(query_id: int, rating: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (query_id, rating, created_at) VALUES (?,?,?)",
                (query_id, rating, _now()))
    conn.commit()
    conn.close()


def get_conversation_history(conversation_id: str, limit: int = 50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM queries WHERE conversation_id=? ORDER BY query_id ASC LIMIT ?",
                (conversation_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_analytics() -> dict:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM queries")
    total_queries = cur.fetchone()["c"]

    cur.execute("SELECT intent, COUNT(*) AS c FROM queries WHERE intent IS NOT NULL "
                "GROUP BY intent ORDER BY c DESC")
    intent_distribution = {r["intent"]: r["c"] for r in cur.fetchall()}

    cur.execute("SELECT language, COUNT(*) AS c FROM queries GROUP BY language")
    language_distribution = {r["language"]: r["c"] for r in cur.fetchall()}

    cur.execute("""
        SELECT entities_json FROM queries WHERE entities_json IS NOT NULL
    """)
    crop_counter = {}
    for row in cur.fetchall():
        try:
            ents = json.loads(row["entities_json"])
        except Exception:
            continue
        for crop in ents.get("CROP", []):
            crop_counter[crop] = crop_counter.get(crop, 0) + 1
    most_searched_crops = dict(sorted(crop_counter.items(), key=lambda kv: -kv[1])[:10])

    cur.execute("SELECT rating, COUNT(*) AS c FROM feedback GROUP BY rating")
    feedback_counts = {r["rating"]: r["c"] for r in cur.fetchall()}
    helpful = feedback_counts.get("helpful", 0)
    not_helpful = feedback_counts.get("not_helpful", 0)
    total_feedback = helpful + not_helpful
    helpful_pct = (helpful / total_feedback * 100) if total_feedback else None

    conn.close()
    return {
        "total_queries": total_queries,
        "intent_distribution": intent_distribution,
        "language_distribution": language_distribution,
        "most_searched_crops": most_searched_crops,
        "feedback_counts": feedback_counts,
        "helpful_percentage": helpful_pct,
    }


def get_entity_frequency(top_n: int = 12) -> dict:
    """Frequency of every extracted entity value, across all entity types and
    all queries system-wide - real data read from the same `entities_json`
    column `get_analytics()` already parses for crop frequency."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT entities_json FROM queries WHERE entities_json IS NOT NULL")
    counter = {}
    for row in cur.fetchall():
        try:
            ents = json.loads(row["entities_json"])
        except Exception:
            continue
        for label, values in ents.items():
            for v in values:
                key = f"{v} ({label})"
                counter[key] = counter.get(key, 0) + 1
    conn.close()
    return dict(sorted(counter.items(), key=lambda kv: -kv[1])[:top_n])


def get_confidence_values() -> dict:
    """Raw intent_confidence and answer_confidence values across every
    logged query, for a real confidence-distribution histogram."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT intent_confidence, answer_confidence FROM queries "
                "WHERE intent_confidence IS NOT NULL OR answer_confidence IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return {
        "intent_confidence": [r["intent_confidence"] for r in rows if r["intent_confidence"] is not None],
        "answer_confidence": [r["answer_confidence"] for r in rows if r["answer_confidence"] is not None],
    }


if __name__ == "__main__":
    init_db()
    upsert_farmer("demo_farmer", name="Demo Farmer", preferred_language="ta",
                   location="Coimbatore", crops_cultivated="Rice, Cotton")
    start_conversation("demo_conv_1", "demo_farmer")
    qid = log_query("demo_conv_1", "demo_farmer", "My rice crop has yellow leaves", "en",
                     None, "Crop Disease", 0.9, {"CROP": ["rice"], "SYMPTOM": ["yellow leaves"]},
                     "Yellowing can indicate...", 0.8)
    log_feedback(qid, "helpful")
    print(get_analytics())
