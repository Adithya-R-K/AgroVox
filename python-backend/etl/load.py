"""
AgroVox - ETL Step 3: Load.

Takes the cleaned, feature-engineered dataset from Transform
(data/processed/farmer_queries_clean.csv) and LOADS it into the two places
the rest of AgroVox actually consumes data from:

  1. A proper SQLite table (`processed_farmer_queries`) - structured,
     query-able storage, exactly what "Load" means in ETL (as opposed to
     leaving cleaned data stranded in a CSV nobody reads from).
  2. The live training corpus used by the text classifier
     (data/intents.csv) - clean, deduplicated, schema-validated rows are
     merged in as additional real-world-shaped training signal alongside
     the original template-generated set, and the classifier is expected
     to be retrained after a Load (see train_classifier.py / etl_pipeline.py).

Run (after etl/transform.py): python etl/load.py
"""
import os
import sys
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOAD_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_farmer_queries (
    record_id INTEGER PRIMARY KEY,
    log_id INTEGER,
    raw_text TEXT,
    cleaned_text TEXT,
    processed_text TEXT,
    stemmed_text TEXT,
    label TEXT,
    language TEXT,
    device TEXT,
    token_count INTEGER,
    char_count INTEGER,
    loaded_at TEXT
);
"""


def load_to_sqlite(clean_df: pd.DataFrame, db_path: str = None) -> int:
    """Loads the cleaned dataset into a dedicated SQLite table, replacing
    any previous load (idempotent - safe to re-run the whole ETL pipeline
    without accumulating duplicate loads)."""
    from datetime import datetime, timezone
    db_path = db_path or config.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.execute(LOAD_TABLE_SCHEMA)
    conn.execute("DELETE FROM processed_farmer_queries")

    cols = ["record_id", "log_id", "raw_text", "cleaned_text", "processed_text",
            "stemmed_text", "label", "language", "device", "token_count", "char_count"]
    to_load = clean_df[cols].copy()
    to_load["loaded_at"] = datetime.now(timezone.utc).isoformat()

    to_load.to_sql("processed_farmer_queries", conn, if_exists="append", index=False)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM processed_farmer_queries").fetchone()[0]
    conn.close()
    return n


def load_to_training_corpus(clean_df: pd.DataFrame, intents_csv: str = None,
                              dedupe_against_existing: bool = True) -> dict:
    """Merges the cleaned, real-world-shaped rows into data/intents.csv (the
    classifier's training corpus), so a subsequent `train_classifier.py`
    run genuinely learns from this cleaned batch too - this is the step
    that makes Load meaningful rather than cosmetic: the loaded data
    actually changes what the deployed model is trained on.

    IMPORTANT (data-leakage guard): deduplication is done on a
    case/whitespace-NORMALIZED key, not an exact string match. Since this
    batch's raw text originated from the same source sentences as the
    existing corpus (just passed through noisy casing + then re-cleaned),
    an exact-match dedupe would let near-identical rows differing only by
    casing through as "new" examples. Those near-duplicates would then be
    split across train/test by `train_test_split`, letting the test set
    leak information the model already saw in training in a different
    case - which is exactly how the classifier evaluation briefly showed
    an inflated, meaningless 100% accuracy during this project's own
    audit. Normalizing before comparing prevents that leakage.
    """
    intents_csv = intents_csv or config.INTENTS_CSV
    existing = pd.read_csv(intents_csv) if os.path.exists(intents_csv) else \
        pd.DataFrame(columns=["text", "label", "language"])

    incoming = clean_df[["cleaned_text", "label", "language"]].rename(
        columns={"cleaned_text": "text"})

    def _norm_key(s):
        return " ".join(str(s).lower().split())

    existing_keys = set(zip(existing["text"].map(_norm_key), existing["label"]))

    n_incoming = len(incoming)
    if dedupe_against_existing:
        is_new = ~incoming.apply(
            lambda r: (_norm_key(r["text"]), r["label"]) in existing_keys, axis=1)
        genuinely_new = incoming[is_new]
    else:
        genuinely_new = incoming

    combined = pd.concat([existing, genuinely_new], ignore_index=True)
    # also collapse any remaining normalized duplicates within the combined set
    combined["_key"] = list(zip(combined["text"].map(_norm_key), combined["label"]))
    combined = combined.drop_duplicates(subset=["_key"], keep="first").drop(columns="_key")

    combined.to_csv(intents_csv, index=False)
    return {"total_rows": len(combined), "incoming_rows": n_incoming,
             "genuinely_new_rows": len(genuinely_new),
             "skipped_as_near_duplicate": n_incoming - len(genuinely_new)}


if __name__ == "__main__":
    clean_path = os.path.join(config.PROCESSED_DATA_DIR, "farmer_queries_clean.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"{clean_path} not found - run etl/transform.py first.")
    clean_df = pd.read_csv(clean_path, keep_default_na=False)

    n_sqlite = load_to_sqlite(clean_df)
    print(f"[LOAD] {n_sqlite} rows now in SQLite table `processed_farmer_queries`.")

    n_corpus = load_to_training_corpus(clean_df)
    print(f"[LOAD] Training corpus data/intents.csv: {n_corpus['genuinely_new_rows']} genuinely "
          f"new rows added, {n_corpus['skipped_as_near_duplicate']} skipped as near-duplicates "
          f"of existing rows -> {n_corpus['total_rows']} total rows.")
    print("[LOAD] Run `python train_classifier.py` to retrain the classifier on the updated corpus.")
