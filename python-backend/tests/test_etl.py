"""Unit/integration tests: ETL pipeline (Extract, Transform, Load).

Specifically guards against the data-leakage bug found during the
production audit: near-duplicate rows (same text, different casing) must
NOT be double-counted as "genuinely new" training examples when Load
merges cleaned data into the training corpus, since that previously caused
an inflated, meaningless 100% classifier accuracy from train/test leakage.
"""
import os
import sys
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from etl.extract import extract_raw_farmer_queries
from etl.transform import load_raw, transform
from etl.load import load_to_sqlite, load_to_training_corpus


def test_extract_produces_more_rows_than_source_due_to_duplicates_and_junk():
    tmp_raw = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    out_path = extract_raw_farmer_queries(out_path=tmp_raw)
    df = pd.read_csv(out_path, keep_default_na=False)
    base_rows = len(pd.read_csv(config.INTENTS_CSV))
    assert len(df) > base_rows  # duplicates + junk rows were injected
    os.unlink(out_path)


def test_transform_drops_blank_and_junk_rows():
    df = pd.DataFrame({
        "log_id": [1, 2, 3, 4],
        "raw_text": ["My rice crop has yellow leaves", "", "   ", "???..."],
        "label": ["Crop Disease", "Crop Disease", "Crop Disease", "Crop Disease"],
        "language": ["en", "en", "en", "en"],
        "device": ["web", "web", "web", "web"],
    })
    clean_df, report = transform(df)
    assert len(clean_df) == 1  # only the first row survives
    assert report["total_rows_dropped"] == 3


def test_transform_imputes_missing_language():
    df = pd.DataFrame({
        "log_id": [1],
        "raw_text": ["My rice crop has yellow leaves"],
        "label": ["Crop Disease"],
        "language": [""],
        "device": ["web"],
    })
    clean_df, report = transform(df)
    assert clean_df.iloc[0]["language"] == "en"
    assert report["missing_language_imputed"] == 1


def test_transform_removes_duplicates():
    df = pd.DataFrame({
        "log_id": [1, 2],
        "raw_text": ["My rice crop has yellow leaves", "MY RICE CROP HAS YELLOW LEAVES"],
        "label": ["Crop Disease", "Crop Disease"],
        "language": ["en", "en"],
        "device": ["web", "web"],
    })
    clean_df, report = transform(df)
    assert len(clean_df) == 1
    assert report["duplicates_removed"] == 1


def test_transform_drops_invalid_labels():
    df = pd.DataFrame({
        "log_id": [1, 2],
        "raw_text": ["My rice crop has yellow leaves", "Some other query text here"],
        "label": ["Crop Disease", "NotARealIntent"],
        "language": ["en", "en"],
        "device": ["web", "web"],
    })
    clean_df, report = transform(df)
    assert len(clean_df) == 1
    assert report["stages"][-1]["dropped_invalid_labels"] == 1


def test_load_to_training_corpus_prevents_near_duplicate_leakage():
    """Regression test for the exact bug found in production audit: loading
    cleaned (lowercased) text that is a case-variant of an EXISTING training
    row must not be counted as new data."""
    tmp_intents = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    pd.DataFrame({
        "text": ["My rice crop has yellow leaves"],
        "label": ["Crop Disease"],
        "language": ["en"],
    }).to_csv(tmp_intents, index=False)

    clean_df = pd.DataFrame({
        "cleaned_text": ["my rice crop has yellow leaves"],  # same content, different case
        "label": ["Crop Disease"],
        "language": ["en"],
    })

    result = load_to_training_corpus(clean_df, intents_csv=tmp_intents)
    assert result["genuinely_new_rows"] == 0
    assert result["skipped_as_near_duplicate"] == 1
    assert result["total_rows"] == 1  # no bloat from the near-duplicate

    os.unlink(tmp_intents)


def test_load_to_training_corpus_adds_genuinely_new_text():
    tmp_intents = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    pd.DataFrame({
        "text": ["My rice crop has yellow leaves"],
        "label": ["Crop Disease"],
        "language": ["en"],
    }).to_csv(tmp_intents, index=False)

    clean_df = pd.DataFrame({
        "cleaned_text": ["completely different sentence about cotton pests"],
        "label": ["Pest Management"],
        "language": ["en"],
    })

    result = load_to_training_corpus(clean_df, intents_csv=tmp_intents)
    assert result["genuinely_new_rows"] == 1
    assert result["total_rows"] == 2

    os.unlink(tmp_intents)


def test_load_to_sqlite_is_idempotent():
    _tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    clean_df = pd.DataFrame({
        "record_id": [1], "log_id": [1], "raw_text": ["test"], "cleaned_text": ["test"],
        "processed_text": ["test"], "stemmed_text": ["test"], "label": ["Soil"],
        "language": ["en"], "device": ["web"], "token_count": [1], "char_count": [4],
    })
    n1 = load_to_sqlite(clean_df, db_path=_tmp_db)
    n2 = load_to_sqlite(clean_df, db_path=_tmp_db)  # re-running must not duplicate
    assert n1 == n2 == 1
    os.unlink(_tmp_db)
