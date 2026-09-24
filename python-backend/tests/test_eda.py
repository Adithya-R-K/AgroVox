"""Unit tests: EDA (exploratory data analysis) functions."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eda import exploratory_analysis as eda_mod


def _sample_df():
    return pd.DataFrame({
        "text": ["My rice crop has yellow leaves", "My rice crop has yellow leaves",
                 "What fertilizer is good for wheat"],
        "label": ["Crop Disease", "Crop Disease", "Fertilizer"],
        "language": ["en", "en", "en"],
    })


def test_class_balance_counts_correctly():
    df = _sample_df()
    balance = eda_mod.class_balance(df)
    row = balance[balance["label"] == "Crop Disease"].iloc[0]
    assert row["count"] == 2


def test_text_length_stats_computes_reasonable_values():
    df = _sample_df()
    stats = eda_mod.text_length_stats(df)
    assert stats["token_length"]["min"] > 0
    assert stats["char_length"]["max"] > 0


def test_top_words_overall_excludes_stopwords():
    df = _sample_df()
    top = eda_mod.top_words_overall(df, top_n=10)
    words = top["word"].tolist()
    assert "rice" in words
    assert "is" not in words  # stopword should not appear


def test_duplicate_and_missing_summary_detects_duplicates():
    df = _sample_df()
    summary = eda_mod.duplicate_and_missing_summary(df)
    assert summary["n_duplicate_rows"] == 1  # the two identical rice rows


def test_knowledge_base_coverage_returns_nonzero_tables():
    coverage = eda_mod.knowledge_base_coverage()
    assert len(coverage) > 0
    assert (coverage["row_count"] > 0).all()
