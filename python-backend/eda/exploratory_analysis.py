"""
AgroVox - Exploratory Data Analysis (EDA)

Produces a full statistical + visual profile of AgroVox's datasets:
  - Raw vs. cleaned data-quality comparison (from the ETL transform report)
  - Class (intent) balance
  - Text length distribution (characters, tokens)
  - Language distribution
  - Most frequent words (after stopword removal) per intent
  - Most frequent agricultural entities across the QA/knowledge base
  - Knowledge-base coverage (rows per category)

This is deliberately built as a library of pure functions returning pandas
DataFrames/dicts (not just a script that prints things), so the exact same
analysis functions power both:
  (a) `python eda/exploratory_analysis.py` — a standalone report (saves
      PNGs + a JSON summary to eda/results/), and
  (b) the Streamlit app's "Data Pipeline (ETL & EDA)" tab, which calls
      these functions directly and renders the results as live Plotly
      charts instead of static images.

Run: python eda/exploratory_analysis.py
"""
import os
import sys
import json
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing.text_preprocessing import preprocess

EDA_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(EDA_RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------ LOADERS --
def load_intents() -> pd.DataFrame:
    return pd.read_csv(config.INTENTS_CSV)


def load_raw_log() -> pd.DataFrame:
    path = os.path.join(config.RAW_DATA_DIR, "farmer_queries_raw.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def load_clean_log() -> pd.DataFrame:
    path = os.path.join(config.PROCESSED_DATA_DIR, "farmer_queries_clean.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def load_transform_report() -> dict:
    path = os.path.join(config.PROCESSED_DATA_DIR, "transform_report.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ ANALYSES --
def class_balance(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    counts = df[label_col].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "percent": pct}).reset_index().rename(
        columns={"index": label_col})


def text_length_stats(df: pd.DataFrame, text_col: str = "text") -> dict:
    lengths_chars = df[text_col].astype(str).str.len()
    lengths_tokens = df[text_col].astype(str).str.split().str.len()
    return {
        "char_length": {
            "mean": float(lengths_chars.mean()), "median": float(lengths_chars.median()),
            "min": int(lengths_chars.min()), "max": int(lengths_chars.max()),
            "std": float(lengths_chars.std()),
        },
        "token_length": {
            "mean": float(lengths_tokens.mean()), "median": float(lengths_tokens.median()),
            "min": int(lengths_tokens.min()), "max": int(lengths_tokens.max()),
            "std": float(lengths_tokens.std()),
        },
        "char_length_series": lengths_chars.tolist(),
        "token_length_series": lengths_tokens.tolist(),
    }


def language_distribution(df: pd.DataFrame, lang_col: str = "language") -> pd.DataFrame:
    counts = df[lang_col].value_counts()
    return counts.reset_index().rename(columns={"index": lang_col, lang_col: "count"}) \
        if hasattr(counts, "reset_index") else pd.DataFrame()


def top_words_overall(df: pd.DataFrame, text_col: str = "text", top_n: int = 20) -> pd.DataFrame:
    counter = Counter()
    for text in df[text_col].astype(str):
        info = preprocess(text, language="en")
        counter.update(info["processed_tokens"])
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def top_words_by_class(df: pd.DataFrame, text_col: str = "text", label_col: str = "label",
                        top_n: int = 8) -> dict:
    result = {}
    for label, group in df.groupby(label_col):
        counter = Counter()
        for text in group[text_col].astype(str):
            info = preprocess(text, language="en")
            counter.update(info["processed_tokens"])
        result[label] = counter.most_common(top_n)
    return result


def duplicate_and_missing_summary(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": len(df),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_values_per_column": {c: int(df[c].isna().sum() + (df[c] == "").sum())
                                        for c in df.columns},
    }
    return summary


def knowledge_base_coverage() -> pd.DataFrame:
    from src.retrieval.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    stats = kb.stats()
    return pd.DataFrame(list(stats.items()), columns=["table", "row_count"])


def qa_confidence_vs_length():
    """Explores whether longer/shorter questions in the QA corpus retrieve
    with different average self-similarity - a simple bivariate EDA check
    of whether question length correlates with retrieval confidence."""
    from src.qa.qa_engine import TFIDFQAEngine
    qa = TFIDFQAEngine()
    rows = []
    for _, row in qa.df.iterrows():
        result = qa.answer(row["question"], top_k=1)
        rows.append({
            "question_length_words": len(row["question"].split()),
            "self_retrieval_confidence": result["confidence"],
            "category": row["category"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ REPORT --
def build_full_report() -> dict:
    intents_df = load_intents()
    raw_df = load_raw_log()
    clean_df = load_clean_log()
    transform_report = load_transform_report()

    report = {
        "intents_dataset": {
            "n_rows": len(intents_df),
            "class_balance": class_balance(intents_df).to_dict(orient="records"),
            "text_length_stats": text_length_stats(intents_df),
            "language_distribution": intents_df["language"].value_counts().to_dict(),
            "top_words_overall": top_words_overall(intents_df).to_dict(orient="records"),
            "duplicate_and_missing": duplicate_and_missing_summary(intents_df),
        },
        "raw_vs_clean": {
            "raw_row_count": len(raw_df) if not raw_df.empty else None,
            "clean_row_count": len(clean_df) if not clean_df.empty else None,
            "transform_report": transform_report,
        },
        "knowledge_base_coverage": knowledge_base_coverage().to_dict(orient="records"),
    }
    return report


def save_plots(report: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    intents_df = load_intents()

    # 1. Class balance
    plt.figure(figsize=(9, 5))
    order = intents_df["label"].value_counts().index
    sns.countplot(data=intents_df, y="label", order=order, color="#2F5233")
    plt.title("AgroVox — Intent Class Balance (data/intents.csv)")
    plt.xlabel("Count")
    plt.ylabel("Intent")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_RESULTS_DIR, "class_balance.png"), dpi=150)
    plt.close()

    # 2. Text length distribution
    lengths = intents_df["text"].astype(str).str.split().str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=15, color="#C97A3A")
    plt.title("AgroVox — Query Length Distribution (tokens)")
    plt.xlabel("Tokens per query")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_RESULTS_DIR, "text_length_distribution.png"), dpi=150)
    plt.close()

    # 3. Top words
    top_words = top_words_overall(intents_df, top_n=20)
    plt.figure(figsize=(8, 7))
    sns.barplot(data=top_words, x="count", y="word", color="#7A9E3F")
    plt.title("AgroVox — Top 20 Most Frequent Words (post stopword removal)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_RESULTS_DIR, "top_words.png"), dpi=150)
    plt.close()

    # 4. Raw vs Clean row counts (ETL funnel)
    tr = load_transform_report()
    if tr:
        stages = tr["stages"]
        plt.figure(figsize=(9, 5))
        names = [s["stage"] for s in stages]
        rows = [s["rows"] for s in stages]
        plt.barh(names, rows, color="#D9A441")
        plt.title("AgroVox — ETL Transform Funnel (rows remaining per stage)")
        plt.xlabel("Rows")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_RESULTS_DIR, "etl_funnel.png"), dpi=150)
        plt.close()

    print(f"Saved EDA plots -> {EDA_RESULTS_DIR}")


if __name__ == "__main__":
    report = build_full_report()
    out_path = os.path.join(EDA_RESULTS_DIR, "eda_summary.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved EDA summary -> {out_path}")
    save_plots(report)

    print("\n--- Quick EDA Summary ---")
    print(f"Intents dataset: {report['intents_dataset']['n_rows']} rows")
    print("Class balance:")
    for row in report["intents_dataset"]["class_balance"]:
        print(f"  {row['label']:22s} {row['count']:4d}  ({row['percent']}%)")
    print(f"Duplicate rows: {report['intents_dataset']['duplicate_and_missing']['n_duplicate_rows']}")
    if report["raw_vs_clean"]["raw_row_count"]:
        print(f"\nRaw log rows: {report['raw_vs_clean']['raw_row_count']}  "
              f"-> Clean rows: {report['raw_vs_clean']['clean_row_count']}")
