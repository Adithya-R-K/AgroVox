"""
AgroVox - ETL Step 2: Transform.

Takes the messy raw log (data/raw/farmer_queries_raw.csv) and applies the
full cleaning + feature-engineering pipeline described in the project spec
(Section 9: missing value handling, duplicate removal, normalization,
tokenization, stopword removal, lemmatization/stemming, dataset validation,
label encoding groundwork), producing an analysis-ready dataset at
data/processed/farmer_queries_clean.csv plus a JSON data-quality report at
data/processed/transform_report.json documenting exactly what was fixed and
how much - this report is what the EDA module and the "Data Pipeline" tab
in the Streamlit app both read from.

Run (after etl/extract.py): python etl/transform.py
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing.text_preprocessing import preprocess, detect_language


def load_raw(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(config.RAW_DATA_DIR, "farmer_queries_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run etl/extract.py first.")
    return pd.read_csv(path, keep_default_na=False)


def transform(df: pd.DataFrame) -> tuple:
    """Runs the full clean/normalize/validate/feature-engineer pipeline.
    Returns (clean_df, report_dict). Every stage's row-count effect is
    tracked in `report_dict` — this row-count lineage (how many rows
    survive each stage, and why rows were dropped) is standard ETL/data-
    quality practice, not just for show.
    """
    report = {"stages": []}
    n_start = len(df)
    report["stages"].append({"stage": "raw_extracted", "rows": n_start})

    working = df.copy()

    # ---- 1. Missing-value handling --------------------------------------
    # Empty/whitespace-only `raw_text` cannot be salvaged - drop.
    working["raw_text"] = working["raw_text"].astype(str)
    is_blank_text = working["raw_text"].str.strip().eq("") | working["raw_text"].isna()
    n_blank_text = int(is_blank_text.sum())
    working = working[~is_blank_text]

    # Empty `label` also cannot be used for supervised training - drop.
    working["label"] = working["label"].astype(str)
    is_blank_label = working["label"].str.strip().eq("") | working["label"].isna()
    n_blank_label = int(is_blank_label.sum())
    working = working[~is_blank_label]

    report["stages"].append({
        "stage": "missing_value_handling",
        "rows": len(working),
        "dropped_blank_text": n_blank_text,
        "dropped_blank_label": n_blank_label,
    })

    # Missing `language` is NOT dropped - it's recoverable via language
    # detection (this is the key distinction in real ETL: missing values
    # are either dropped, or imputed/recovered depending on whether a
    # reliable recovery method exists).
    n_missing_language_before = int((working["language"].astype(str).str.strip() == "").sum())
    working["language"] = working.apply(
        lambda r: detect_language(r["raw_text"]) if str(r["language"]).strip() == ""
        else r["language"], axis=1)
    report["stages"].append({
        "stage": "language_imputation_via_detection",
        "rows": len(working),
        "imputed_language_count": n_missing_language_before,
    })

    # Missing `device` -> impute with an explicit "unknown" category rather
    # than dropping (device is metadata, not needed for NLP training).
    working["device"] = working["device"].astype(str)
    n_missing_device = int((working["device"].str.strip() == "").sum())
    working["device"] = working["device"].replace("", "unknown")

    # ---- 2. Punctuation-only / junk-content rows -------------------------
    # e.g. "???...!!" survives the blank-text check but has zero usable
    # linguistic content once cleaned.
    def _has_alpha_content(text):
        return any(ch.isalpha() for ch in text)

    is_junk_content = ~working["raw_text"].apply(_has_alpha_content)
    n_junk_content = int(is_junk_content.sum())
    working = working[~is_junk_content]
    report["stages"].append({
        "stage": "junk_content_removal",
        "rows": len(working),
        "dropped_punctuation_only_rows": n_junk_content,
    })

    # ---- 3. Text normalization (casing, whitespace) ----------------------
    working["cleaned_text"] = working["raw_text"].apply(
        lambda t: preprocess(t)["cleaned_text"])

    # ---- 4. Duplicate removal --------------------------------------------
    # Real duplicate submissions (identical normalized text + label), not
    # just identical log_id - this catches double-taps even if the logging
    # system assigned different device/log_id metadata to each tap.
    n_before_dedup = len(working)
    working = working.drop_duplicates(subset=["cleaned_text", "label"], keep="first")
    n_duplicates_removed = n_before_dedup - len(working)
    report["stages"].append({
        "stage": "duplicate_removal",
        "rows": len(working),
        "duplicates_removed": int(n_duplicates_removed),
    })

    # ---- 5. Tokenization, stopword removal, lemmatization/stemming ------
    def _full_preprocess(row):
        info = preprocess(row["raw_text"], language=row["language"])
        return pd.Series({
            "tokens": " ".join(info["tokens"]),
            "processed_text": info["processed_text"],
            "stemmed_text": info.get("stemmed_text", info["processed_text"]),
            "token_count": len(info["tokens"]),
            "char_count": len(info["cleaned_text"]),
        })

    feature_cols = working.apply(_full_preprocess, axis=1)
    working = pd.concat([working.reset_index(drop=True), feature_cols.reset_index(drop=True)], axis=1)

    # ---- 6. Dataset validation -------------------------------------------
    # Drop rows that, after full preprocessing, ended up with zero tokens
    # (can happen for e.g. pure-stopword or pure-number text).
    n_before_validation = len(working)
    working = working[working["token_count"] > 0]
    n_empty_after_processing = n_before_validation - len(working)
    report["stages"].append({
        "stage": "post_processing_validation",
        "rows": len(working),
        "dropped_empty_after_processing": int(n_empty_after_processing),
    })

    # Validate labels are within the known label set (schema validation)
    valid_labels = set(config.INTENT_LABELS)
    invalid_label_mask = ~working["label"].isin(valid_labels)
    n_invalid_labels = int(invalid_label_mask.sum())
    working = working[~invalid_label_mask]
    report["stages"].append({
        "stage": "schema_validation_labels",
        "rows": len(working),
        "dropped_invalid_labels": n_invalid_labels,
    })

    working = working.reset_index(drop=True)
    working.insert(0, "record_id", range(1, len(working) + 1))

    report["final_row_count"] = len(working)
    report["total_rows_dropped"] = n_start - len(working)
    report["retention_rate"] = round(len(working) / n_start, 4) if n_start else 0.0
    report["missing_device_imputed"] = n_missing_device
    report["missing_language_imputed"] = n_missing_language_before
    report["duplicates_removed"] = int(n_duplicates_removed)

    return working, report


if __name__ == "__main__":
    raw_df = load_raw()
    clean_df, report = transform(raw_df)
    out_csv = os.path.join(config.PROCESSED_DATA_DIR, "farmer_queries_clean.csv")
    clean_df.to_csv(out_csv, index=False)
    out_report = os.path.join(config.PROCESSED_DATA_DIR, "transform_report.json")
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[TRANSFORM] {report['total_rows_dropped']} of {len(raw_df)} raw rows dropped "
          f"({(1-report['retention_rate'])*100:.1f}%); {len(clean_df)} clean rows retained.")
    print(f"[TRANSFORM] Wrote clean dataset -> {out_csv}")
    print(f"[TRANSFORM] Wrote data-quality report -> {out_report}")
