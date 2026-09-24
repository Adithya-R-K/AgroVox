"""
AgroVox - Real-dataset ingestion: Kisan Call Centre (KCC) -> intents.csv

The classifier previously trained only on data/intents.csv, a template-
generated file (see generate_intents.py) - grammatically clean but
artificially uniform, which makes the classifier look better than it would
on real, messy farmer language. This script replaces that file with real
farmer queries from the Kisan Call Centre (KCC) dataset - the Government of
India's record of actual calls farmers made to agricultural helplines -
remapped onto AgroVox's 10 intent labels (see config.INTENT_LABELS).

Get the raw data first (pick one):
  - Kaggle mirror (single CSV, easiest for a lab project):
    https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa
  - Official source (Government of India Open Data Platform), split into
    many per-state/month files:
    https://www.data.gov.in/catalog/district-wise-and-month-wise-queries-farmers-kisan-call-centre-kcc

Save the CSV anywhere and point --input at it, e.g.:
    data/raw/kcc_queries.csv

The published schema for this dataset is (case may vary slightly by mirror):
    StateName, DistrictName, BlockName, Season, Sector, Category, Crop,
    QueryType, QueryText, KccAns, ...
This script auto-detects the query-text and category columns from a list of
common aliases (see COLUMN_ALIASES below). If it can't find them, it prints
the actual column names it found so you can add the right alias yourself -
open this file and add a line to COLUMN_ALIASES, no need to touch any other
module.

Usage:
    python data/build_real_intents.py --input data/raw/kcc_queries.csv
    python data/build_real_intents.py --input data/raw/kcc_queries.csv \
        --state "TAMIL NADU" --max-per-label 300

Run this BEFORE train_classifier.py / etl/extract.py, since both of those
read data/intents.csv as their starting point.
"""
import os
import re
import sys
import argparse
import random
import shutil

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# Column auto-detection: add more aliases here if your mirror uses different
# header names - the script will tell you the real header names if none of
# these match.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "text": ["querytext", "query text", "queries", "query", "questions", "question"],
    # KCC's official schema has BOTH a broad "Category" (e.g. "Weather") and a
    # more specific "QueryType" (e.g. "Rainfall") - both are matched and
    # combined below since either can carry the label-relevant keyword.
    "category": ["category", "sector"],
    "querytype": ["querytype", "query type"],
    "state": ["statename", "state name", "state"],
}

# ---------------------------------------------------------------------------
# Real KCC "Category"/"QueryType" values -> AgroVox's 10 intent labels.
# Checked as LEFT-word-boundary matches (regex \b before the keyword only,
# not a full \bword\b) against category+querytype combined, in order (first
# match wins). Left-only boundary avoids two opposite failure modes: a plain
# substring check like `"rot" in text` wrongly fires on "p-ROT-ection", while
# a full \bword\b requires an exact word and misses plurals like "practices"
# when the rule says "practice". Left-boundary-only rejects the first (no
# word boundary before "rot" inside "protection" - "p" and "r" are both word
# chars) while still matching the second (boundary exists before "cultural").
#
# Anything left unmatched falls back to "General Agriculture" - the script
# prints those leftover raw values so you can add a rule for them instead of
# silently mis-labelling. One case worth knowing up front: real KCC data uses
# "Plant Protection" as an umbrella QueryType covering BOTH pest and disease
# queries with no further distinction in the category field - that ambiguity
# can't be resolved from the category alone, so map_to_intent_label() falls
# back to matching the query TEXT itself in that case (see below).
# ---------------------------------------------------------------------------
LABEL_RULES = [
    (["disease", "fungal", "blight", "wilt", "root rot", "fruit rot", "virus", "rust"], "Crop Disease"),
    (["pest", "insect", "borer", "mite", "rodent", "weed", "nematode"], "Pest Management"),
    (["fertil", "nutrient", "manure", "micronutrient"], "Fertilizer"),
    (["irrigat", "water manage", "water requirement"], "Irrigation"),
    (["soil health", "soil test", "soil"], "Soil"),
    (["weather", "climate", "rainfall", "monsoon"], "Weather"),
    (["harvest", "post harvest", "post-harvest", "storage"], "Harvesting"),
    (["variety", "varieties", "hybrid", "hybrids"], "Crop Information"),
    (["sow", "sowing", "nursery", "transplant", "land preparation",
      "seed", "planting material", "spacing", "cultivat", "cultural practice"], "Cultivation"),
]
FALLBACK_LABEL = "General Agriculture"
# Category/QueryType values known to be too generic to map directly - for
# these, classify from the query TEXT instead (see map_to_intent_label).
AMBIGUOUS_CATEGORY_TERMS = ["plant protection"]

_COMPILED_RULES = [
    (re.compile(r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")", re.IGNORECASE), label)
    for keywords, label in LABEL_RULES
]
_AMBIGUOUS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in AMBIGUOUS_CATEGORY_TERMS) + r")", re.IGNORECASE)


def _find_column(df: pd.DataFrame, aliases: list) -> str:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def map_to_intent_label(category_text: str, query_text: str = "") -> str:
    text = category_text or ""
    if _AMBIGUOUS_RE.search(text) and query_text:
        # e.g. "Plant Protection" alone doesn't say pest vs. disease -
        # classify from what the farmer actually asked instead.
        for pattern, label in _COMPILED_RULES:
            if pattern.search(query_text):
                return label
        return FALLBACK_LABEL
    for pattern, label in _COMPILED_RULES:
        if pattern.search(text):
            return label
    return FALLBACK_LABEL


def _has_alpha_content(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def load_and_map(input_path: str, state_filter: str, min_words: int, max_words: int) -> pd.DataFrame:
    # Government CSV exports are often UTF-8-with-BOM or Latin-1; try both.
    try:
        header = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig")
    except UnicodeDecodeError:
        header = pd.read_csv(input_path, nrows=0, encoding="latin-1")

    text_col = _find_column(header, COLUMN_ALIASES["text"])
    cat_col = _find_column(header, COLUMN_ALIASES["category"])
    qtype_col = _find_column(header, COLUMN_ALIASES["querytype"])
    state_col = _find_column(header, COLUMN_ALIASES["state"])

    if not text_col or not (cat_col or qtype_col):
        raise SystemExit(
            "Could not auto-detect the query-text / category columns.\n"
            f"Columns found in {input_path}:\n  " + ", ".join(header.columns) +
            "\nOpen data/build_real_intents.py and add the right names to "
            "COLUMN_ALIASES['text'] / ['category'] / ['querytype'], then re-run."
        )
    print(f"[detect] text column      = '{text_col}'")
    print(f"[detect] category column  = {cat_col!r}")
    print(f"[detect] querytype column = {qtype_col!r}")
    if state_filter and not state_col:
        raise SystemExit(
            f"--state was given but no state column was found. Columns: {', '.join(header.columns)}")
    if state_col:
        print(f"[detect] state column     = '{state_col}'")

    usecols = [c for c in [text_col, cat_col, qtype_col, state_col] if c]

    kept_rows = []
    category_counter = {}
    n_read = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, dtype=str,
                              encoding="utf-8-sig", encoding_errors="replace",
                              chunksize=100_000, on_bad_lines="skip"):
        n_read += len(chunk)
        if state_col and state_filter:
            chunk = chunk[chunk[state_col].str.contains(state_filter, case=False, na=False)]

        cat_series = chunk[cat_col].fillna("") if cat_col else pd.Series([""] * len(chunk), index=chunk.index)
        qtype_series = chunk[qtype_col].fillna("") if qtype_col else pd.Series([""] * len(chunk), index=chunk.index)
        combined_cat = cat_series.astype(str) + " " + qtype_series.astype(str)

        for raw_text, raw_cat in zip(chunk[text_col].fillna(""), combined_cat):
            text = " ".join(str(raw_text).split())  # collapse whitespace
            if not text or not _has_alpha_content(text):
                continue
            n_words = len(text.split())
            if n_words < min_words or n_words > max_words:
                continue
            label = map_to_intent_label(raw_cat, text)
            if label == FALLBACK_LABEL:
                category_counter[raw_cat] = category_counter.get(raw_cat, 0) + 1
            kept_rows.append({"text": text, "label": label, "language": "en"})

    print(f"[read] {n_read} raw rows scanned")
    print(f"[filter] {len(kept_rows)} rows kept after text/word-count cleaning"
          + (f" and state filter '{state_filter}'" if state_filter else ""))

    df = pd.DataFrame(kept_rows).drop_duplicates(subset=["text"])
    print(f"[dedupe] {len(df)} unique rows remain")

    # Surface which raw category strings actually ended up in the fallback
    # bucket after per-row resolution (including the text-based fallback for
    # ambiguous categories), so the mapping rules can be refined if a big
    # chunk still lands here.
    fallback_categories = category_counter
    if fallback_categories:
        top_fallback = sorted(fallback_categories.items(), key=lambda kv: -kv[1])[:10]
        print(f"\n[fallback] top raw category values mapped to '{FALLBACK_LABEL}' "
              "(add a rule to LABEL_RULES if one of these deserves its own label):")
        for cat, count in top_fallback:
            print(f"    {count:>7}  {cat!r}")

    return df


def balance_labels(df: pd.DataFrame, max_per_label: int, min_per_label: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    balanced = []
    print("\n[labels] distribution after mapping:")
    for label in config.INTENT_LABELS:
        subset = df[df["label"] == label]
        n = len(subset)
        if n > max_per_label:
            subset = subset.sample(n=max_per_label, random_state=seed)
        flag = "  <-- fewer than min-per-label, classifier will be weak on this class" \
            if n < min_per_label else ""
        print(f"    {label:<22} {n:>6} available -> {len(subset):>6} kept{flag}")
        balanced.append(subset)
    result = pd.concat(balanced, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the downloaded raw KCC CSV")
    parser.add_argument("--output", default=config.INTENTS_CSV, help="Where to write the final intents.csv")
    parser.add_argument("--state", default=None, help='Optional state filter, e.g. "TAMIL NADU"')
    parser.add_argument("--max-per-label", type=int, default=300,
                         help="Cap rows per intent label so no class dominates training")
    parser.add_argument("--min-per-label", type=int, default=15,
                         help="Warn if a label ends up with fewer real rows than this")
    parser.add_argument("--min-words", type=int, default=3, help="Drop queries shorter than this")
    parser.add_argument("--max-words", type=int, default=40, help="Drop queries longer than this")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    df = load_and_map(args.input, args.state, args.min_words, args.max_words)
    if df.empty:
        raise SystemExit("No usable rows survived cleaning - check --state spelling "
                          "and the column auto-detection output above.")

    df = balance_labels(df, args.max_per_label, args.min_per_label, args.seed)

    if os.path.exists(args.output):
        backup_path = os.path.join(os.path.dirname(args.output), "intents_synthetic_backup.csv")
        if not os.path.exists(backup_path):
            shutil.copy(args.output, backup_path)
            print(f"\n[backup] previous (synthetic) intents.csv preserved -> {backup_path}")

    df[["text", "label", "language"]].to_csv(args.output, index=False)
    print(f"\n[done] wrote {len(df)} real-query rows -> {args.output}")
    print("Next steps:\n"
          "  python etl/extract.py\n"
          "  python etl/transform.py\n"
          "  python train_classifier.py")


if __name__ == "__main__":
    main()
