"""
AgroVox - ETL Step 0: Simulated raw data source.

In a real deployment, "raw" farmer query logs would arrive messy: typed by
many different people, on many devices, with inconsistent casing, stray
whitespace, duplicate submissions (double-tapped "Send"), missing metadata
(some app versions didn't log the language), and a few completely empty/junk
rows (accidental submissions). This script generates a realistic MESSY raw
log at data/raw/farmer_queries_raw.csv, deliberately reusing the clean
intents.csv content as its base and then injecting exactly the kinds of
defects a real ETL pipeline has to handle - so etl/transform.py has genuine
cleaning work to do, and eda/exploratory_analysis.py has genuine data
quality issues to surface.

This is the "Extract" landing zone: data as it exists BEFORE any cleaning.

Run: python etl/extract.py
"""
import os
import sys
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

random.seed(config.RANDOM_SEED + 1)


def _messify_casing(text: str) -> str:
    choice = random.random()
    if choice < 0.15:
        return text.upper()
    if choice < 0.30:
        return text.lower()
    if choice < 0.35:
        return text.title()
    return text


def _add_whitespace_noise(text: str) -> str:
    if random.random() < 0.25:
        text = "   " + text
    if random.random() < 0.25:
        text = text + "   "
    if random.random() < 0.15:
        text = text.replace(" ", "  ", 1)  # one doubled space
    return text


def extract_raw_farmer_queries(out_path: str = None, duplicate_rate: float = 0.08,
                                 missing_language_rate: float = 0.20,
                                 empty_row_rate: float = 0.03) -> str:
    """Reads the clean, template-generated data/intents.csv and produces a
    deliberately messy 'raw log' CSV simulating real-world data collection
    defects, written to data/raw/farmer_queries_raw.csv.

    Returns the output path. This is intentionally a *superset* row count
    of intents.csv (duplicates + junk rows added), which is exactly what a
    real Extract step hands to Transform.
    """
    out_path = out_path or os.path.join(config.RAW_DATA_DIR, "farmer_queries_raw.csv")

    if not os.path.exists(config.INTENTS_CSV):
        raise FileNotFoundError(
            f"{config.INTENTS_CSV} not found - run data/generate_intents.py first.")

    import csv as _csv
    with open(config.INTENTS_CSV, newline="", encoding="utf-8") as f:
        base_rows = list(_csv.DictReader(f))

    raw_rows = []
    log_id = 1000

    for row in base_rows:
        text = row["text"]
        label = row["label"]
        language = row["language"]

        # 1) normal row, but with real-world casing/whitespace noise
        noisy_text = _add_whitespace_noise(_messify_casing(text))
        include_lang = random.random() > missing_language_rate
        raw_rows.append({
            "log_id": log_id, "raw_text": noisy_text, "label": label,
            "language": language if include_lang else "",
            "device": random.choice(["android-app-v1", "android-app-v2", "web", "ussd-sms", ""]),
        })
        log_id += 1

        # 2) occasionally duplicate the submission (double-tap / retry)
        if random.random() < duplicate_rate:
            raw_rows.append({
                "log_id": log_id, "raw_text": noisy_text, "label": label,
                "language": language if include_lang else "",
                "device": random.choice(["android-app-v1", "android-app-v2", "web"]),
            })
            log_id += 1

    # 3) inject a handful of completely empty / junk rows (accidental sends,
    #    logging glitches) - a real pipeline must detect and drop these
    n_junk = max(1, int(len(raw_rows) * empty_row_rate))
    for _ in range(n_junk):
        junk_kind = random.choice(["empty", "whitespace_only", "punctuation_only", "null_label"])
        if junk_kind == "empty":
            raw_text, label = "", random.choice(config.INTENT_LABELS)
        elif junk_kind == "whitespace_only":
            raw_text, label = "     ", random.choice(config.INTENT_LABELS)
        elif junk_kind == "punctuation_only":
            raw_text, label = "???...!!", random.choice(config.INTENT_LABELS)
        else:
            raw_text, label = random.choice(base_rows)["text"], ""
        raw_rows.append({
            "log_id": log_id, "raw_text": raw_text, "label": label,
            "language": "", "device": "",
        })
        log_id += 1

    random.shuffle(raw_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["log_id", "raw_text", "label", "language", "device"])
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"[EXTRACT] Wrote {len(raw_rows)} raw (messy) rows -> {out_path}")
    print(f"[EXTRACT]   base clean rows: {len(base_rows)}")
    print(f"[EXTRACT]   + duplicates injected: ~{int(len(base_rows)*duplicate_rate)}")
    print(f"[EXTRACT]   + junk/empty rows injected: {n_junk}")
    print(f"[EXTRACT]   ~{missing_language_rate*100:.0f}% of rows have a missing `language` field")
    return out_path


if __name__ == "__main__":
    extract_raw_farmer_queries()
