"""
AgroVox - ETL Pipeline Orchestrator.

Runs Extract -> Transform -> Load in sequence, printing a data-lineage
summary (row counts at every stage) - the ETL pattern used throughout data
engineering to make sure nothing silently vanishes or duplicates between
stages.

Run: python etl/etl_pipeline.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from etl.extract import extract_raw_farmer_queries
from etl.transform import load_raw, transform
from etl.load import load_to_sqlite, load_to_training_corpus


def run_pipeline(retrain_after_load: bool = False):
    print("=" * 70)
    print("AgroVox ETL Pipeline")
    print("=" * 70)
    t0 = time.time()

    # ---- EXTRACT ----------------------------------------------------------
    print("\n[1/3] EXTRACT — pulling raw (messy) farmer query log...")
    raw_path = extract_raw_farmer_queries()

    # ---- TRANSFORM ----------------------------------------------------------
    print("\n[2/3] TRANSFORM — cleaning, validating, feature-engineering...")
    raw_df = load_raw(raw_path)
    clean_df, report = transform(raw_df)
    clean_csv = os.path.join(config.PROCESSED_DATA_DIR, "farmer_queries_clean.csv")
    clean_df.to_csv(clean_csv, index=False)
    report_path = os.path.join(config.PROCESSED_DATA_DIR, "transform_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- LOAD ----------------------------------------------------------
    print("\n[3/3] LOAD — persisting cleaned data into SQLite + training corpus...")
    n_sqlite = load_to_sqlite(clean_df)
    corpus_result = load_to_training_corpus(clean_df)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("ETL PIPELINE SUMMARY (data lineage)")
    print("=" * 70)
    for stage in report["stages"]:
        print(f"  {stage['stage']:38s} -> {stage['rows']:5d} rows")
    print(f"  {'LOADED into SQLite':38s} -> {n_sqlite:5d} rows")
    print(f"  {'training corpus: genuinely new rows':38s} -> {corpus_result['genuinely_new_rows']:5d} rows")
    print(f"  {'training corpus: total after merge':38s} -> {corpus_result['total_rows']:5d} rows")
    print(f"\nRetention rate: {report['retention_rate']*100:.1f}% "
          f"({report['total_rows_dropped']} rows dropped as unrecoverable/invalid)")
    print(f"Completed in {elapsed:.1f}s")

    if retrain_after_load:
        print("\nRetraining classifier on the updated training corpus...")
        os.system(f"{sys.executable} {os.path.join(config.BASE_DIR, 'evaluation', 'classification_metrics.py')}")

    return {"raw_path": raw_path, "clean_csv": clean_csv, "report": report,
             "n_sqlite": n_sqlite, "corpus_result": corpus_result, "elapsed_seconds": elapsed}


if __name__ == "__main__":
    run_pipeline(retrain_after_load="--retrain" in sys.argv)
