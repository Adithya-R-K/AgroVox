"""
AgroVox - Pull real Kisan Call Centre (KCC) records from the data.gov.in OGD API.

Fetches "Kisan Call Centre (KCC) - Transcripts of farmers queries & answers"
(resource id cef25fe2-9231-4128-8aec-2c948fedd43f on data.gov.in) directly
via HTTP, paginating through results and optionally filtering server-side by
state, and writes the combined records to a single local CSV that
data/build_real_intents.py can then ingest.

Get a personal API key first (the public sample key on the page is capped at
10 records/call): log in at data.gov.in -> click "Generate API Key" on the
resource's API tab. It's instant, no approval wait.

Usage:
    python data/fetch_kcc_api.py --api-key YOUR_KEY_HERE --state "Tamil Nadu"
    python data/fetch_kcc_api.py --api-key YOUR_KEY_HERE --state "Tamil Nadu" \
        --max-records 5000 --output data/raw/kcc_queries.csv
"""
import os
import sys
import csv
import time
import argparse

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DEFAULT_RESOURCE_ID = "cef25fe2-9231-4128-8aec-2c948fedd43f"
API_BASE = "https://api.data.gov.in/resource"

# data.gov.in's server stalls/times out requests sent with the default
# `python-requests/x.x` User-Agent (confirmed: identical request completes in
# ~1s with this header vs. a 30s read-timeout without it) - looks like basic
# bot-fingerprint throttling on their side, not a real rate limit.
REQUEST_HEADERS = {"User-Agent": "curl/8.17.0"}


def fetch_records(api_key: str, resource_id: str, state: str, page_size: int,
                   max_records: int, max_retries: int = 3) -> list:
    all_records = []
    offset = 0
    while True:
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": page_size,
            "offset": offset,
        }
        if state:
            params["filters[StateName]"] = state

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(f"{API_BASE}/{resource_id}", params=params,
                                     headers=REQUEST_HEADERS, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                print(f"[warn] request failed (attempt {attempt}/{max_retries}) at offset {offset}: {exc}")
                if attempt == max_retries:
                    print("[stop] giving up after repeated failures; keeping what was fetched so far.")
                    return all_records
                time.sleep(2 * attempt)

        records = payload.get("records", [])
        if not records:
            print(f"[done] no more records at offset {offset} - stopping.")
            break

        all_records.extend(records)
        print(f"[fetch] offset {offset:>7} -> got {len(records):>5} records "
              f"(total so far: {len(all_records)})")

        offset += len(records)
        if len(records) < page_size:
            print("[done] last page was short - reached the end of matching records.")
            break
        if max_records and len(all_records) >= max_records:
            print(f"[stop] reached --max-records cap of {max_records}.")
            all_records = all_records[:max_records]
            break

        time.sleep(0.3)  # be polite to the public API

    return all_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", required=True, help="Your personal data.gov.in API key")
    parser.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID,
                         help="OGD resource id (default: the KCC transcripts dataset)")
    parser.add_argument("--state", default="TAMILNADU",
                         help='Server-side StateName filter. NOTE: this dataset stores state names as one '
                              'uppercase word with no space, e.g. "TAMILNADU", "UTTAR PRADESH" (some states '
                              'do have spaces - check with a small unfiltered query first if unsure). '
                              'Pass "" for no filter (all of India).')
    parser.add_argument("--page-size", type=int, default=1000,
                         help="Records per API call (the platform may cap this lower than what you ask for)")
    parser.add_argument("--max-records", type=int, default=20000,
                         help="Safety cap on total records fetched (0 = unlimited)")
    parser.add_argument("--output", default=os.path.join(config.RAW_DATA_DIR, "kcc_queries.csv"))
    args = parser.parse_args()

    print(f"[start] resource={args.resource_id} state_filter={args.state!r} "
          f"page_size={args.page_size} max_records={args.max_records or 'unlimited'}")
    records = fetch_records(args.api_key, args.resource_id, args.state,
                             args.page_size, args.max_records)

    if not records:
        raise SystemExit("No records were fetched - check your API key and state spelling.")

    fieldnames = list(records[0].keys())
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[done] wrote {len(records)} records -> {args.output}")
    print("Next step:\n"
          f"  python data/build_real_intents.py --input {args.output} "
          f'--state "{args.state or "TAMILNADU"}"')


if __name__ == "__main__":
    main()
