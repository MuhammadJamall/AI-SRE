"""Compare ingested traces with the official HDFS Event_traces.csv.

Run after a full ingest:
    python scripts/verify_hdfs_vs_event_traces.py data/raw/hdfs data/processed/hdfs
"""
import sys
from pathlib import Path

import pandas as pd


def main(raw_dir, processed_dir):
    ours = (
        pd.read_parquet(Path(processed_dir) / "traces.parquet")
        .set_index("trace_id")["event_sequence"]
    )
    official = pd.read_csv(
        Path(raw_dir) / "Event_traces.csv",
        usecols=["BlockId", "Features"],
    )

    missing = mismatches = 0
    examples = []
    for block_id, features in zip(official["BlockId"], official["Features"]):
        expected = [
            event.strip().strip("'\"")
            for event in str(features).strip().strip("[]").split(",")
            if event.strip()
        ]
        actual = ours.get(block_id)
        if actual is None:
            missing += 1
        elif list(actual) != expected:
            mismatches += 1
            if len(examples) < 3:
                examples.append((block_id, expected[:12], list(actual)[:12]))

    total = len(official)
    matched = total - missing - mismatches
    rate = matched / total if total else 0.0
    print(
        f"official traces: {total}\n"
        f"missing in ours: {missing}\n"
        f"sequence mismatches: {mismatches}\n"
        f"exact match rate: {rate:.4%}"
    )
    for example in examples:
        print("EXAMPLE", example)


if __name__ == "__main__":
    main(*sys.argv[1:3])
