from pathlib import Path

import pandas as pd

from src.common.io import load_json


def load_incidents(processed_dir: Path) -> list[dict]:
    path = Path(processed_dir) / "correlation" / "incidents.json"
    if not path.exists():
        raise FileNotFoundError(f"Incidents not found at {path}. Run Module 4 first.")
    return load_json(path)


def load_traces_for_ids(processed_dir: Path, trace_ids: list[str]) -> dict[str, list[str]]:
    """Load only the trace events associated with incident log anomalies."""
    if not trace_ids:
        return {}

    path = Path(processed_dir) / "hdfs" / "traces.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Traces not found at {path}. Run Module 1 first.")

    trace_ids = {str(tid) for tid in trace_ids}
    df = pd.read_parquet(path, columns=["trace_id", "event_sequence"])
    filtered = df[df["trace_id"].isin(trace_ids)].copy()

    traces_map: dict[str, list[str]] = {}
    for _, row in filtered.iterrows():
        sequence = row["event_sequence"]
        if isinstance(sequence, str):
            events = [part.strip() for part in sequence.split(" -> ") if part.strip()]
        elif sequence is None:
            events = []
        else:
            events = [str(part) for part in list(sequence)]
        traces_map[str(row["trace_id"])] = events

    return traces_map
