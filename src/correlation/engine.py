import pandas as pd


def align_timestamps(metric_df: pd.DataFrame, log_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shifts HDFS log timestamps to align with the start of the SMD metric test set."""
    metric_df = metric_df.copy()
    log_df = log_df.copy()

    if "timestamp" not in metric_df.columns and "start_time" in metric_df.columns:
        metric_df = metric_df.rename(columns={"start_time": "timestamp"})
    if "timestamp" not in log_df.columns and "start_time" in log_df.columns:
        log_df = log_df.rename(columns={"start_time": "timestamp"})

    if metric_df.empty or log_df.empty:
        return metric_df, log_df

    metric_df["timestamp"] = pd.to_datetime(metric_df["timestamp"])
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

    smd_start = metric_df["timestamp"].min()
    hdfs_start = log_df["timestamp"].min()

    if pd.isna(smd_start) or pd.isna(hdfs_start):
        return metric_df, log_df

    delta = smd_start - hdfs_start
    shifted = log_df.copy()
    shifted["timestamp"] = shifted["timestamp"] + delta
    return metric_df, shifted


def _build_incident(signals_df_list, start_time, end_time, inc_id) -> dict:
    signals = []
    max_score = 0.0
    metric_count = 0
    log_count = 0

    for row in signals_df_list:
        sig = {
            "type": f"{row.get('source', 'signal')}_anomaly",
            "timestamp": row["timestamp"].isoformat(),
            "score": float(row.get("anomaly_score", 0.0)),
        }
        if row.get("source") == "metric":
            sig["entity_id"] = row.get("entity_id")
            metric_count += 1
        else:
            sig["trace_id"] = row.get("trace_id") or row.get("entity_id")
            log_count += 1

        if sig["score"] > max_score:
            max_score = sig["score"]
        signals.append(sig)

    return {
        "incident_id": f"INC-{inc_id:04d}",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": (end_time - start_time).total_seconds() / 60.0,
        "severity_score": max_score,
        "metric_signals_count": metric_count,
        "log_signals_count": log_count,
        "signals": signals,
    }


def group_incidents(df: pd.DataFrame, max_gap_minutes: float) -> list[dict]:
    """Groups anomalies into incidents based on temporal proximity."""
    if df.empty:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)
    incidents = []
    current_signals = []
    current_start = df.iloc[0]["timestamp"]
    current_end = df.iloc[0]["timestamp"]

    for _, row in df.iterrows():
        gap_minutes = (row["timestamp"] - current_end).total_seconds() / 60.0
        if gap_minutes > max_gap_minutes:
            incidents.append(_build_incident(current_signals, current_start, current_end, len(incidents) + 1))
            current_signals = []
            current_start = row["timestamp"]

        current_signals.append(row)
        current_end = row["timestamp"]

    if current_signals:
        incidents.append(_build_incident(current_signals, current_start, current_end, len(incidents) + 1))

    return incidents
