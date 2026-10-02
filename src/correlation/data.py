from pathlib import Path

import pandas as pd


def load_metric_anomalies(processed_dir: Path) -> pd.DataFrame:
    path = Path(processed_dir) / "anomaly_metrics" / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Metric predictions not found at {path}. Run Module 2 first.")
    df = pd.read_parquet(path)
    df = df[df["is_anomaly"] == True].copy()
    df["source"] = "metric"
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_log_anomalies(processed_dir: Path, model: str = "lstm") -> pd.DataFrame:
    path = Path(processed_dir) / "anomaly_logs" / f"predictions_{model}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Log predictions not found at {path}. Run Module 3 first.")
    df = pd.read_parquet(path)
    df = df[df["is_anomaly"] == True].copy()
    df["source"] = "log"
    if "start_time" in df.columns:
        df = df.rename(columns={"start_time": "timestamp"})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["entity_id"] = df.get("trace_id", df.get("entity_id", None))
    return df
