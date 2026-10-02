"""Validation + light cleaning shared by all ingestion adapters."""
import numpy as np
import pandas as pd

from src.common.schema import (
    LOG_REQUIRED, METRIC_REQUIRED, TRACE_REQUIRED, VALID_LABELS, VALID_SPLITS,
)


class ValidationError(Exception):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def _check_columns(df: pd.DataFrame, required: list) -> None:
    missing = [c for c in required if c not in df.columns]
    _require(not missing, f"Missing required columns: {missing}")
    _require(len(df) > 0, "DataFrame is empty")


def _check_timestamp(df: pd.DataFrame, col: str = "timestamp") -> None:
    _require(pd.api.types.is_datetime64_any_dtype(df[col]),
             f"'{col}' must be datetime64, got {df[col].dtype}")
    _require(df[col].notna().all(), f"'{col}' contains NaT (invalid timestamps)")


def _check_label_split(df: pd.DataFrame) -> None:
    _require(set(df["label"].unique()) <= VALID_LABELS,
             f"Labels must be in {VALID_LABELS}, got {set(df['label'].unique())}")
    _require(set(df["split"].unique()) <= VALID_SPLITS,
             f"Splits must be in {VALID_SPLITS}, got {set(df['split'].unique())}")


# ---------------------------------------------------------------- metrics
def metric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in METRIC_REQUIRED]


def count_missing(df: pd.DataFrame) -> int:
    cols = metric_columns(df)
    vals = df[cols].replace([np.inf, -np.inf], np.nan)
    return int(vals.isna().sum().sum())


def clean_metrics(df: pd.DataFrame):
    """Sort, drop duplicate (entity, timestamp), fill NaN/inf. Returns (df, report)."""
    df = df.copy()
    cols = metric_columns(df)
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    n_missing = int(df[cols].isna().sum().sum())
    n_before = len(df)
    df = df.drop_duplicates(["entity_id", "timestamp"], keep="first")
    df = df.sort_values(["entity_id", "timestamp"], kind="stable").reset_index(drop=True)
    if n_missing:
        df[cols] = df.groupby("entity_id")[cols].transform(lambda s: s.ffill().bfill())
        df[cols] = df[cols].fillna(0.0)
    return df, {"missing_values_filled": n_missing,
                "duplicate_rows_dropped": n_before - len(df)}


def check_no_time_leakage(df: pd.DataFrame) -> None:
    """For every entity, all train timestamps must precede all test timestamps."""
    for entity, g in df.groupby("entity_id"):
        tr = g.loc[g["split"] == "train", "timestamp"]
        te = g.loc[g["split"] == "test", "timestamp"]
        if len(tr) and len(te):
            _require(tr.max() < te.min(),
                     f"Train/test time leakage in entity '{entity}'")


def validate_metrics(df: pd.DataFrame) -> dict:
    _check_columns(df, METRIC_REQUIRED)
    _check_timestamp(df)
    cols = metric_columns(df)
    _require(len(cols) > 0, "No metric columns found")
    for c in cols:
        _require(pd.api.types.is_numeric_dtype(df[c]), f"Metric '{c}' is not numeric")
    vals = df[cols].to_numpy(dtype="float64")
    _require(not np.isnan(vals).any(), "NaN values present (run clean_metrics first)")
    _require(np.isfinite(vals).all(), "Infinite values present")
    _check_label_split(df)
    _require(not df.duplicated(["entity_id", "timestamp"]).any(),
             "Duplicate (entity_id, timestamp) rows")
    mono = df.groupby("entity_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing)
    _require(bool(mono.all()), "Timestamps not sorted within an entity")
    check_no_time_leakage(df)
    return {
        "rows": int(len(df)),
        "entities": int(df["entity_id"].nunique()),
        "n_metrics": len(cols),
        "anomaly_rate": float(df["label"].mean()),
        "rows_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
    }


# ------------------------------------------------------------------- logs
def validate_logs(df: pd.DataFrame) -> dict:
    _check_columns(df, LOG_REQUIRED)
    _check_timestamp(df)
    for c in ["log_message", "event_id", "trace_id"]:
        _require(df[c].notna().all(), f"Null values in '{c}'")
    _check_label_split(df)
    g = df.groupby("trace_id")
    _require(g["label"].nunique().max() == 1, "A trace has inconsistent labels")
    _require(g["split"].nunique().max() == 1, "A trace appears in both train and test (leakage)")
    return {"rows": int(len(df)), "traces": int(df["trace_id"].nunique())}


def validate_traces(df: pd.DataFrame) -> dict:
    _check_columns(df, TRACE_REQUIRED)
    _check_timestamp(df, "start_time")
    _require(df["trace_id"].is_unique, "Duplicate trace_id in traces table")
    _require(all(len(s) > 0 for s in df["event_sequence"]), "Empty event sequence found")
    _check_label_split(df)
    return {
        "traces": int(len(df)),
        "anomaly_rate": float(df["label"].mean()),
        "traces_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
    }