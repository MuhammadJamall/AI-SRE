import pandas as pd
import pytest

from src.correlation.data import load_log_anomalies, load_metric_anomalies
from src.correlation.engine import _build_incident, align_timestamps, group_incidents
from src.correlation.pipeline import run_correlation


def make_mock_metric_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 10:00:00", "2020-01-01 10:02:00", "2020-01-01 12:00:00"]),
        "entity_id": ["m1", "m1", "m2"],
        "anomaly_score": [0.9, 0.8, 0.95],
        "is_anomaly": [True, True, True],
    })


def make_mock_log_df():
    return pd.DataFrame({
        "start_time": pd.to_datetime(["2008-11-09 10:01:00", "2008-11-09 10:03:00", "2008-11-09 14:00:00"]),
        "trace_id": ["blk_1", "blk_2", "blk_3"],
        "anomaly_score": [0.85, 0.75, 0.99],
        "is_anomaly": [True, True, True],
    })


def test_align_timestamps():
    m_df, l_df = make_mock_metric_df(), make_mock_log_df()
    m_aligned, l_aligned = align_timestamps(m_df, l_df)
    assert l_aligned.iloc[0]["timestamp"] == pd.Timestamp("2020-01-01 10:00:00")
    assert l_aligned.iloc[1]["timestamp"] == pd.Timestamp("2020-01-01 10:02:00")
    assert len(m_aligned) == len(m_df)
    assert len(l_aligned) == len(l_df)


def test_group_incidents_temporal_gap():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 10:00:00", "2020-01-01 10:02:00", "2020-01-01 10:10:00"]),
        "source": ["metric", "log", "metric"],
        "entity_id": ["m1", "blk_1", "m1"],
        "anomaly_score": [0.9, 0.8, 0.95],
        "is_anomaly": [True, True, True],
    })
    incidents = group_incidents(df, max_gap_minutes=5.0)
    assert len(incidents) == 2
    assert incidents[0]["metric_signals_count"] == 1
    assert incidents[0]["log_signals_count"] == 1


def test_build_incident_schema():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 10:00:00"]),
        "source": ["metric"],
        "entity_id": ["m1"],
        "anomaly_score": [0.9],
    })
    inc = _build_incident([df.iloc[0]], df.iloc[0]["timestamp"], df.iloc[0]["timestamp"], 1)
    assert inc["incident_id"] == "INC-0001"
    assert inc["signals"][0]["type"] == "metric_anomaly"


@pytest.mark.real_data
def test_real_correlation_smoke(tmp_path):
    from src.common.config import load_config, resolve

    cfg = load_config()
    processed_dir = resolve(cfg["data"]["processed_dir"])

    if not (processed_dir / "anomaly_metrics" / "predictions.parquet").exists() or not (
        processed_dir / "anomaly_logs" / "predictions_lstm.parquet"
    ).exists():
        pytest.skip("Module 2 or 3 outputs not found.")

    incidents = run_correlation(processed_dir, max_gap_minutes=5.0, results_dir=tmp_path, log_model="lstm")
    assert len(incidents) > 0
    assert (tmp_path / "incidents.json").exists()
