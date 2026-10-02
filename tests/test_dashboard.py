from pathlib import Path

from src.common.config import resolve

from app.dashboard import load_pipeline_data


def test_dashboard_load_pipeline_data_merges_incidents():
    processed_dir = resolve("data/processed")
    data = load_pipeline_data(processed_dir)

    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "incident" in first
    assert "explanation" in first
    assert "remediation" in first
    assert "incident_id" in first["incident"]
