import pytest

from src.context_builder.builder import build_incident_context
from src.context_builder.pipeline import run_context_builder


def make_mock_incident():
    return {
        "incident_id": "INC-0001",
        "start_time": "2020-01-01T10:00:00",
        "end_time": "2020-01-01T10:05:00",
        "duration_minutes": 5.0,
        "severity_score": 0.92,
        "signals": [
            {"type": "metric_anomaly", "entity_id": "machine-1-1", "score": 0.92},
            {"type": "log_anomaly", "trace_id": "blk_123", "score": 0.85},
        ],
    }


def test_build_context_formatting():
    incident = make_mock_incident()
    traces_map = {"blk_123": ["E1", "E2", "E5", "E29", "END"]}

    context = build_incident_context(incident, traces_map, max_events=50)

    assert context["incident_id"] == "INC-0001"
    assert "10:00:00" in context["time_window"]
    assert context["overall_severity_score"] == 0.92
    assert len(context["evidence"]["metrics"]) == 1
    assert context["evidence"]["metrics"][0]["entity_id"] == "machine-1-1"
    assert len(context["evidence"]["logs"]) == 1
    assert "E1 -> E2 -> E5 -> E29 -> END" in context["evidence"]["logs"][0]["event_sequence"]
    assert "Concurrent metric and log anomalies" in context["correlation_summary"]
    assert "instruction_to_llm" in context


def test_build_context_truncation():
    incident = make_mock_incident()
    long_seq = [f"E{i}" for i in range(100)] + ["END"]
    traces_map = {"blk_123": long_seq}

    context = build_incident_context(incident, traces_map, max_events=10)
    seq_str = context["evidence"]["logs"][0]["event_sequence"]

    assert "(truncated)" in seq_str
    assert seq_str.count("->") < 15


@pytest.mark.real_data
def test_real_context_builder_smoke(tmp_path):
    from src.common.config import load_config, resolve

    cfg = load_config()
    processed_dir = resolve(cfg["data"]["processed_dir"])

    inc_path = processed_dir / "correlation" / "incidents.json"
    if not inc_path.exists():
        pytest.skip("Module 4 incidents.json not found.")

    packages = run_context_builder(processed_dir, tmp_path, max_events=30)

    assert len(packages) > 0
    assert (tmp_path / "context_packages.json").exists()

    pkg = packages[0]
    assert "incident_id" in pkg
    assert "evidence" in pkg
    assert "instruction_to_llm" in pkg
