import json

import pytest

from src.llm.client import MockLLMClient
from src.llm.pipeline import run_llm_explainer
from src.llm.prompt import SYSTEM_PROMPT, build_prompt


def make_mock_context():
    return {
        "incident_id": "INC-0001",
        "time_window": "2020-01-01T10:00:00 to 2020-01-01T10:05:00",
        "overall_severity_score": 0.92,
        "correlation_summary": "Concurrent metric and log anomalies detected.",
        "evidence": {
            "metrics": [{"entity_id": "machine-1-1", "anomaly_score": 0.92}],
            "logs": [{"trace_id": "blk_123", "anomaly_score": 0.85, "event_sequence": "E1 -> E2 -> E5"}],
        },
    }


def test_prompt_builds_correctly():
    context = make_mock_context()
    prompt = build_prompt(context)

    assert SYSTEM_PROMPT in prompt
    assert "INC-0001" in prompt
    assert "machine-1-1" in prompt
    assert "blk_123" in prompt
    assert "Generate the JSON response now." in prompt


def test_mock_client_returns_valid_json():
    client = MockLLMClient()
    response = client.generate("dummy prompt")

    parsed = json.loads(response)
    assert "incident_summary" in parsed
    assert "observed_evidence" in parsed
    assert "recommended_investigation" in parsed
    assert isinstance(parsed["recommended_investigation"], list)


@pytest.mark.real_data
def test_real_llm_pipeline_smoke(tmp_path):
    from src.common.config import load_config, resolve

    cfg = load_config()
    context_dir = resolve(cfg.get("context_builder", {}).get("results_dir", "data/processed/context_builder"))
    if not (context_dir / "context_packages.json").exists():
        pytest.skip("Module 5 context_packages.json not found.")

    llm_cfg = {"provider": "mock", "results_dir": str(tmp_path)}
    explanations = run_llm_explainer(context_dir, tmp_path, llm_cfg)

    assert len(explanations) > 0
    assert (tmp_path / "explanations.json").exists()

    exp = explanations[0]
    assert "incident_id" in exp
    assert "probable_cause" in exp
