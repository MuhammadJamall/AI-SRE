import pytest

from src.evaluation.llm_heuristics import evaluate_llm_grounding, extract_entities_from_text
from src.evaluation.report import generate_final_report


def test_entity_extraction():
    text = "machine-1-1 had high CPU. Trace blk_123 failed at E5."
    entities = extract_entities_from_text(text)
    assert "machine-1-1" in entities
    assert "blk_123" in entities
    assert "E5" in entities


def test_llm_heuristic_grounding():
    context = {
        "incident_id": "INC-001",
        "evidence": {
            "metrics": [{"entity_id": "machine-1-1"}],
            "logs": [{"trace_id": "blk_123", "event_sequence": ["E1", "E2"]}],
        },
    }

    good_exp = {
        "incident_id": "INC-001",
        "incident_summary": "Issue on machine-1-1 and blk_123.",
        "observed_evidence": ["E1 led to E2"],
        "probable_cause": "Unknown",
        "confidence_uncertainty": "Low",
        "recommended_investigation": ["Check logs"],
    }
    res_good = evaluate_llm_grounding(good_exp, context)
    assert res_good["is_factually_grounded"] is True
    assert res_good["is_structurally_valid"] is True

    bad_exp = {
        "incident_id": "INC-001",
        "incident_summary": "Issue on machine-9-9 and database.",
        "observed_evidence": ["E1 led to E2"],
    }
    res_bad = evaluate_llm_grounding(bad_exp, context)
    assert res_bad["is_factually_grounded"] is False
    assert "machine-9-9" in res_bad["hallucinated_entities"]
    assert res_bad["is_structurally_valid"] is False


@pytest.mark.real_data
def test_e2e_integration_and_report(tmp_path):
    from src.common.config import load_config, resolve

    cfg = load_config()
    processed_dir = resolve(cfg["data"]["processed_dir"])

    if not (processed_dir / "correlation" / "incidents.json").exists():
        pytest.skip("Modules 4-7 outputs not found. Run pipeline first.")

    report = generate_final_report(processed_dir, tmp_path, max_hallucinations=0)

    assert report["integration_status"] == "PASS"
    assert "system_profiling" in report
    assert "llm_evaluation" in report
    assert (tmp_path / "final_report.json").exists()
