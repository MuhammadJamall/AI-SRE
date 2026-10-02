import pytest

from src.remediation.advisor import generate_safe_remediations
from src.remediation.safety import categorize_action, is_action_safe

FORBIDDEN = ["restart", "delete", "drop", "kill", "rm -rf", "reboot", "scale down", "rollback"]
CATEGORIES = {
    "log_inspection": ["log", "trace"],
    "metric_analysis": ["metric", "cpu"],
    "workload_check": ["workload", "process"],
}


def test_safety_filter_blocks_destructive():
    assert is_action_safe("Inspect the CPU logs", FORBIDDEN)[0] is True
    assert is_action_safe("Restart the server", FORBIDDEN)[0] is False
    assert is_action_safe("Delete the corrupted files", FORBIDDEN)[0] is False
    assert is_action_safe("Run rm -rf /tmp/data", FORBIDDEN)[0] is False
    assert is_action_safe("Restarting the investigation process", FORBIDDEN)[0] is True


def test_categorization():
    assert categorize_action("Check the system logs", CATEGORIES) == "log_inspection"
    assert categorize_action("Compare CPU metrics", CATEGORIES) == "metric_analysis"
    assert categorize_action("Do something else", CATEGORIES) == "general_investigation"


def test_advisor_filters_and_fallback():
    explanation = {
        "incident_id": "INC-0001",
        "recommended_investigation": [
            "Inspect the HDFS logs for trace blk_123",
            "Restart the datanode service",
            "Compare CPU metrics with baseline",
        ],
    }

    config = {"forbidden_keywords": FORBIDDEN, "categories": CATEGORIES}
    result = generate_safe_remediations(explanation, config)

    assert result["incident_id"] == "INC-0001"
    assert result["total_safe"] == 2
    assert result["total_blocked"] == 1
    assert result["blocked_recommendations"][0]["action"] == "Restart the datanode service"
    safe_actions = result["safe_recommendations"]
    assert safe_actions[0]["category"] == "log_inspection"
    assert safe_actions[1]["category"] == "metric_analysis"


def test_advisor_fallback_on_all_unsafe():
    explanation = {
        "incident_id": "INC-0002",
        "recommended_investigation": ["Delete everything", "Kill all processes"],
    }
    config = {"forbidden_keywords": FORBIDDEN, "categories": CATEGORIES}
    result = generate_safe_remediations(explanation, config)

    assert result["total_safe"] == 1
    assert result["total_blocked"] == 2
    assert "Review the observed evidence" in result["safe_recommendations"][0]["action"]


@pytest.mark.real_data
def test_real_remediation_smoke(tmp_path):
    from src.common.config import load_config, resolve
    from src.remediation.pipeline import run_remediation_advisor

    cfg = load_config()
    exp_dir = resolve(cfg.get("llm", {}).get("results_dir", "data/processed/llm_explanations"))
    if not (exp_dir / "explanations.json").exists():
        pytest.skip("Module 6 explanations.json not found.")

    rem_cfg = cfg.get("remediation", {})
    rem_cfg["results_dir"] = str(tmp_path)

    remediations = run_remediation_advisor(exp_dir, tmp_path, rem_cfg)

    assert len(remediations) > 0
    assert (tmp_path / "remediations.json").exists()

    for rem in remediations:
        for action in rem["safe_recommendations"]:
            assert action["is_destructive"] is False
