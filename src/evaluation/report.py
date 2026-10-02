import json
from pathlib import Path

from src.common.io import save_json
from src.common.logging_utils import get_logger
from src.evaluation.llm_heuristics import evaluate_llm_grounding
from src.evaluation.profiler import profile_module

log = get_logger(__name__)


def generate_final_report(processed_dir: Path, results_dir: Path, max_hallucinations: int = 0):
    """Aggregate profiling, detection metrics, LLM grounding checks, and end-to-end integration status."""
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)

    report = {
        "system_profiling": getattr(profile_module, "stats", {}),
        "detection_metrics": {},
        "llm_evaluation": {},
        "integration_status": "UNKNOWN",
    }

    m2_summary_path = processed_dir / "anomaly_metrics" / "summary.json"
    if m2_summary_path.exists():
        report["detection_metrics"]["metric_anomaly"] = json.loads(m2_summary_path.read_text(encoding="utf-8"))

    m3_summary_path = processed_dir / "anomaly_logs" / "summary.json"
    if m3_summary_path.exists():
        report["detection_metrics"]["log_anomaly"] = json.loads(m3_summary_path.read_text(encoding="utf-8"))

    ctx_path = processed_dir / "context_builder" / "context_packages.json"
    exp_path = processed_dir / "llm_explanations" / "explanations.json"

    if ctx_path.exists() and exp_path.exists():
        contexts = {item["incident_id"]: item for item in json.loads(ctx_path.read_text(encoding="utf-8")) if "incident_id" in item}
        explanations = json.loads(exp_path.read_text(encoding="utf-8"))

        total_hallucinations = 0
        total_structural_failures = 0

        for explanation in explanations:
            incident_id = explanation.get("incident_id")
            context_package = contexts.get(incident_id, {})
            eval_result = evaluate_llm_grounding(explanation, context_package)
            report["llm_evaluation"][incident_id] = eval_result
            total_hallucinations += eval_result["hallucination_count"]
            if not eval_result["is_structurally_valid"]:
                total_structural_failures += 1

        report["llm_evaluation"]["aggregate"] = {
            "total_incidents_evaluated": len(explanations),
            "total_hallucinated_entities": total_hallucinations,
            "structural_failures": total_structural_failures,
            "pass_threshold_met": total_hallucinations <= max_hallucinations and total_structural_failures == 0,
        }

    required_files = [
        "correlation/incidents.json",
        "context_builder/context_packages.json",
        "llm_explanations/explanations.json",
        "remediation/remediations.json",
    ]
    all_exist = all((processed_dir / rel).exists() for rel in required_files)
    report["integration_status"] = "PASS" if all_exist else "FAIL"

    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, results_dir / "final_report.json")
    log.info(f"Final report generated at {results_dir / 'final_report.json'}")
    return report
