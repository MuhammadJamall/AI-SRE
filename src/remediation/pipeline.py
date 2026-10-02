import json
from pathlib import Path

from src.common.io import save_json
from src.common.logging_utils import get_logger
from src.remediation.advisor import generate_safe_remediations

log = get_logger(__name__)


def run_remediation_advisor(explanations_dir: Path, results_dir: Path, remediation_config: dict):
    log.info("Loading LLM explanations from Module 6...")
    exp_path = Path(explanations_dir) / "explanations.json"
    if not exp_path.exists():
        raise FileNotFoundError(f"Explanations not found at {exp_path}. Run Module 6 first.")

    with exp_path.open("r", encoding="utf-8") as f:
        explanations = json.load(f)

    if not explanations:
        log.warning("No explanations to process.")
        return []

    remediations = []
    for exp in explanations:
        remediations.append(generate_safe_remediations(exp, remediation_config))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(remediations, results_dir / "remediations.json")

    total_safe = sum(r["total_safe"] for r in remediations)
    total_blocked = sum(r["total_blocked"] for r in remediations)
    log.info(f"Generated {total_safe} safe recommendations. Blocked {total_blocked} unsafe actions.")

    return remediations
