import json
import re

from src.common.logging_utils import get_logger

log = get_logger(__name__)


def extract_entities_from_text(text: str) -> set[str]:
    """Extract likely operational entities such as machine IDs, block IDs, and event IDs."""
    patterns = [
        r"machine-\d+-\d+",
        r"blk_[A-Za-z0-9_-]+",
        r"E\d+",
    ]
    found: set[str] = set()
    for pattern in patterns:
        found.update(re.findall(pattern, text))
    return found


def _collect_context_entities(context_package: dict) -> set[str]:
    context_text = json.dumps(context_package, ensure_ascii=False)
    return extract_entities_from_text(context_text)


def evaluate_llm_grounding(explanation: dict, context_package: dict) -> dict:
    """Check if the LLM output introduces entities or structure not grounded in the provided context."""
    context_entities = _collect_context_entities(context_package or {})
    llm_text = json.dumps(explanation, ensure_ascii=False)
    llm_entities = extract_entities_from_text(llm_text)

    hallucinated = llm_entities - context_entities

    required_keys = {
        "incident_summary",
        "observed_evidence",
        "probable_cause",
        "confidence_uncertainty",
        "recommended_investigation",
    }
    missing_keys = sorted(required_keys - set(explanation.keys()))

    return {
        "incident_id": explanation.get("incident_id"),
        "context_entities_found": len(context_entities),
        "llm_entities_generated": len(llm_entities),
        "hallucinated_entities": sorted(hallucinated),
        "hallucination_count": len(hallucinated),
        "missing_structural_keys": missing_keys,
        "is_structurally_valid": len(missing_keys) == 0,
        "is_factually_grounded": len(hallucinated) == 0,
    }
