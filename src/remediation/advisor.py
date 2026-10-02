from src.common.logging_utils import get_logger
from src.remediation.safety import categorize_action, is_action_safe

log = get_logger(__name__)


def generate_safe_remediations(explanation: dict, config: dict) -> dict:
    """Filter and categorize LLM recommendations into safe remediation actions."""
    incident_id = explanation.get("incident_id", "UNKNOWN")
    raw_recommendations = explanation.get("recommended_investigation", [])

    if not isinstance(raw_recommendations, list):
        log.warning(f"Incident {incident_id}: recommendations is not a list. Defaulting to empty.")
        raw_recommendations = []

    forbidden = config.get("forbidden_keywords", [])
    categories_map = config.get("categories", {})

    safe_actions = []
    blocked_actions = []

    for action in raw_recommendations:
        if not isinstance(action, str) or not action.strip():
            continue

        is_safe, reason = is_action_safe(action, forbidden)
        category = categorize_action(action, categories_map)

        if is_safe:
            safe_actions.append({
                "action": action.strip(),
                "category": category,
                "is_destructive": False,
            })
        else:
            log.warning(f"Incident {incident_id}: Blocked unsafe action '{action}' (Reason: {reason})")
            blocked_actions.append({
                "action": action.strip(),
                "blocked_reason": reason,
            })

    if not safe_actions:
        safe_actions.append({
            "action": "Review the observed evidence and correlate with recent system changes.",
            "category": "general_investigation",
            "is_destructive": False,
        })

    return {
        "incident_id": incident_id,
        "safe_recommendations": safe_actions,
        "blocked_recommendations": blocked_actions,
        "total_safe": len(safe_actions),
        "total_blocked": len(blocked_actions),
    }
