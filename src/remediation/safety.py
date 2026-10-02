import re

from src.common.logging_utils import get_logger

log = get_logger(__name__)


def is_action_safe(action: str, forbidden_keywords: list[str]) -> tuple[bool, str]:
    """Check if an action contains destructive keywords."""
    action_lower = action.lower()

    for keyword in forbidden_keywords:
        if " " in keyword:
            if keyword in action_lower:
                return False, f"Contains forbidden phrase: '{keyword}'"
        else:
            if re.search(rf"\b{re.escape(keyword)}\b", action_lower):
                return False, f"Contains forbidden keyword: '{keyword}'"

    return True, "Safe"


def categorize_action(action: str, category_keywords: dict[str, list[str]]) -> str:
    """Assign a category to the action based on keywords."""
    action_lower = action.lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in action_lower:
                return category
    return "general_investigation"
