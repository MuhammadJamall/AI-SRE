import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path=None) -> dict:
    """Load YAML config. Order: explicit path > $AETHER_CONFIG > configs/default.yaml."""
    path = Path(path or os.environ.get("AETHER_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p) -> Path:
    """Relative paths are resolved against the project root (no personal absolute paths)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p