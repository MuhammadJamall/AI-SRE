import time
import tracemalloc
from contextlib import contextmanager

from src.common.logging_utils import get_logger

log = get_logger(__name__)


@contextmanager
def profile_module(module_name: str):
    """Track runtime and peak memory for one module execution block."""
    log.info(f"--- Starting {module_name} ---")
    tracemalloc.start()
    start_time = time.time()

    try:
        yield
    finally:
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        log.info(
            f"--- Finished {module_name} in {duration:.2f}s | Peak Memory: {peak / 1024 / 1024:.2f} MB ---"
        )

        if not hasattr(profile_module, "stats"):
            profile_module.stats = {}
        profile_module.stats[module_name] = {
            "duration_seconds": round(duration, 3),
            "peak_memory_mb": round(peak / 1024 / 1024, 2),
            "current_memory_mb": round(current / 1024 / 1024, 2),
        }
