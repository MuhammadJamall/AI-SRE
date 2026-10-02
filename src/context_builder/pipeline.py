from pathlib import Path

from src.common.io import save_json
from src.common.logging_utils import get_logger
from src.context_builder.builder import build_incident_context
from src.context_builder.data import load_incidents, load_traces_for_ids

log = get_logger(__name__)


def run_context_builder(processed_dir: Path, results_dir: Path, max_events: int = 50):
    log.info("Loading incidents from Module 4...")
    incidents = load_incidents(processed_dir)
    log.info(f"Loaded {len(incidents)} incidents.")

    if not incidents:
        log.warning("No incidents to process. Exiting.")
        return []

    trace_ids = set()
    for incident in incidents:
        for signal in incident.get("signals", []):
            if signal.get("type") == "log_anomaly" and signal.get("trace_id"):
                trace_ids.add(str(signal["trace_id"]))

    log.info(f"Fetching event sequences for {len(trace_ids)} unique trace IDs...")
    traces_map = load_traces_for_ids(processed_dir, list(trace_ids))
    log.info(f"Successfully retrieved {len(traces_map)} trace sequences.")

    context_packages = [
        build_incident_context(inc, traces_map, max_events=max_events)
        for inc in incidents
    ]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(context_packages, results_dir / "context_packages.json")
    log.info(f"Saved {len(context_packages)} context packages to {results_dir / 'context_packages.json'}")

    return context_packages
