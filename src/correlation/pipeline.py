from pathlib import Path

import pandas as pd

from src.common.io import save_json, save_parquet
from src.common.logging_utils import get_logger
from src.correlation.data import load_log_anomalies, load_metric_anomalies
from src.correlation.engine import align_timestamps, group_incidents

log = get_logger(__name__)


def run_correlation(processed_dir: Path, max_gap_minutes: float, results_dir: Path, log_model: str = "lstm"):
    log.info("Loading metric anomalies...")
    metric_anoms = load_metric_anomalies(processed_dir)
    log.info(f"Loaded {len(metric_anoms)} metric anomalies.")

    log.info(f"Loading log anomalies (model={log_model})...")
    log_anoms = load_log_anomalies(processed_dir, model=log_model)
    log.info(f"Loaded {len(log_anoms)} log anomalies.")

    metric_anoms, log_anoms = align_timestamps(metric_anoms, log_anoms)
    combined = pd.concat([metric_anoms, log_anoms], ignore_index=True)

    log.info(f"Grouping into incidents with max gap = {max_gap_minutes} minutes...")
    incidents = group_incidents(combined, max_gap_minutes)
    log.info(f"Created {len(incidents)} incidents.")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(incidents, results_dir / "incidents.json")

    summary = [{
        "incident_id": inc["incident_id"],
        "start_time": inc["start_time"],
        "end_time": inc["end_time"],
        "duration_minutes": inc["duration_minutes"],
        "severity_score": inc["severity_score"],
        "metric_signals_count": inc["metric_signals_count"],
        "log_signals_count": inc["log_signals_count"],
        "total_signals": len(inc["signals"]),
    } for inc in incidents]

    save_json(summary, results_dir / "summary.json")
    save_parquet(pd.DataFrame(summary), results_dir / "incidents.parquet")

    return incidents
