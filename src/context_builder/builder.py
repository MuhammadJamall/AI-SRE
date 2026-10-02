from src.common.logging_utils import get_logger

log = get_logger(__name__)


def _truncate_sequence(sequence: list[str], max_events: int) -> str:
    if not sequence:
        return "UNKNOWN"
    if max_events is None or max_events <= 0:
        return " -> ".join(sequence[:10])
    if len(sequence) <= max_events:
        return " -> ".join(sequence)

    left = max_events // 2
    right = max_events - left - 1
    preview = sequence[:left] + ["... (truncated) ..."] + sequence[-right:]
    return " -> ".join(preview)


def build_incident_context(incident: dict, traces_map: dict[str, list[str]], max_events: int = 50) -> dict:
    """Create a compact, LLM-ready evidence package for one incident."""
    signals = incident.get("signals", [])
    metric_signals = [s for s in signals if s.get("type") == "metric_anomaly"]
    log_signals = [s for s in signals if s.get("type") == "log_anomaly"]

    metric_evidence = []
    for sig in metric_signals:
        metric_evidence.append({
            "entity_id": sig.get("entity_id"),
            "anomaly_score": round(float(sig.get("score", 0.0)), 4),
            "observation": "Reconstruction error exceeded the normal baseline threshold.",
        })

    log_evidence = []
    for sig in log_signals:
        trace_id = sig.get("trace_id")
        sequence = traces_map.get(trace_id, ["UNKNOWN"])
        seq_display = _truncate_sequence(sequence, max_events=max_events)
        log_evidence.append({
            "trace_id": trace_id,
            "anomaly_score": round(float(sig.get("score", 0.0)), 4),
            "event_sequence": seq_display,
            "observation": "Event sequence likelihood deviated from normal operational patterns.",
        })

    if metric_evidence and log_evidence:
        correlation_summary = "Concurrent metric and log anomalies detected within the same time window."
    elif metric_evidence:
        correlation_summary = "Isolated metric anomaly detected (no correlated log anomalies in window)."
    elif log_evidence:
        correlation_summary = "Isolated log anomaly detected (no correlated metric anomalies in window)."
    else:
        correlation_summary = "No direct metric or log anomalies were attached to this incident."

    return {
        "incident_id": incident.get("incident_id"),
        "time_window": (
            f"{incident.get('start_time')} to {incident.get('end_time')} "
            f"(Duration: {float(incident.get('duration_minutes', 0.0)):.1f} mins)"
        ),
        "overall_severity_score": round(float(incident.get("severity_score", 0.0)), 4),
        "evidence": {
            "metrics": metric_evidence,
            "logs": log_evidence,
        },
        "correlation_summary": correlation_summary,
        "instruction_to_llm": (
            "Based only on the provided evidence, generate: "
            "1. Incident Summary, 2. Observed Evidence, 3. Probable Cause, "
            "4. Confidence/Uncertainty, 5. Recommended Investigation. "
            "Do not hallucinate facts. Clearly separate observed facts from probable causes."
        ),
    }
