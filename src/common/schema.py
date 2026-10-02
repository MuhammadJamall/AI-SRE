"""Standard Aether data formats. Every downstream module relies only on these."""

# Metrics: timestamp, entity_id, <one numeric column per metric>, label, split
METRIC_REQUIRED = ["timestamp", "entity_id", "label", "split"]

# Logs (one row per log line)
LOG_REQUIRED = ["timestamp", "log_message", "event_id", "trace_id", "label", "split"]

# Traces (one row per session/block: ordered event sequence)
TRACE_REQUIRED = ["trace_id", "start_time", "event_sequence", "label", "split"]

VALID_SPLITS = {"train", "test"}
VALID_LABELS = {0, 1}