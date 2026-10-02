# Aether: AI SRE Copilot

Detects anomalies in server metrics (SMD) and logs (HDFS), correlates them into
incidents, explains them with a pretrained LLM, and recommends safe troubleshooting steps.
Recommendations only: Aether never executes remediation.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest -v

## Module 1: ingestion
    python -m src.ingestion.pipeline smd
    python -m src.ingestion.pipeline hdfs --max-lines 500000   # quick check; omit for full run

Paths come from `configs/default.yaml` (override with `--config` or `$AETHER_CONFIG`).
Outputs go to `data/processed/{smd,hdfs}/`.
# AI-SRE

# AI-SRE
