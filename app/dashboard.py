import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - import is optional outside dashboard runtime
    st = None

import pandas as pd

from src.common.config import load_config, resolve


if st is not None:
    st.set_page_config(page_title="Aether: AI SRE Copilot", page_icon="🛡️", layout="wide")


def load_json_safe(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_data(processed_dir: Path):
    """Load and merge Module 4, 6, and 7 outputs by incident_id."""
    processed_dir = Path(processed_dir)
    inc_path = processed_dir / "correlation" / "incidents.json"
    exp_path = processed_dir / "llm_explanations" / "explanations.json"
    rem_path = processed_dir / "remediation" / "remediations.json"

    incidents = load_json_safe(inc_path)
    explanations = {item["incident_id"]: item for item in load_json_safe(exp_path) if "incident_id" in item}
    remediations = {item["incident_id"]: item for item in load_json_safe(rem_path) if "incident_id" in item}

    merged_data = []
    for inc in incidents:
        inc_id = inc["incident_id"]
        merged_data.append({
            "incident": inc,
            "explanation": explanations.get(inc_id, {}),
            "remediation": remediations.get(inc_id, {}),
        })
    return merged_data


def main():
    if st is None:
        raise RuntimeError("streamlit is required to run the dashboard. Install dependencies with: pip install -r requirements.txt")

    st.title("🛡️ Aether: AI SRE Copilot")
    st.caption("Automated Anomaly Detection, Correlation, and Safe Root Cause Analysis.")

    try:
        cfg = load_config()
        processed_dir = resolve(cfg["data"]["processed_dir"])
    except Exception as exc:
        st.error(f"Failed to load configuration: {exc}")
        st.stop()

    with st.spinner("Loading pipeline data..."):
        data = load_pipeline_data(processed_dir)

    if not data:
        st.warning("⚠️ No incidents found. Please ensure Modules 1 through 7 have been executed successfully.")
        st.info(f"Expected data in: `{processed_dir}`")
        st.stop()

    st.sidebar.header("🔍 Incident Selector")
    incident_ids = [item["incident"]["incident_id"] for item in data]
    selected_id = st.sidebar.selectbox("Choose an Incident to Investigate:", incident_ids)

    selected_data = next((item for item in data if item["incident"]["incident_id"] == selected_id), None)
    if selected_data is None:
        st.stop()

    inc = selected_data["incident"]
    exp = selected_data["explanation"]
    rem = selected_data["remediation"]

    st.subheader(f"Incident Overview: {inc['incident_id']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Severity Score", f"{inc['severity_score']:.2f}")
    col2.metric("Duration", f"{inc['duration_minutes']:.1f} mins")
    col3.metric("Metric Signals", inc["metric_signals_count"])
    col4.metric("Log Signals", inc["log_signals_count"])

    st.divider()

    tab_timeline, tab_evidence, tab_ai, tab_remedy = st.tabs([
        "📅 Timeline & Overview",
        "🔬 Deep Evidence",
        "🧠 AI Explanation",
        "🛠️ Safe Recommendations",
    ])

    with tab_timeline:
        st.markdown(f"**Time Window:** `{inc['start_time']}` to `{inc['end_time']}`")
        st.markdown("**Correlation Summary:**")

        timeline_rows = []
        for signal in inc.get("signals", []):
            timeline_rows.append({
                "Timestamp": signal["timestamp"],
                "Type": "Metric Anomaly" if signal["type"] == "metric_anomaly" else "Log Anomaly",
                "Entity / Trace": signal.get("entity_id", signal.get("trace_id", "N/A")),
                "Anomaly Score": f"{signal['score']:.3f}",
            })

        if timeline_rows:
            df_timeline = pd.DataFrame(timeline_rows).sort_values("Timestamp")
            st.dataframe(df_timeline, use_container_width=True, hide_index=True)
        else:
            st.info("No signals recorded for this incident.")

    with tab_evidence:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📊 Metric Evidence")
            metric_signals = [signal for signal in inc.get("signals", []) if signal["type"] == "metric_anomaly"]
            if metric_signals:
                for signal in metric_signals:
                    st.markdown(f"- **Entity:** `{signal['entity_id']}` | **Score:** `{signal['score']:.3f}`")
            else:
                st.caption("No metric anomalies in this window.")

        with col_right:
            st.markdown("#### 📜 Log Evidence (Event Sequences)")
            log_signals = [signal for signal in inc.get("signals", []) if signal["type"] == "log_anomaly"]
            if log_signals:
                for signal in log_signals:
                    st.markdown(f"**Trace ID:** `{signal.get('trace_id', 'N/A')}` (Score: `{signal['score']:.3f}`)")
                    st.caption("Detailed event sequence available in the context package.")
            else:
                st.caption("No log anomalies in this window.")

    with tab_ai:
        if not exp:
            st.warning("AI Explanation not available for this incident.")
        else:
            st.markdown("#### 📝 Incident Summary")
            st.info(exp.get("incident_summary", "No summary generated."))

            st.markdown("#### 🔍 Observed Evidence")
            for item in exp.get("observed_evidence", []):
                st.markdown(f"- {item}")

            st.markdown("#### 💡 Probable Cause")
            st.success(exp.get("probable_cause", "No probable cause identified."))

            st.markdown("#### ⚠️ Confidence & Uncertainty")
            st.warning(exp.get("confidence_uncertainty", "No uncertainty assessment provided."))

    with tab_remedy:
        if not rem:
            st.warning("Remediation advice not available for this incident.")
        else:
            safe_actions = rem.get("safe_recommendations", [])
            blocked_actions = rem.get("blocked_recommendations", [])

            st.markdown("#### ✅ Recommended Investigation Steps (Safe & Non-Destructive)")
            if safe_actions:
                grouped = {}
                for action in safe_actions:
                    category = action.get("category", "general")
                    grouped.setdefault(category, []).append(action["action"])

                for category, actions in grouped.items():
                    st.markdown(f"**{category.replace('_', ' ').title()}:**")
                    for action in actions:
                        st.markdown(f"1. {action}")
            else:
                st.info("No specific safe actions recommended. Please review the AI explanation.")

            if blocked_actions:
                st.divider()
                st.markdown("#### 🚫 Blocked Unsafe Actions (Prevented by Aether)")
                for blocked in blocked_actions:
                    st.error(f"~~{blocked['action']}~~ *(Reason: {blocked['blocked_reason']})*")


if __name__ == "__main__":
    main()
