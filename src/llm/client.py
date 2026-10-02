import json
import re
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock that stays grounded in the incident evidence supplied in the prompt."""

    @staticmethod
    def _extract_context(prompt: str) -> dict:
        incident_match = re.search(r"Incident ID:\s*(.+?)\n", prompt)
        incident_id = incident_match.group(1).strip() if incident_match else "UNKNOWN"

        evidence_match = re.search(r"Structured Evidence:\s*(.*?)\n--- END OF EVIDENCE ---", prompt, re.S)
        evidence_text = evidence_match.group(1).strip() if evidence_match else "{}"
        evidence = json.loads(evidence_text) if evidence_text else {}
        return {"incident_id": incident_id, "evidence": evidence}

    def generate(self, prompt: str) -> str:
        context = self._extract_context(prompt)
        evidence = context.get("evidence", {})
        metric_items = evidence.get("metrics", [])
        log_items = evidence.get("logs", [])

        metric_entities = [item.get("entity_id") for item in metric_items if isinstance(item, dict) and item.get("entity_id")]
        trace_ids = [item.get("trace_id") for item in log_items if isinstance(item, dict) and item.get("trace_id")]

        machine_id = metric_entities[0] if metric_entities else None
        trace_id = trace_ids[0] if trace_ids else None

        event_ids = []
        for item in log_items:
            event_sequence = item.get("event_sequence", "")
            event_ids.extend(re.findall(r"E\d+", str(event_sequence)))

        if event_ids:
            observed_events = " -> ".join(event_ids[:5])
        else:
            observed_events = "the observed event sequence"

        incident_summary = (
            f"Incident {context.get('incident_id', 'UNKNOWN')} shows correlated anomaly signals in the available evidence."
            if machine_id is None and trace_id is None
            else (
                f"Incident {context.get('incident_id', 'UNKNOWN')} shows metric anomalies on {machine_id} "
                f"and correlated log anomalies in trace {trace_id}."
                if machine_id and trace_id
                else (
                    f"Incident {context.get('incident_id', 'UNKNOWN')} shows metric anomalies on {machine_id}."
                    if machine_id
                    else f"Incident {context.get('incident_id', 'UNKNOWN')} shows correlated log anomalies in trace {trace_id}."
                )
            )
        )

        observed_evidence = [
            (
                f"Metric reconstruction anomalies exceeded the normal baseline on {machine_id}."
                if machine_id
                else "Metric reconstruction anomalies exceeded the normal baseline in the observed window."
            ),
            (
                f"Event sequence {observed_events} deviated from the normal pattern in trace {trace_id}."
                if trace_id
                else "The observed event sequence deviated from the normal pattern for the corresponding log anomaly."
            ),
        ]

        probable_cause = (
            f"The evidence is most consistent with localized workload pressure or a transient execution issue affecting {machine_id} "
            f"and the related event sequence in {trace_id}."
            if machine_id and trace_id
            else (
                f"The evidence is most consistent with a localized workload pressure or transient execution issue affecting {machine_id}."
                if machine_id
                else "The evidence suggests a transient anomaly pattern in the relevant subsystem, but the available context is limited."
            )
        )

        recommended_investigation = [
            (
                f"Inspect workload and resource usage on {machine_id} around the incident window."
                if machine_id
                else "Inspect the affected subsystem and compare the relevant workload signals around the incident window."
            ),
            (
                f"Compare recent baseline metrics and related event traces for {trace_id} over the last 24 hours."
                if trace_id
                else "Compare recent baseline metrics and correlating event sequences over the last 24 hours."
            ),
            "Validate whether the issue appears in recurring operational patterns before any broader action.",
        ]

        explanation = {
            "incident_id": context.get("incident_id", "UNKNOWN"),
            "incident_summary": incident_summary,
            "observed_evidence": observed_evidence,
            "probable_cause": probable_cause,
            "confidence_uncertainty": "Medium confidence. The correlation is strong, and the observed facts are directly supported by the provided evidence; however, lower-level process data is still missing.",
            "recommended_investigation": recommended_investigation,
        }
        return json.dumps(explanation, indent=2)


class OllamaClient(LLMClient):
    """OpenAI-compatible client for local Ollama or similar inference servers."""

    def __init__(self, base_url: str, model_name: str, api_key: str = "", temperature: float = 0.1):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
            raise ImportError("Please install 'openai': pip install openai") from exc

        self.client = OpenAI(base_url=base_url, api_key=api_key or "ollama")
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
