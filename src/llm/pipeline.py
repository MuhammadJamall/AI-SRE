import json
from pathlib import Path

from src.common.io import save_json
from src.common.logging_utils import get_logger
from src.llm.client import MockLLMClient, OllamaClient
from src.llm.prompt import build_prompt

log = get_logger(__name__)


def run_llm_explainer(context_dir: Path, results_dir: Path, llm_config: dict):
    log.info("Loading context packages from Module 5...")
    context_path = Path(context_dir) / "context_packages.json"
    if not context_path.exists():
        raise FileNotFoundError(f"Context packages not found at {context_path}. Run Module 5 first.")

    with context_path.open("r", encoding="utf-8") as f:
        packages = json.load(f)

    if not packages:
        log.warning("No context packages to process.")
        return []

    provider = str(llm_config.get("provider", "mock")).lower()
    if provider == "mock":
        client = MockLLMClient()
        log.info("Using MockLLMClient (safe for testing).")
    elif provider in {"ollama", "openai"}:
        client = OllamaClient(
            base_url=llm_config.get("base_url", "http://localhost:11434/v1"),
            model_name=llm_config.get("model_name", "phi3"),
            api_key=llm_config.get("api_key", ""),
            temperature=llm_config.get("temperature", 0.1),
        )
        log.info(f"Using {provider.capitalize()} client with model: {llm_config.get('model_name', 'phi3')}")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    explanations = []
    for index, pkg in enumerate(packages, start=1):
        log.info(f"Generating explanation for {pkg['incident_id']} ({index}/{len(packages)})...")
        prompt = build_prompt(pkg)

        try:
            response_text = client.generate(prompt)
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError as exc:
            log.error(f"Failed to parse JSON for {pkg['incident_id']}: {exc}")
            parsed_response = {
                "incident_id": pkg["incident_id"],
                "error": "LLM failed to return valid JSON",
                "raw_response": response_text if "response_text" in locals() else "",
            }

        parsed_response.setdefault("incident_id", pkg["incident_id"])
        explanations.append(parsed_response)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(explanations, results_dir / "explanations.json")
    log.info(f"Saved {len(explanations)} explanations to {results_dir / 'explanations.json'}")

    return explanations
