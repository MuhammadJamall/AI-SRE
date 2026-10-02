import argparse

from src.common.config import load_config, resolve
from src.common.logging_utils import setup_logging
from src.llm.pipeline import run_llm_explainer


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 6: LLM RCA Explainer")
    p.add_argument("--config", default=None)
    p.add_argument("--provider", default=None, choices=["mock", "ollama", "openai"])
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    llm_cfg = cfg.get("llm", {})
    if args.provider:
        llm_cfg["provider"] = args.provider

    context_dir = resolve(cfg.get("context_builder", {}).get("results_dir", "data/processed/context_builder"))
    results_dir = resolve(llm_cfg.get("results_dir", "data/processed/llm_explanations"))

    explanations = run_llm_explainer(context_dir, results_dir, llm_cfg)
    print(f"LLM explanation complete. {len(explanations)} explanations saved in {results_dir}")


if __name__ == "__main__":
    main()
