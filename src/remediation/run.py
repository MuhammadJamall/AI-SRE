import argparse

from src.common.config import load_config, resolve
from src.common.logging_utils import setup_logging
from src.remediation.pipeline import run_remediation_advisor


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 7: Remediation Advisor")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    rem_cfg = cfg.get("remediation", {})
    explanations_dir = resolve(cfg.get("llm", {}).get("results_dir", "data/processed/llm_explanations"))
    results_dir = resolve(rem_cfg.get("results_dir", "data/processed/remediation"))

    run_remediation_advisor(explanations_dir, results_dir, rem_cfg)
    print(f"Remediation advisory complete. Output saved to {results_dir}")


if __name__ == "__main__":
    main()
