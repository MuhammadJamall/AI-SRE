import argparse

from src.common.config import load_config, resolve
from src.common.logging_utils import setup_logging
from src.correlation.pipeline import run_correlation


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 4: Incident Correlation")
    p.add_argument("--config", default=None)
    p.add_argument("--log-model", default="lstm", choices=["ngram", "lstm", "gru"])
    p.add_argument("--max-gap", type=float, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    corr_cfg = cfg.get("correlation", {})
    max_gap = args.max_gap if args.max_gap is not None else corr_cfg.get("max_gap_minutes", 5.0)

    processed_dir = resolve(cfg["data"]["processed_dir"])
    results_dir = resolve(corr_cfg.get("results_dir", "data/processed/correlation"))

    incidents = run_correlation(processed_dir, max_gap, results_dir, log_model=args.log_model)
    print(f"Correlation complete. {len(incidents)} incidents generated in {results_dir}")


if __name__ == "__main__":
    main()
