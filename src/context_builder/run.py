import argparse

from src.common.config import load_config, resolve
from src.common.logging_utils import setup_logging
from src.context_builder.pipeline import run_context_builder


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 5: Incident Context Builder")
    p.add_argument("--config", default=None)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    cb_cfg = cfg.get("context_builder", {})
    max_events = args.max_events if args.max_events is not None else cb_cfg.get("max_events_in_sequence", 50)

    processed_dir = resolve(cfg["data"]["processed_dir"])
    results_dir = resolve(cb_cfg.get("results_dir", "data/processed/context_builder"))

    packages = run_context_builder(processed_dir, results_dir, max_events=max_events)
    print(f"Context building complete. {len(packages)} packages saved in {results_dir}")


if __name__ == "__main__":
    main()
