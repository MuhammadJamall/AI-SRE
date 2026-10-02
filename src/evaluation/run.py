import argparse
from pathlib import Path

from src.common.config import load_config, resolve
from src.common.logging_utils import get_logger, setup_logging
from src.context_builder.pipeline import run_context_builder
from src.correlation.pipeline import run_correlation
from src.evaluation.profiler import profile_module
from src.evaluation.report import generate_final_report
from src.llm.pipeline import run_llm_explainer
from src.remediation.pipeline import run_remediation_advisor

log = get_logger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aether Module 9: Final Evaluation & Integration")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-pipeline", action="store_true", help="Only evaluate existing outputs without rerunning Modules 4-7.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    processed_dir = resolve(cfg["data"]["processed_dir"])
    eval_cfg = cfg.get("evaluation", {})
    results_dir = resolve(eval_cfg.get("results_dir", "data/processed/evaluation"))

    if not args.skip_pipeline:
        log.info("Running End-to-End Pipeline (Modules 4 to 7)...")

        with profile_module("Module_4_Correlation"):
            run_correlation(
                processed_dir,
                float(cfg["correlation"]["max_gap_minutes"]),
                resolve(cfg["correlation"]["results_dir"]),
            )

        with profile_module("Module_5_Context_Builder"):
            run_context_builder(
                processed_dir,
                resolve(cfg["context_builder"]["results_dir"]),
                max_events=int(cfg["context_builder"].get("max_events_in_sequence", 50)),
            )

        with profile_module("Module_6_LLM_Explainer"):
            run_llm_explainer(
                resolve(cfg["context_builder"]["results_dir"]),
                resolve(cfg["llm"]["results_dir"]),
                cfg["llm"],
            )

        with profile_module("Module_7_Remediation"):
            run_remediation_advisor(
                resolve(cfg["llm"]["results_dir"]),
                resolve(cfg["remediation"]["results_dir"]),
                cfg["remediation"],
            )
    else:
        log.info("Skipping pipeline execution. Evaluating existing outputs...")
        profile_module.stats = {"Pipeline_Skipped": {"duration_seconds": 0, "peak_memory_mb": 0}}

    with profile_module("Module_9_Evaluation_Report"):
        report = generate_final_report(
            processed_dir,
            results_dir,
            max_hallucinations=int(eval_cfg.get("max_hallucinated_entities", 0)),
        )

    print("\n" + "=" * 50)
    print("AETHER FINAL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Integration Status: {report['integration_status']}")
    llm_agg = report.get("llm_evaluation", {}).get("aggregate", {})
    print(f"LLM Structural Failures: {llm_agg.get('structural_failures', 'N/A')}")
    print(f"LLM Hallucinated Entities: {llm_agg.get('total_hallucinated_entities', 'N/A')}")
    print(f"LLM Grounding Pass: {llm_agg.get('pass_threshold_met', 'N/A')}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
