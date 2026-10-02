"""Usage:
python -m src.anomaly_logs.run                       # models from config (ngram + lstm)
python -m src.anomaly_logs.run --models ngram        # baseline only (seconds)
python -m src.anomaly_logs.run --models ngram lstm gru --epochs 10
"""
import argparse
import json

from src.anomaly_logs.data import load_traces, load_vocab
from src.anomaly_logs.pipeline import run_model
from src.anomaly_metrics.train import get_device
from src.common.config import load_config, resolve
from src.common.io import save_json, save_parquet
from src.common.logging_utils import setup_logging


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 3: log anomaly detection")
    p.add_argument("--config", default=None)
    p.add_argument("--models", nargs="+", default=None, choices=["ngram", "lstm", "gru"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    lc = dict(cfg["logs_model"])
    if args.epochs:
        lc["epochs"] = args.epochs
    device = get_device(args.device or lc.get("device", "cpu"))
    seed = cfg["project"]["seed"]

    pdir = resolve(cfg["data"]["processed_dir"]) / "hdfs"
    traces, vocab = load_traces(pdir), load_vocab(pdir)
    model_root, results_root = resolve(lc["model_dir"]), resolve(lc["results_dir"])

    summary = {}
    for name in args.models or lc["models"]:
        m, preds = run_model(name, traces, vocab, lc, device, seed,
                             model_dir=model_root / name, results_dir=results_root / name,
                             plots=not args.no_plots)
        save_parquet(preds, results_root / f"predictions_{name}.parquet")
        summary[name] = {k: m[k] for k in ("precision", "recall", "f1", "roc_auc", "average_precision",
                                           "tp", "fp", "fn", "tn", "threshold")}
        summary[name]["oracle_best_f1"] = (m["oracle_best_f1"] or {}).get("f1")
        if "deeplog_rule" in m:
            summary[name]["deeplog_rule_f1"] = m["deeplog_rule"]["f1"]
    save_json(summary, results_root / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
