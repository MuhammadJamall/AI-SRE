"""Usage:
python -m src.anomaly_metrics.run                       # machines from config
python -m src.anomaly_metrics.run --machines machine-1-1
python -m src.anomaly_metrics.run --all --epochs 20     # all 28 machines
python -m src.anomaly_metrics.run --device mps
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.anomaly_metrics.data import load_split_frames
from src.anomaly_metrics.evaluate import confusion, prf
from src.anomaly_metrics.pipeline import run_machine
from src.anomaly_metrics.train import get_device
from src.common.config import load_config, resolve
from src.common.io import save_json, save_parquet
from src.common.logging_utils import setup_logging


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 2: metric anomaly detection")
    p.add_argument("--config", default=None)
    p.add_argument("--machines", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    mc = dict(cfg["metrics_model"])
    if args.epochs:
        mc["epochs"] = args.epochs
    device = get_device(args.device or mc.get("device", "cpu"))
    seed = cfg["project"]["seed"]

    machines = None if args.all else (args.machines or mc["machines"])
    train, test = load_split_frames(resolve(cfg["data"]["processed_dir"]) / "smd", machines)
    machines = sorted(train["entity_id"].unique())

    model_root, results_root = resolve(mc["model_dir"]), resolve(mc["results_dir"])
    all_metrics, all_preds = [], []
    for m in machines:
        metrics, preds = run_machine(
            m, train[train.entity_id == m], test[test.entity_id == m], mc, device, seed,
            model_dir=model_root / m, results_dir=results_root / m, plots=not args.no_plots)
        all_metrics.append(metrics)
        all_preds.append(preds)

    tot = {k: sum(m[k] for m in all_metrics) for k in ("tn", "fp", "fn", "tp")}
    summary = {
        "machines": machines,
        "micro": {**tot, **prf(tot)},
        "macro": {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in ("precision", "recall", "f1")},
        "macro_point_adjusted_f1": sum(m["point_adjusted"]["f1"] for m in all_metrics) / len(all_metrics),
        "per_machine": {m["entity_id"]: {k: m[k] for k in ("precision", "recall", "f1", "n_flagged", "n_test_anomalies", "epochs_run")}
                        | {"point_adjusted_f1": m["point_adjusted"]["f1"],
                           "oracle_best_f1": (m["oracle_best_f1"] or {}).get("f1")}
                        for m in all_metrics},
    }
    save_json(summary, results_root / "summary.json")
    save_parquet(pd.concat(all_preds, ignore_index=True), results_root / "predictions.parquet")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
