import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.anomaly_metrics.data import StandardScaler, chronological_split, to_arrays
from src.anomaly_metrics.evaluate import evaluate
from src.anomaly_metrics.plots import plot_example_window, plot_score_distribution, plot_timeline
from src.anomaly_metrics.train import reconstruction_errors, train_model
from src.common.io import save_json
from src.common.logging_utils import get_logger

log = get_logger(__name__)
TOP_K_AFFECTED = 5


def _affected_metrics(errors, thresholds, columns):
    ratios = errors / thresholds
    order = np.argsort(ratios)[::-1]
    return [columns[j] for j in order[:TOP_K_AFFECTED] if ratios[j] > 1.0]


def run_machine(machine, train_df, test_df, cfg, device, seed=42,
                model_dir=None, results_dir=None, plots=True):
    """Train on one machine's train rows, score its test rows. Returns (metrics, predictions_df)."""
    window = cfg["window"]
    train_df = train_df.sort_values("timestamp")
    test_df = test_df.sort_values("timestamp")
    x_train, cols = to_arrays(train_df)
    x_test, _ = to_arrays(test_df)
    y_test = test_df["label"].to_numpy(dtype="int64")

    # scaler fits on the fit portion only; threshold comes from the (normal) validation tail
    x_fit, x_val = chronological_split(x_train, cfg["val_fraction"])
    scaler = StandardScaler().fit(x_fit)
    fit_s, val_s, test_s = (scaler.transform(a) for a in (x_fit, x_val, x_test))

    model, history = train_model(fit_s, val_s, cfg, device, seed)

    val_err = reconstruction_errors(model, val_s, window, device)
    pct = cfg["threshold_percentile"]
    threshold = max(float(np.percentile(val_err.mean(axis=1), pct)), 1e-12)
    feat_thr = np.maximum(np.percentile(val_err, pct, axis=0), 1e-12)

    test_err = reconstruction_errors(model, test_s, window, device)
    raw = test_err.mean(axis=1)
    score = raw / (raw + threshold)
    is_anom = raw > threshold

    affected = [[] for _ in range(len(raw))]
    for i in np.where(is_anom)[0]:
        affected[i] = _affected_metrics(test_err[i], feat_thr, cols)

    preds = pd.DataFrame({
        "entity_id": machine,
        "timestamp": test_df["timestamp"].to_numpy(),
        "anomaly_score": score.astype("float64"),
        "raw_error": raw.astype("float64"),
        "is_anomaly": is_anom,
        "label": y_test,
        "affected_metrics": affected,
    })

    metrics = evaluate(y_test, is_anom.astype(int), raw)
    metrics.update({
        "entity_id": machine, "threshold": threshold, "threshold_percentile": pct,
        "epochs_run": len(history["train_loss"]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "history": history, "n_test": int(len(y_test)), "n_test_anomalies": int(y_test.sum()),
        "n_flagged": int(is_anom.sum()),
    })

    if model_dir is not None:
        mdir = Path(model_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": dict(cfg), "n_features": len(cols)}, mdir / "model.pt")
        scaler.save(mdir / "scaler.json")
        save_json({"threshold": threshold, "percentile": pct, "feature_thresholds": feat_thr.tolist(),
                   "window": window, "metric_columns": cols}, mdir / "threshold.json")
    if results_dir is not None:
        rdir = Path(results_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        save_json(metrics, rdir / "metrics.json")
        if plots:
            plot_score_distribution(val_err.mean(axis=1), raw, y_test, threshold, rdir / "score_distribution.png")
            plot_timeline(raw, y_test, is_anom.astype(int), threshold, rdir / "timeline.png")
            plot_example_window(test_s, test_err, y_test, cols, rdir / "example_anomaly.png")
    log.info("%s: P=%.3f R=%.3f F1=%.3f | PA-F1=%.3f", machine,
             metrics["precision"], metrics["recall"], metrics["f1"], metrics["point_adjusted"]["f1"])
    return metrics, preds
