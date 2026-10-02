from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.anomaly_logs.baseline import NGramModel
from src.anomaly_logs.data import (
    N_SPECIAL, SPECIAL_NAMES, encode, split_traces, unique_sequences, vocab_size,
)
from src.anomaly_logs.evaluate import average_precision, roc_auc
from src.anomaly_logs.scoring import score_sequences
from src.anomaly_logs.train import train_model
from src.anomaly_metrics.evaluate import best_f1, confusion, prf
from src.anomaly_metrics.plots import plot_score_distribution
from src.common.io import save_json
from src.common.logging_utils import get_logger

log = get_logger(__name__)


def run_model(name, traces, vocab, cfg, device, seed=42, model_dir=None, results_dir=None, plots=True):
    """Fit on normal train traces, calibrate the threshold on normal validation traces,
    score the test split. Returns (metrics, predictions_df)."""
    parts = split_traces(traces, cfg["val_fraction"])
    for k in ("fit", "val", "test"):
        if parts[k].empty:
            raise ValueError(f"No '{k}' traces available")
    inv = {v: k for k, v in vocab.items()} | SPECIAL_NAMES
    vsize = vocab_size(vocab)

    def prep(df):
        uniq, counts, inverse = unique_sequences(df["event_sequence"].tolist())
        return [encode(s, vocab) for s in uniq], counts, inverse

    fit_enc, fit_counts, _ = prep(parts["fit"])
    val_enc, val_counts, val_inv = prep(parts["val"])
    test_enc, _, test_inv = prep(parts["test"])

    history, model = None, None
    if name == "ngram":
        model = NGramModel(cfg["ngram_order"], vsize, cfg["ngram_smoothing"]).fit(fit_enc, fit_counts)
        scorer = model.score
    elif name in ("lstm", "gru"):
        model, history = train_model(fit_enc, fit_counts, val_enc, val_counts, vsize,
                                     dict(cfg, rnn_type=name), device, seed)
        scorer = lambda enc: score_sequences(model, enc, device, cfg["topg"])  # noqa: E731
    else:
        raise ValueError(f"Unknown model '{name}' (use ngram, lstm or gru)")

    val_stats, test_stats = scorer(val_enc), scorer(test_enc)
    raw_val, raw_test = val_stats["max_nll"][val_inv], test_stats["max_nll"][test_inv]

    pct = cfg["threshold_percentile"]
    thr = max(float(np.percentile(raw_val, pct)), 1e-12)     # normal validation traces only
    y = parts["test"]["label"].to_numpy(dtype=int)
    pred = raw_test > thr

    surprising = np.array([int(e[p + 1]) for e, p in zip(test_enc, test_stats["max_pos"])])
    preds = pd.DataFrame({
        "trace_id": parts["test"]["trace_id"].to_numpy(),
        "start_time": parts["test"]["start_time"].to_numpy(),
        "anomaly_score": raw_test / (raw_test + thr),
        "raw_score": raw_test,
        "is_anomaly": pred,
        "label": y,
        "most_surprising_event": [inv[t] for t in surprising[test_inv]],
        "position": test_stats["max_pos"][test_inv],
    })

    c = confusion(y, pred)
    metrics = {
        "model": name, **c, **prf(c),
        "roc_auc": roc_auc(y, raw_test), "average_precision": average_precision(y, raw_test),
        "oracle_best_f1": best_f1(y, raw_test),
        "threshold": thr, "threshold_percentile": pct,
        "n_fit_traces": int(len(parts["fit"])), "n_val_traces": int(len(parts["val"])),
        "n_test_traces": int(len(y)), "n_test_anomalies": int(y.sum()),
        "n_unique_fit_sequences": len(fit_enc),
        "n_train_anomalies_ignored": parts["n_train_anomalies_ignored"],
        "history": history,
    }
    if "topg_miss" in test_stats:    # original DeepLog rule: any step outside the top-g predictions
        dl = test_stats["topg_miss"][test_inv] > 0
        cd = confusion(y, dl)
        metrics["deeplog_rule"] = {"topg": cfg["topg"], **cd, **prf(cd),
                                   "val_false_positive_rate": float((val_stats["topg_miss"][val_inv] > 0).mean())}

    if model_dir is not None:
        mdir = Path(model_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        if name == "ngram":
            save_json(model.to_dict(), mdir / "ngram.json")
        else:
            torch.save({"state_dict": model.state_dict(), "config": dict(cfg), "vocab": vocab}, mdir / "model.pt")
        save_json({"threshold": thr, "percentile": pct, "score": "max_nll"}, mdir / "threshold.json")
    if results_dir is not None:
        rdir = Path(results_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        save_json(metrics, rdir / "metrics.json")
        if plots:
            plot_score_distribution(raw_val, raw_test, y, thr, rdir / "score_distribution.png")
    log.info("%s: P=%.3f R=%.3f F1=%.3f AUC=%s AP=%s", name, metrics["precision"], metrics["recall"],
             metrics["f1"], metrics["roc_auc"], metrics["average_precision"])
    return metrics, preds
