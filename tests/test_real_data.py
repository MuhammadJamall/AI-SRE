import json
import os

import pandas as pd
import pytest

from src.common.config import load_config, resolve
from src.ingestion.hdfs import ingest_hdfs
from src.ingestion.smd import ingest_smd

CFG = load_config()
SMD = resolve(CFG["datasets"]["smd"]["raw_path"])
HDFS = resolve(CFG["datasets"]["hdfs"]["raw_path"])


@pytest.mark.real_data
@pytest.mark.skipif(not SMD.exists(), reason="SMD not found")
def test_real_smd(tmp_path):
    df = ingest_smd(SMD, tmp_path)
    assert df["entity_id"].nunique() == 28
    assert len([c for c in df.columns if c.startswith("m")]) == 38
    rep = json.loads((tmp_path / "report.json").read_text())
    unequal = rep["machines_with_unequal_lengths"]
    assert len(unequal) == 15
    assert all(
        rep["lengths"][m]["test"] == rep["lengths"][m]["train"] + 1
        for m in unequal
    )
    test = pd.read_parquet(tmp_path / "test.parquet")
    assert 0 < test["label"].mean() < 0.2


@pytest.mark.real_data
@pytest.mark.skipif(not HDFS.exists(), reason="HDFS not found")
def test_real_hdfs_sample(tmp_path):
    traces = ingest_hdfs(HDFS, tmp_path, max_lines=500_000)
    assert len(traces) > 1000
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["n_official_templates"] == 29
    assert rep["unknown_event_fraction"] < 0.01


@pytest.mark.real_data
@pytest.mark.skipif(
    not (HDFS.exists() and os.environ.get("AETHER_FULL")),
    reason="set AETHER_FULL=1 for the full run",
)
def test_real_hdfs_full(tmp_path):
    traces = ingest_hdfs(HDFS, tmp_path)
    assert len(traces) == 575_061
    assert int(traces["label"].sum()) == 16_838


from src.anomaly_logs.data import load_traces, load_vocab
from src.anomaly_metrics.data import load_split_frames
from src.anomaly_metrics.pipeline import run_machine
from src.anomaly_metrics.train import get_device

PROCESSED_SMD = resolve(CFG["data"]["processed_dir"]) / "smd"


@pytest.mark.real_data
@pytest.mark.skipif(not (PROCESSED_SMD / "train.parquet").exists(), reason="run Module 1 first")
def test_real_metric_model_smoke(tmp_path):
    mc = dict(CFG["metrics_model"], epochs=2)
    train, test = load_split_frames(PROCESSED_SMD, ["machine-1-1"])
    metrics, preds = run_machine("machine-1-1", train, test, mc, get_device("cpu"), 42,
                                 results_dir=tmp_path, plots=False)
    assert len(preds) == len(test)
    assert metrics["final_train_loss"] > 0 and 0 <= metrics["f1"] <= 1


PROCESSED_HDFS = resolve(CFG["data"]["processed_dir"]) / "hdfs"


@pytest.mark.real_data
@pytest.mark.skipif(not (PROCESSED_HDFS / "traces.parquet").exists(), reason="run Module 1 first")
def test_real_log_models_smoke(tmp_path):
    from src.anomaly_logs.pipeline import run_model

    lc = dict(CFG["logs_model"], epochs=1, samples_per_epoch=20_000)
    traces, vocab = load_traces(PROCESSED_HDFS), load_vocab(PROCESSED_HDFS)
    m, preds = run_model("ngram", traces, vocab, lc, get_device("cpu"), 42, plots=False)
    assert len(preds) == 172_630 and m["roc_auc"] > 0.7
    m2, _ = run_model("lstm", traces, vocab, lc, get_device("cpu"), 42, plots=False)
    assert 0 <= m2["f1"] <= 1