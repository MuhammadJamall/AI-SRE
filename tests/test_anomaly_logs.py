import numpy as np
import pandas as pd
import pytest
import torch

from src.anomaly_logs.baseline import NGramModel
from src.anomaly_logs.data import (
    END, START, UNK, build_vocab, encode, is_val_trace, make_batches, pad_batch,
    split_traces, unique_sequences, vocab_size,
)
from src.anomaly_logs.evaluate import average_precision, roc_auc
from src.anomaly_logs.model import NextEventModel
from src.anomaly_logs.pipeline import run_model
from src.anomaly_logs.scoring import score_sequences
from src.anomaly_logs.train import train_model

VOCAB = build_vocab([f"E{i}" for i in range(1, 11)])
NORMALS = [("E1", "E2", "E3", "E4"), ("E1", "E2", "E5", "E4"), ("E1", "E2", "E3", "E5", "E4")]
ANOMS = [("E1", "E4", "E3", "E2"), ("E1", "E2", "E3"), ("E1", "E2", "E9", "E4"), ("E4", "E4", "E4", "E4")]
CFG = dict(val_fraction=0.1, ngram_order=3, ngram_smoothing=0.01, embedding_size=8, hidden_size=16,
           num_layers=1, batch_size=64, samples_per_epoch=3000, epochs=12, lr=0.01, patience=5,
           topg=3, threshold_percentile=99.5)
DEV = torch.device("cpu")


def make_traces(n_train=600, n_test=300, test_override=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_train):
        anom = i < 5                                   # a few train anomalies that must be ignored
        seq = ANOMS[i % 4] if anom else NORMALS[rng.integers(3)]
        rows.append((f"tr{i}", seq, int(anom), "train"))
    for i in range(n_test):
        anom = i % 10 == 0
        seq = ANOMS[rng.integers(4)] if anom else NORMALS[rng.integers(3)]
        if test_override is not None:
            seq = test_override
        rows.append((f"te{i}", seq, int(anom), "test"))
    df = pd.DataFrame(rows, columns=["trace_id", "event_sequence", "label", "split"])
    df.insert(1, "start_time", pd.Timestamp("2008-11-09") + pd.to_timedelta(np.arange(len(df)), unit="s"))
    return df


def test_vocab_encode_and_unknown():
    assert VOCAB["E1"] == 4 and VOCAB["E10"] == 13           # numeric order, not lexicographic
    assert vocab_size(VOCAB) == 14
    enc = encode(("E1", "ZZZ"), VOCAB)
    assert enc.tolist() == [START, VOCAB["E1"], UNK, END]


def test_unique_sequences_roundtrip():
    seqs = [("a",), ("b",), ("a",), ("a",)]
    uniq, counts, inv = unique_sequences(seqs)
    assert uniq == [("a",), ("b",)] and counts.tolist() == [3, 1]
    assert [uniq[i] for i in inv] == seqs


def test_pad_batch_and_batches():
    encs = [encode(("E1", "E2"), VOCAB), encode(("E1",), VOCAB)]
    inp, tgt, mask = pad_batch(encs)
    assert inp.shape == (2, 3) and mask.tolist() == [[True, True, True], [True, True, False]]
    assert tgt[0, -1] == END
    seen = sorted(i for b in make_batches(range(2), encs, 1, np.random.default_rng(0)) for i in b)
    assert seen == [0, 1]


def test_split_uses_only_normal_train_traces_and_is_disjoint():
    parts = split_traces(make_traces(), 0.1)
    assert (parts["fit"]["label"] == 0).all() and (parts["val"]["label"] == 0).all()
    assert set(parts["fit"]["split"]) == {"train"} == set(parts["val"]["split"])
    assert set(parts["test"]["split"]) == {"test"}
    ids = [set(parts[k]["trace_id"]) for k in ("fit", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert parts["n_train_anomalies_ignored"] == 5
    assert is_val_trace("tr1", 0.1) == is_val_trace("tr1", 0.1)


def test_metric_functions():
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(y, s) == pytest.approx(0.75)
    assert average_precision(y, s) == pytest.approx(0.8333333, rel=1e-5)
    assert roc_auc([0, 0], [1, 2]) is None


def test_ngram_scores_anomalies_higher():
    encs = [encode(s, VOCAB) for s in NORMALS]
    m = NGramModel(3, vocab_size(VOCAB), 0.01).fit(encs, [100, 100, 100])
    normal = m.score(encs)["max_nll"]
    anom = m.score([encode(s, VOCAB) for s in ANOMS])["max_nll"]
    assert anom.min() > 3 * normal.max()


def test_neural_model_learns_and_flags_anomalies():
    uniq = list(NORMALS)
    enc = [encode(s, VOCAB) for s in uniq]
    counts = np.array([100, 100, 50])
    model, hist = train_model(enc, counts, enc, counts, vocab_size(VOCAB), dict(CFG, rnn_type="lstm"), DEV, seed=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    n = score_sequences(model, enc, DEV, topg=3)
    a = score_sequences(model, [encode(s, VOCAB) for s in ANOMS], DEV, topg=3)
    assert a["max_nll"].min() > n["max_nll"].max()
    assert a["topg_miss"].mean() > n["topg_miss"].mean()


def test_model_rejects_bad_rnn_type():
    with pytest.raises(ValueError):
        NextEventModel(14, rnn_type="transformer")


@pytest.mark.parametrize("name", ["ngram", "lstm", "gru"])
def test_run_model_end_to_end(name, tmp_path):
    metrics, preds = run_model(name, make_traces(), VOCAB, CFG, DEV, seed=1,
                               model_dir=tmp_path / "m", results_dir=tmp_path / "r")
    assert metrics["roc_auc"] > 0.9 and metrics["recall"] > 0.5
    assert {"trace_id", "start_time", "anomaly_score", "is_anomaly", "label",
            "most_surprising_event", "position"} <= set(preds.columns)
    assert preds["anomaly_score"].between(0, 1).all()
    assert (preds["is_anomaly"] == (preds["anomaly_score"] > 0.5)).all()
    assert len(preds) == 300
    assert (tmp_path / "r" / "metrics.json").exists() and (tmp_path / "m" / "threshold.json").exists()


def test_threshold_independent_of_test_data():
    a, _ = run_model("ngram", make_traces(), VOCAB, CFG, DEV, plots=False)
    b, _ = run_model("ngram", make_traces(test_override=("E4", "E4")), VOCAB, CFG, DEV, plots=False)
    assert a["threshold"] == pytest.approx(b["threshold"])


def test_ngram_reproducible():
    _, a = run_model("ngram", make_traces(), VOCAB, CFG, DEV, plots=False)
    _, b = run_model("ngram", make_traces(), VOCAB, CFG, DEV, plots=False)
    pd.testing.assert_frame_equal(a, b)


def test_neural_reproducible():
    _, a = run_model("lstm", make_traces(), VOCAB, CFG, DEV, seed=5, plots=False)
    _, b = run_model("lstm", make_traces(), VOCAB, CFG, DEV, seed=5, plots=False)
    np.testing.assert_allclose(a["raw_score"], b["raw_score"], rtol=1e-4, atol=1e-6)
