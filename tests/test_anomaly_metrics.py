import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.anomaly_metrics.data import StandardScaler, chronological_split, windows_view
from src.anomaly_metrics.evaluate import confusion, evaluate, point_adjust, prf
from src.anomaly_metrics.model import LSTMAutoencoder
from src.anomaly_metrics.pipeline import _affected_metrics, run_machine
from src.anomaly_metrics.train import reconstruction_errors, train_model

CFG = dict(window=20, train_stride=2, val_fraction=0.2, hidden_size=16, latent_size=8,
           num_layers=1, batch_size=32, epochs=15, lr=0.005, patience=5, threshold_percentile=99.0)
DEV = torch.device("cpu")
F = 3


def make_frames(n_train=800, n_test=400, spike=True, test_scale=1.0):
    rng = np.random.default_rng(0)
    t = np.arange(n_train + n_test)
    base = np.stack([np.sin(t / 10 + k) for k in range(F)], 1) + 0.05 * rng.standard_normal((len(t), F))
    cols = [f"m{i:02d}" for i in range(F)]
    ts = pd.date_range("2020-01-01", periods=len(t), freq="min")

    def frame(sl, split):
        df = pd.DataFrame(base[sl], columns=cols)
        df.insert(0, "entity_id", "m")
        df.insert(0, "timestamp", ts[sl])
        df["label"], df["split"] = 0, split
        return df

    train = frame(slice(0, n_train), "train")
    test = frame(slice(n_train, None), "test").reset_index(drop=True)
    test[cols] = test[cols] * test_scale
    if spike:
        test.loc[200:219, cols] += 4.0
        test.loc[200:219, "label"] = 1
    return train, test


def test_scaler_roundtrip_and_constant_column():
    x = np.column_stack([np.arange(10.0), np.ones(10)]).astype("float32")
    s = StandardScaler().fit(x)
    z = s.transform(x)
    assert np.isfinite(z).all() and abs(z[:, 0].mean()) < 1e-6
    s2 = StandardScaler.from_dict(s.to_dict())
    np.testing.assert_allclose(s2.transform(x), z, rtol=1e-5)


def test_windows_and_split():
    x = np.arange(30, dtype="float32").reshape(10, 3)
    assert windows_view(x, 4).shape == (7, 4, 3)
    with pytest.raises(ValueError):
        windows_view(x, 11)
    a, b = chronological_split(x, 0.2)
    assert len(a) == 8 and len(b) == 2 and b[0, 0] == x[8, 0]


def test_model_output_shape():
    m = LSTMAutoencoder(F, 8, 4)
    assert m(torch.zeros(5, 20, F)).shape == (5, 20, F)


def test_metrics_math():
    yt = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    yp = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    c = confusion(yt, yp)
    assert c == {"tn": 4, "fp": 1, "fn": 3, "tp": 2}
    assert prf(c)["precision"] == pytest.approx(2 / 3)
    adj = point_adjust(yt, yp)
    assert adj.tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    assert evaluate(yt, yp)["point_adjusted"]["recall"] == 1.0


def test_affected_metrics_only_include_features_above_validation_threshold():
    actual = np.array([3.0, 0.5, 2.5, 1.0, 1.2, 4.0])
    thresholds = np.ones_like(actual)
    columns = [f"m{i:02d}" for i in range(len(actual))]
    assert _affected_metrics(actual, thresholds, columns) == ["m05", "m00", "m02", "m04"]
    assert _affected_metrics(np.array([1.0, 0.5]), np.ones(2), ["m00", "m01"]) == []


def test_training_loss_decreases_and_inference_shape():
    train, _ = make_frames()
    x = train[[f"m{i:02d}" for i in range(F)]].to_numpy("float32")
    fit, val = chronological_split(x, 0.2)
    sc = StandardScaler().fit(fit)
    model, hist = train_model(sc.transform(fit), sc.transform(val), CFG, DEV, seed=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    err = reconstruction_errors(model, sc.transform(val), CFG["window"], DEV)
    assert err.shape == val.shape and np.isfinite(err).all()


def test_run_machine_detects_injected_anomaly(tmp_path):
    train, test = make_frames()
    metrics, preds = run_machine("m", train, test, CFG, DEV, seed=1,
                                 model_dir=tmp_path / "model", results_dir=tmp_path / "res")
    assert {"entity_id", "timestamp", "anomaly_score", "raw_error", "is_anomaly", "affected_metrics"} <= set(preds.columns)
    assert preds["anomaly_score"].between(0, 1).all()
    assert (preds["is_anomaly"] == (preds["anomaly_score"] > 0.5)).all()
    spike = preds.iloc[200:220]
    normal = preds.drop(preds.index[200:220])
    assert spike["raw_error"].mean() > 3 * normal["raw_error"].mean()
    assert spike["is_anomaly"].any()
    assert all(len(a) > 0 for a in preds.loc[preds.is_anomaly, "affected_metrics"])
    for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        assert key in metrics
    for f in ("model.pt", "scaler.json", "threshold.json"):
        assert (tmp_path / "model" / f).exists()
    for f in ("metrics.json", "score_distribution.png", "timeline.png", "example_anomaly.png"):
        assert (tmp_path / "res" / f).exists()


def test_reproducible():
    train, test = make_frames()
    _, a = run_machine("m", train, test, copy.deepcopy(CFG), DEV, seed=7, plots=False)
    _, b = run_machine("m", train, test, copy.deepcopy(CFG), DEV, seed=7, plots=False)
    np.testing.assert_allclose(a["raw_error"].to_numpy(), b["raw_error"].to_numpy(), rtol=1e-4, atol=1e-6)


def test_threshold_does_not_depend_on_test_data():
    train, test1 = make_frames()
    _, test2 = make_frames(test_scale=3.0, spike=False)
    m1, _ = run_machine("m", train, test1, copy.deepcopy(CFG), DEV, seed=3, plots=False)
    m2, _ = run_machine("m", train, test2, copy.deepcopy(CFG), DEV, seed=3, plots=False)
    assert m1["threshold"] == pytest.approx(m2["threshold"], rel=1e-5)
