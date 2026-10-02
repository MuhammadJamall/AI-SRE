import json

import pandas as pd
import pytest

from src.common.validation import validate_metrics
from src.ingestion.smd import ingest_smd


def test_smd_ingest_outputs(smd_root, tmp_path):
    out = tmp_path / "out"
    ingest_smd(smd_root, out)
    train, test = pd.read_parquet(out / "train.parquet"), pd.read_parquet(out / "test.parquet")
    assert len(train) == 400 and len(test) == 401          # lengths NOT assumed equal
    assert set(train["split"]) == {"train"} and set(test["split"]) == {"test"}
    assert train["label"].sum() == 0                       # train treated as normal
    assert test["label"].sum() == 20
    assert [c for c in train.columns if c.startswith("m")] == [f"m{i:02d}" for i in range(5)]
    validate_metrics(pd.concat([train, test], ignore_index=True))


def test_machine_boundaries_and_chronology(smd_root, tmp_path):
    out = tmp_path / "out"
    ingest_smd(smd_root, out)
    train, test = pd.read_parquet(out / "train.parquet"), pd.read_parquet(out / "test.parquet")
    for m in ["machine-1-1", "machine-1-2"]:
        assert train.loc[train.entity_id == m, "timestamp"].max() < test.loc[test.entity_id == m, "timestamp"].min()


def test_unequal_lengths_reported(smd_root, tmp_path):
    out = tmp_path / "out"
    ingest_smd(smd_root, out)
    rep = json.loads((out / "report.json").read_text())
    assert rep["machines_with_unequal_lengths"] == ["machine-1-2"]
    assert rep["lengths"]["machine-1-2"] == {"train": 200, "test": 201}


def test_interpretation_converted_to_zero_based(smd_root, tmp_path):
    out = tmp_path / "out"
    ingest_smd(smd_root, out)
    interp = json.loads((out / "interpretation.json").read_text())
    assert interp["machine-1-1"][0] == {"start_index": 50, "end_index": 59,
                                        "affected_metrics": ["m00", "m02"]}


def test_reproducible(smd_root, tmp_path):
    a = ingest_smd(smd_root, tmp_path / "a")
    b = ingest_smd(smd_root, tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_smd(tmp_path / "nope", tmp_path / "out")


def test_missing_label_file(smd_root, tmp_path):
    (smd_root / "test_label" / "machine-1-2.txt").unlink()
    with pytest.raises(FileNotFoundError):
        ingest_smd(smd_root, tmp_path / "out")


def test_non_numeric_rejected(smd_root, tmp_path):
    (smd_root / "train" / "machine-1-1.txt").write_text("a,b,c,d,e\n" * 5)
    with pytest.raises(ValueError):
        ingest_smd(smd_root, tmp_path / "out")


def test_label_length_mismatch_rejected(smd_root, tmp_path):
    (smd_root / "test_label" / "machine-1-1.txt").write_text("0\n" * 10)
    with pytest.raises(ValueError):
        ingest_smd(smd_root, tmp_path / "out")