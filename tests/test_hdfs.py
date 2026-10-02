import json
import shutil

import pandas as pd
import pytest

from src.common.validation import ValidationError
from src.ingestion.hdfs import (
    LINE_RE, TemplateMatcher, UNKNOWN_EVENT, ingest_hdfs, load_templates,
    mask_content, parse_timestamp,
)


def test_parse_timestamp():
    assert str(parse_timestamp("081109", "203615")) == "2008-11-09 20:36:15"


def test_line_regex_accepts_both_layouts():
    a = LINE_RE.match("081109 203518 143 INFO dfs.DataNode$X: Receiving block blk_1 src: a")
    b = LINE_RE.match("081109 203518 081109 203518 INFO dfs.DataNode$X: Receiving block blk_1 src: a")
    assert a and b
    assert a.group(1) == b.group(1) == "081109"
    assert a.group(2) == b.group(2) == "203518"
    assert a.group(6) == b.group(6)


def test_mask_content_is_id_invariant():
    a = mask_content("Receiving block blk_1 src: /10.0.0.1:5 dest: /10.0.0.2:6")
    b = mask_content("Receiving block blk_99 src: /1.2.3.4:7 dest: /5.6.7.8:9")
    assert a == b


def test_template_matcher(hdfs_root):
    tm = TemplateMatcher(load_templates(hdfs_root / "preprocessed" / "HDFS_templates.csv"))
    assert tm.match("PacketResponder 7 for block blk_5 terminating") == "E3"
    assert tm.match("something never seen before") == UNKNOWN_EVENT


def test_template_matcher_accepts_loghub_wildcards():
    tm = TemplateMatcher({"E11": "[*]PacketResponder[*]for block[*]terminating[*]"})
    assert tm.match("PacketResponder 7 for block blk_5 terminating") == "E11"


def test_templates_bad_columns(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValidationError):
        load_templates(p)


def test_hdfs_ingest(hdfs_root, tmp_path):
    out = tmp_path / "out"
    traces = ingest_hdfs(hdfs_root, out)
    rep = json.loads((out / "report.json").read_text())
    assert rep["skipped_lines"] == 1
    assert rep["unknown_event_lines"] == 0
    assert len(traces) == 3

    seq = {r.trace_id: list(r.event_sequence) for r in traces.itertuples()}
    assert seq["blk_1"] == seq["blk_2"] == ["E1", "E2", "E3"]
    assert seq["blk_3"] == ["E1", "E4"]
    assert dict(zip(traces.trace_id, traces.label)) == {"blk_1": 0, "blk_2": 0, "blk_3": 1}

    logs = pd.read_parquet(out / "logs.parquet")
    assert len(logs) == 8
    assert set(logs["event_id"]) <= {"E1", "E2", "E3", "E4"}
    assert logs.groupby("trace_id")["split"].nunique().max() == 1      # no leakage


def test_flat_layout_and_loghub_template_name(hdfs_root, tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    shutil.copy(hdfs_root / "HDFS.log", flat)
    shutil.copy(hdfs_root / "preprocessed" / "anomaly_label.csv", flat)
    shutil.copy(
        hdfs_root / "preprocessed" / "HDFS_templates.csv",
        flat / "HDFS.log_templates.csv",
    )

    traces = ingest_hdfs(flat, tmp_path / "out")

    assert len(traces) == 3


def test_streaming_in_small_batches(hdfs_root, tmp_path):
    ingest_hdfs(hdfs_root, tmp_path / "a", batch_size=2)               # forces multiple batches
    ingest_hdfs(hdfs_root, tmp_path / "b", batch_size=1000)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / "logs.parquet"),
                                  pd.read_parquet(tmp_path / "b" / "logs.parquet"))


def test_deterministic_split(hdfs_root, tmp_path):
    a = ingest_hdfs(hdfs_root, tmp_path / "a")
    b = ingest_hdfs(hdfs_root, tmp_path / "b")
    assert list(a["split"]) == list(b["split"])


def test_missing_files(hdfs_root, tmp_path):
    (hdfs_root / "HDFS.log").unlink()
    with pytest.raises(FileNotFoundError):
        ingest_hdfs(hdfs_root, tmp_path / "out")


def test_missing_labels(hdfs_root, tmp_path):
    (hdfs_root / "preprocessed" / "anomaly_label.csv").unlink()
    with pytest.raises(FileNotFoundError):
        ingest_hdfs(hdfs_root, tmp_path / "out")