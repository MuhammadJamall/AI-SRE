"""HDFS_v1 adapter -> logs.parquet + traces.parquet (streaming, never loads HDFS.log fully).

- event_id comes from the official HDFS_templates.csv (matched with anchored regexes);
  lines matching no template get event_id 'UNKNOWN' and are counted in the report
- trace_id = Block ID; a log row is attached to the first block on the line, while trace
  sequences include every labelled block on the line
- split = deterministic hash of trace_id, so a trace never appears in both splits
"""
import re
import zlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.io import save_json, save_parquet
from src.common.logging_utils import get_logger
from src.common.validation import ValidationError, validate_logs, validate_traces

log = get_logger(__name__)

UNKNOWN_EVENT = "UNKNOWN"
LINE_RE = re.compile(r"^(\d{6}) (\d{6}) (\d{6} \d{6}|\d+) (\w+) ([\w.$]+): (.*)$")
def find_file(root: Path, names) -> Path:
    """Find a file in the flat layout or in a 'preprocessed/' subfolder."""
    for base in (root, root / "preprocessed"):
        for name in names:
            if (base / name).is_file():
                return base / name
    raise FileNotFoundError(f"None of {list(names)} found under {root} (or {root / 'preprocessed'})")
BLK_RE = re.compile(r"blk_-?\d+")

# Used ONLY to build a cache key so we don't run all 29 regexes on every line.
_CACHE_MASKS = [
    (re.compile(r"blk_-?\d+"), "<BLK>"),
    (re.compile(r"/?\d+\.\d+\.\d+\.\d+(?::\d+)?"), "<IP>"),
    (re.compile(r"(?<![\w<])/[\w.\-$]+(?:/[\w.\-$]+)*"), "<PATH>"),
    (re.compile(r"(?<![\w<])-?\d+(?![\w>])"), "<NUM>"),
]

LOG_COLUMNS = ["timestamp", "log_message", "event_id", "trace_id", "label", "split"]
LOG_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns")),
    ("log_message", pa.string()),
    ("event_id", pa.string()),
    ("trace_id", pa.string()),
    ("label", pa.int8()),
    ("split", pa.string()),
])


def _require_file(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing HDFS file: {path}")
    return Path(path)


def parse_timestamp(date: str, time: str) -> datetime:
    """HDFS stamps look like '081109' '203615' (yyMMdd HHmmss)."""
    return datetime(2000 + int(date[0:2]), int(date[2:4]), int(date[4:6]),
                    int(time[0:2]), int(time[2:4]), int(time[4:6]))


def mask_content(content: str) -> str:
    for pattern, token in _CACHE_MASKS:
        content = pattern.sub(token, content)
    return content


def template_to_regex(template: str):
    parts = [re.escape(p) for p in re.split(r"<\*>|\[\*\]", template)]
    return re.compile("^" + ".*?".join(parts) + "$")


class TemplateMatcher:
    """Maps a log message to an official event id."""

    def __init__(self, templates: dict, max_cache: int = 200_000):
        # most specific (most literal characters) first
        ordered = sorted(templates.items(), key=lambda kv: -len(kv[1].replace("<*>", "")))
        self._compiled = [(eid, template_to_regex(t)) for eid, t in ordered]
        self._cache = {}
        self._max_cache = max_cache

    def match(self, content: str) -> str:
        key = mask_content(content)
        eid = self._cache.get(key)
        if eid is None:
            eid = UNKNOWN_EVENT
            for candidate, rx in self._compiled:
                if rx.match(content):
                    eid = candidate
                    break
            if len(self._cache) >= self._max_cache:
                self._cache.clear()
            self._cache[key] = eid
        return eid


def load_templates(path: Path) -> dict:
    df = pd.read_csv(_require_file(path))
    missing = {"EventId", "EventTemplate"} - set(df.columns)
    if missing:
        raise ValidationError(f"{path} lacks expected columns {sorted(missing)}; found {list(df.columns)}")
    return dict(zip(df["EventId"].astype(str), df["EventTemplate"].astype(str)))


def load_labels(path: Path) -> dict:
    df = pd.read_csv(_require_file(path))
    if not {"BlockId", "Label"} <= set(df.columns):
        raise ValidationError(f"{path} must have BlockId and Label columns; found {list(df.columns)}")
    return {b: int(str(l).strip().lower() == "anomaly") for b, l in zip(df["BlockId"], df["Label"])}


def split_for(trace_id: str, train_ratio: float) -> str:
    return "train" if zlib.crc32(trace_id.encode()) % 1000 < int(train_ratio * 1000) else "test"


def ingest_hdfs(raw_root, out_dir, max_lines=None, batch_size=200_000, train_ratio=0.7):
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    log_path = find_file(raw_root, ["HDFS.log"])
    labels = load_labels(find_file(raw_root, ["anomaly_label.csv"]))
    templates = load_templates(find_file(
        raw_root, ["HDFS.log_templates.csv", "HDFS_templates.csv"]
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    matcher = TemplateMatcher(templates)
    traces = {}  # trace_id -> [start_time, [event ids]]
    batch, state = [], {"writer": None, "rows": 0}
    skipped = unlabeled = unknown = 0

    def flush():
        if not batch:
            return
        df = pd.DataFrame(batch, columns=LOG_COLUMNS)
        validate_logs(df)
        table = pa.Table.from_pandas(df, schema=LOG_SCHEMA, preserve_index=False)
        if state["writer"] is None:
            state["writer"] = pq.ParquetWriter(out_dir / "logs.parquet", LOG_SCHEMA)
        state["writer"].write_table(table)
        state["rows"] += len(df)
        batch.clear()
        log.info("wrote %d log rows so far", state["rows"])

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                m = LINE_RE.match(line.rstrip("\n"))
                if not m:
                    skipped += 1
                    continue
                date, time, _pid, _level, _comp, content = m.groups()
                blocks = list(dict.fromkeys(BLK_RE.findall(content)))
                if not blocks:
                    skipped += 1
                    continue

                ts = parse_timestamp(date, time)
                event_id = matcher.match(content)
                if event_id == UNKNOWN_EVENT:
                    unknown += 1
                for b in blocks:
                    if b in labels:
                        traces.setdefault(b, [ts, []])[1].append(event_id)

                primary = blocks[0]
                if primary not in labels:
                    unlabeled += 1
                    continue
                batch.append((ts, content, event_id, primary, labels[primary],
                              split_for(primary, train_ratio)))
                if len(batch) >= batch_size:
                    flush()
        flush()
    finally:
        if state["writer"] is not None:
            state["writer"].close()

    if state["rows"] == 0:
        raise ValidationError("No log rows produced; check paths/labels")

    trace_df = pd.DataFrame(
        [(tid, start, seq, labels[tid], split_for(tid, train_ratio))
         for tid, (start, seq) in traces.items()],
        columns=["trace_id", "start_time", "event_sequence", "label", "split"],
    ).sort_values("start_time", kind="stable").reset_index(drop=True)

    parsed_lines = state["rows"] + unlabeled
    report = {
        "dataset": "hdfs_v1",
        "log_rows": state["rows"],
        "skipped_lines": skipped,
        "lines_with_unlabeled_block": unlabeled,
        "n_official_templates": len(templates),
        "unknown_event_lines": unknown,
        "unknown_event_fraction": unknown / parsed_lines if parsed_lines else 0.0,
        **validate_traces(trace_df),
    }
    save_parquet(trace_df, out_dir / "traces.parquet")
    save_json(templates, out_dir / "templates.json")
    save_json(report, out_dir / "report.json")
    return trace_df