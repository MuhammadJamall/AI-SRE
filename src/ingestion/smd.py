"""SMD adapter -> canonical metrics tables (train.parquet / test.parquet).

Facts about the raw data (from the dataset docs):
- files are headerless numeric matrices, no real timestamps -> deterministic synthetic
  1-minute timestamps; test continues right after train (chronological, per machine)
- labels exist for test only; train rows are labelled 0 (treated as normal for training)
- train and test lengths can differ (some machines have one extra test row)
- interpretation labels use 1-based indexes -> converted to 0-based
"""
from pathlib import Path

import pandas as pd

from src.common.io import save_json, save_parquet
from src.common.logging_utils import get_logger
from src.common.validation import clean_metrics, validate_metrics

log = get_logger(__name__)

BASE_TIME = pd.Timestamp("2020-01-01 00:00:00")
REQUIRED_DIRS = ("train", "test", "test_label")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMD file: {path}")
    return path


def _read_matrix(path: Path):
    # dtype float64 -> raises ValueError on non-numeric content
    return pd.read_csv(_require_file(path), header=None, dtype="float64").to_numpy()


def parse_interpretation(path: Path, metric_names: list) -> list:
    """Lines look like '15849-16368:1,9,10,12'. Row range and dimensions are 1-based."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rng, dims = line.split(":")
        start, end = (int(x) for x in rng.split("-"))
        out.append({
            "start_index": start - 1,
            "end_index": end - 1,
            "affected_metrics": [metric_names[int(d) - 1] for d in dims.split(",")],
        })
    return out


def load_machine(root, machine_id: str):
    """Returns (dataframe, interpretation or None, (n_train, n_test))."""
    root = Path(root)
    train = _read_matrix(root / "train" / f"{machine_id}.txt")
    test = _read_matrix(root / "test" / f"{machine_id}.txt")
    labels = pd.read_csv(_require_file(root / "test_label" / f"{machine_id}.txt"),
                         header=None)[0].to_numpy(dtype="int8")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{machine_id}: train/test feature count mismatch")
    if len(labels) != len(test):
        raise ValueError(f"{machine_id}: {len(labels)} labels for {len(test)} test rows")

    cols = [f"m{i:02d}" for i in range(train.shape[1])]

    def build(values, start, label, split):
        df = pd.DataFrame(values, columns=cols)
        df.insert(0, "entity_id", machine_id)
        df.insert(0, "timestamp", pd.date_range(start, periods=len(values), freq="min"))
        df["label"] = label
        df["split"] = split
        return df

    train_df = build(train, BASE_TIME, 0, "train")
    test_df = build(test, BASE_TIME + pd.Timedelta(minutes=len(train)), labels, "test")

    interp = None
    ipath = root / "interpretation_label" / f"{machine_id}.txt"
    if ipath.is_file():
        interp = parse_interpretation(ipath, cols)
    return pd.concat([train_df, test_df], ignore_index=True), interp, (len(train), len(test))


def ingest_smd(raw_root, out_dir, machines=None) -> pd.DataFrame:
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    for sub in REQUIRED_DIRS:
        if not (raw_root / sub).is_dir():
            raise FileNotFoundError(f"Missing SMD directory: {raw_root / sub}")
    available = sorted(p.stem for p in (raw_root / "train").glob("machine-*.txt"))
    if not available:
        raise FileNotFoundError(f"No machine-*.txt files under {raw_root / 'train'}")
    machines = machines or available

    frames, interpretation, lengths = [], {}, {}
    for m in machines:
        df, interp, (n_tr, n_te) = load_machine(raw_root, m)
        frames.append(df)
        lengths[m] = {"train": n_tr, "test": n_te}
        if interp is not None:
            interpretation[m] = interp
        log.info("loaded %s (train=%d, test=%d)", m, n_tr, n_te)

    df, clean_report = clean_metrics(pd.concat(frames, ignore_index=True))
    report = {"dataset": "smd", **clean_report, **validate_metrics(df),
              "machines_with_unequal_lengths": [m for m, l in lengths.items() if l["train"] != l["test"]],
              "lengths": lengths}

    save_parquet(df[df["split"] == "train"].reset_index(drop=True), out_dir / "train.parquet")
    save_parquet(df[df["split"] == "test"].reset_index(drop=True), out_dir / "test.parquet")
    save_json(interpretation, out_dir / "interpretation.json")
    save_json(report, out_dir / "report.json")
    return df