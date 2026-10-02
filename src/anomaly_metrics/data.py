import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.common.validation import metric_columns


def load_split_frames(processed_dir, machines=None):
    pdir = Path(processed_dir)
    filters = [("entity_id", "in", list(machines))] if machines else None
    train = pd.read_parquet(pdir / "train.parquet", filters=filters)
    test = pd.read_parquet(pdir / "test.parquet", filters=filters)
    if train.empty or test.empty:
        raise ValueError(f"No rows found for machines={machines} in {pdir}")
    return train, test


def to_arrays(df: pd.DataFrame):
    cols = metric_columns(df)
    return df[cols].to_numpy(dtype="float32"), cols


def chronological_split(x: np.ndarray, val_fraction: float):
    n_val = int(len(x) * val_fraction)
    if n_val < 1:
        raise ValueError("val_fraction too small for this series")
    return x[:-n_val], x[-n_val:]


def windows_view(x: np.ndarray, window: int) -> np.ndarray:
    """(T, F) -> (T-window+1, window, F) as a memory-free view."""
    if x.ndim != 2:
        raise ValueError("expected a 2-D array (time, features)")
    if len(x) < window:
        raise ValueError(f"series of length {len(x)} is shorter than window {window}")
    return sliding_window_view(x, (window, x.shape[1]))[:, 0]


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)  # constant columns
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict):
        s = cls()
        s.mean, s.std = np.array(d["mean"], dtype="float32"), np.array(d["std"], dtype="float32")
        return s

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()))
