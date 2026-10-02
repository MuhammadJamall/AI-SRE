import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

PAD, START, END, UNK = 0, 1, 2, 3
N_SPECIAL = 4
SPECIAL_NAMES = {PAD: "PAD", START: "START", END: "END", UNK: "UNK"}
TRACE_COLUMNS = ["trace_id", "start_time", "event_sequence", "label", "split"]


def load_traces(processed_dir) -> pd.DataFrame:
    path = Path(processed_dir) / "traces.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run Module 1 (HDFS ingestion) first")
    df = pd.read_parquet(path)
    missing = [c for c in TRACE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"traces.parquet is missing columns {missing}")
    df["event_sequence"] = df["event_sequence"].map(tuple)
    return df


def _event_key(e: str):
    m = re.fullmatch(r"E(\d+)", e)
    return (0, int(m.group(1))) if m else (1, e)


def build_vocab(event_ids) -> dict:
    ordered = sorted(set(event_ids), key=_event_key)
    return {e: N_SPECIAL + i for i, e in enumerate(ordered)}


def load_vocab(processed_dir) -> dict:
    """Vocabulary = the official templates written by Module 1 (not derived from the data)."""
    path = Path(processed_dir) / "templates.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run Module 1 first")
    return build_vocab(json.loads(path.read_text()).keys())


def vocab_size(vocab: dict) -> int:
    return N_SPECIAL + len(vocab)


def encode(seq, vocab: dict) -> np.ndarray:
    """[START, e1..en, END]; unseen event ids -> UNK."""
    return np.array([START] + [vocab.get(e, UNK) for e in seq] + [END], dtype=np.int64)


def unique_sequences(seqs):
    """Returns (unique_seqs, counts, inverse) so that seqs[i] == unique_seqs[inverse[i]]."""
    index, uniq, counts = {}, [], []
    inverse = np.empty(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        j = index.get(s)
        if j is None:
            j = len(uniq)
            index[s] = j
            uniq.append(s)
            counts.append(0)
        counts[j] += 1
        inverse[i] = j
    return uniq, np.asarray(counts, dtype=np.int64), inverse


def is_val_trace(trace_id, val_fraction: float) -> bool:
    return zlib.crc32((str(trace_id) + "|val").encode()) % 1000 < int(val_fraction * 1000)


def split_traces(df: pd.DataFrame, val_fraction: float) -> dict:
    """fit/val = NORMAL traces of the train split only; test = the test split.
    Train anomalies are never used (normal-only training)."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    normal = train[train["label"] == 0]
    mask = normal["trace_id"].map(lambda t: is_val_trace(t, val_fraction)).to_numpy()
    return {"fit": normal[~mask], "val": normal[mask], "test": test,
            "n_train_anomalies_ignored": int((train["label"] == 1).sum())}


def pad_batch(encoded):
    """Next-event batches: inputs = enc[:-1], targets = enc[1:], right-padded with PAD."""
    lengths = [len(e) - 1 for e in encoded]
    T = max(lengths)
    inp = np.zeros((len(encoded), T), dtype=np.int64)
    tgt = np.zeros((len(encoded), T), dtype=np.int64)
    mask = np.zeros((len(encoded), T), dtype=bool)
    for i, e in enumerate(encoded):
        n = len(e) - 1
        inp[i, :n], tgt[i, :n], mask[i, :n] = e[:-1], e[1:], True
    return inp, tgt, mask


def make_batches(indices, encoded, batch_size, rng):
    """Length-bucketed batches (little padding), batch order shuffled."""
    idx = np.asarray(indices)
    idx = idx[np.argsort([len(encoded[i]) for i in idx], kind="stable")]
    batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    for b in rng.permutation(len(batches)):
        yield batches[b]
