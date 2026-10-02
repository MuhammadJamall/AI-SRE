"""Shared input/output helpers for ingestion adapters."""

import json
from pathlib import Path

import pandas as pd


def load_json(path) -> object:
	path = Path(path)
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def save_json(data, path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_parquet(df: pd.DataFrame, path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(path, index=False)
