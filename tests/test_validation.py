import numpy as np
import pandas as pd
import pytest

from src.common.validation import ValidationError, validate_metrics


def valid_metrics():
	return pd.DataFrame({
		"timestamp": pd.to_datetime([
			"2024-01-01", "2024-01-02", "2024-01-03",
		]),
		"entity_id": ["machine-1"] * 3,
		"m00": [1.0, 2.0, 3.0],
		"label": [0, 0, 1],
		"split": ["train", "train", "test"],
	})


def test_valid_metrics_summary():
	report = validate_metrics(valid_metrics())
	assert report["rows"] == 3
	assert report["entities"] == 1
	assert report["n_metrics"] == 1


def test_rejects_nan_metric():
	df = valid_metrics()
	df.loc[1, "m00"] = np.nan
	with pytest.raises(ValidationError, match="NaN values"):
		validate_metrics(df)


def test_rejects_infinite_metric():
	df = valid_metrics()
	df.loc[1, "m00"] = np.inf
	with pytest.raises(ValidationError, match="Infinite values"):
		validate_metrics(df)


def test_rejects_duplicate_entity_timestamp():
	df = valid_metrics()
	df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
	with pytest.raises(ValidationError, match="Duplicate"):
		validate_metrics(df)


def test_rejects_train_test_time_leakage():
	df = valid_metrics()
	df.loc[0, "split"] = "test"
	df.loc[2, "split"] = "test"
	with pytest.raises(ValidationError, match="time leakage"):
		validate_metrics(df)
