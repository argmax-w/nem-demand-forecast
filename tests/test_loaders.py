"""Processed panel schema validation."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_panel

from nemforecastdemand.data.loaders import load_split, load_splits, validate_panel


def test_valid_panel_passes(panel):
    validate_panel(panel)


def test_missing_column_rejected(panel):
    with pytest.raises(ValueError, match="missing columns"):
        validate_panel(panel.drop(columns="temp_c"))


def test_wrong_dtype_rejected(panel):
    bad = panel.assign(demand_mw=panel["demand_mw"].astype("float64"))
    with pytest.raises(ValueError, match="dtype"):
        validate_panel(bad)


def test_grid_gap_rejected(panel):
    with pytest.raises(ValueError, match="half-hourly grid"):
        validate_panel(panel.drop(panel.index[5]))


def test_nan_rejected(panel):
    bad = panel.copy()
    bad.iloc[3, bad.columns.get_loc("temp_c")] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_panel(bad)


def test_naive_index_rejected(panel):
    bad = panel.copy()
    bad.index = bad.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        validate_panel(bad)


def test_negative_demand_rejected(panel):
    bad = panel.copy()
    bad.iloc[0, bad.columns.get_loc("demand_mw")] = -1.0
    with pytest.raises(ValueError, match="positive"):
        validate_panel(bad)


def test_load_splits_round_trip(tmp_path):
    full = make_panel("2025-06-01", days=10)
    parts = {"train": full.iloc[:336], "validation": full.iloc[336:408], "test": full.iloc[408:]}
    for name, frame in parts.items():
        frame.to_parquet(tmp_path / f"{name}.parquet")
    splits = load_splits(tmp_path)
    assert len(splits["train"]) == 336
    pd.testing.assert_frame_equal(splits["test"], parts["test"], check_freq=False)


def test_load_splits_rejects_gap(tmp_path):
    full = make_panel("2025-06-01", days=10)
    parts = {"train": full.iloc[:336], "validation": full.iloc[340:408], "test": full.iloc[408:]}
    for name, frame in parts.items():
        frame.to_parquet(tmp_path / f"{name}.parquet")
    with pytest.raises(ValueError, match="contiguous"):
        load_splits(tmp_path)


def test_unknown_split_name(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        load_split("holdout", tmp_path)
