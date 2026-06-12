"""Processed panel schema validation."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_panel

from nemforecastdemand.data.loaders import load_panel, load_splits, validate_panel


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


def _write_processed(tmp_path, full, labels):
    full.to_parquet(tmp_path / "panel.parquet")
    pd.DataFrame({"split": pd.Series(labels, index=full.index)}).to_parquet(
        tmp_path / "split_labels.parquet"
    )


def test_load_panel_round_trip(tmp_path):
    full = make_panel("2024-09-01", days=10)
    _write_processed(tmp_path, full, ["train"] * len(full))
    pd.testing.assert_frame_equal(load_panel(tmp_path), full, check_freq=False)


def test_load_splits_slices_panel_by_label(tmp_path):
    # Train first half, then two disjoint evaluation windows (non-contiguous).
    full = make_panel("2024-09-01", days=10)
    labels = np.array(["none"] * len(full), dtype=object)
    labels[: 5 * 48] = "train"
    labels[6 * 48 : 7 * 48] = "validation"
    labels[8 * 48 : 9 * 48] = "test"
    _write_processed(tmp_path, full, labels)
    splits = load_splits(tmp_path)
    assert len(splits["train"]) == 5 * 48
    assert len(splits["validation"]) == 48
    assert splits["train"].index[-1] < splits["validation"].index[0]


def test_load_splits_rejects_train_overlapping_eval(tmp_path):
    full = make_panel("2024-09-01", days=10)
    labels = np.array(["none"] * len(full), dtype=object)
    labels[: 8 * 48] = "train"  # train runs past the validation window
    labels[6 * 48 : 7 * 48] = "validation"
    labels[9 * 48 :] = "test"
    _write_processed(tmp_path, full, labels)
    with pytest.raises(ValueError, match="strictly precede"):
        load_splits(tmp_path)


def test_validate_panel_allows_gaps_without_grid(tmp_path):
    full = make_panel("2024-09-01", days=10)
    block = full.iloc[list(range(48)) + list(range(96, 144))]  # a deliberate gap
    validate_panel(block, "block", require_grid=False)
    with pytest.raises(ValueError, match="grid"):
        validate_panel(block, "block", require_grid=True)
