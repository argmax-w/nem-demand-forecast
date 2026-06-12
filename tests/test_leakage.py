"""Leakage and representativeness audit on the committed splits and features.

The mechanical split properties (day-boundary cuts, qualifying origins) are
exercised on synthetic indices in test_splits.py. This module audits the
committed splits themselves and the two routes by which future information
could reach a model: feature rows computed from data after their own
timestamp, and statistics computed over data outside the fitting window.
"""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import MARKET_TZ, make_panel

from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.models.base import (
    build_design,
    recency_features,
    stacked_origin_design,
    variance_design,
)
from nemforecastdemand.models.bsts import prepare_inputs
from nemforecastdemand.splits import horizon_index

LOCAL_TZ = "Australia/Sydney"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def committed(cfg):
    if not (cfg.paths.processed / "train.parquet").exists():
        pytest.skip("committed splits not present")
    return load_splits(cfg.paths.processed)


def test_committed_splits_form_one_unbroken_chronology(committed):
    train, validation, test = (committed[name].index for name in ("train", "validation", "test"))
    assert train[-1] < validation[0] <= validation[-1] < test[0]
    union = train.union(validation).union(test)
    assert union.equals(pd.date_range(union[0], union[-1], freq="30min"))
    for index in (validation, test):
        market = index[0].tz_convert(MARKET_TZ)
        assert (market.hour, market.minute) == (0, 0)
    assert abs(len(train) / len(union) - 0.70) < 0.01
    assert abs(len(validation) / len(union) - 0.15) < 0.01


def test_committed_splits_are_representative(committed):
    # Every split is at least seven weeks, so each sees the daily and
    # weekly cycles many times over, and each contains public holidays.
    for frame in committed.values():
        assert len(frame) >= 49 * 48
        assert frame["is_holiday"].any()

    # The seasonal design runs on the Sydney clock, so daylight saving
    # must be exercised on both sides of the fit: train straddles the
    # October transition and test the April one.
    for name in ("train", "test"):
        index = committed[name].index[::48]
        offsets = {ts.tz_convert(LOCAL_TZ).utcoffset() for ts in index}
        assert len(offsets) == 2

    # Training demand brackets what evaluation asks of the models, so the
    # test set probes interpolation, not extrapolation in the target.
    train_demand = committed["train"]["demand_mw"]
    for name in ("validation", "test"):
        demand = committed[name]["demand_mw"]
        inside = demand.between(train_demand.min(), train_demand.max()).mean()
        assert inside > 0.99


def test_design_rows_are_invariant_to_future_data(cfg):
    # Every design column is a pointwise or backwards-looking transform:
    # deleting the future must not change a single historical row. This
    # catches centred windows, full-panel normalisation and similar
    # accidents in one sweep.
    full = make_panel("2025-06-01", days=28)
    truncated = full.iloc[: 20 * 48]
    for builder in (build_design, variance_design):
        whole = builder(full, cfg).loc[truncated.index]
        early = builder(truncated, cfg)
        pd.testing.assert_frame_equal(whole, early)


def test_horizon_lag_features_predate_the_origin(cfg):
    # The shortest demand lag must clear the horizon, otherwise designs
    # for late horizon steps would read demand after the forecast origin.
    assert min(cfg.features.demand_lags) >= cfg.horizon

    panel = make_panel("2025-06-01", days=28)
    origin = panel.index[20 * 48]
    steps = horizon_index(origin, cfg.horizon)
    design = build_design(panel, cfg).loc[steps]
    for lag in cfg.features.demand_lags:
        sources = steps - pd.Timedelta("30min") * lag
        assert (sources < origin).all()
        expected = panel["demand_mw"].reindex(sources).to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(design[f"demand_lag{lag}"].to_numpy(), expected)


def test_recency_features_anchor_strictly_before_the_origin(cfg):
    # The deviations are differences of observed demand at fixed offsets
    # behind the origin, constant across the block; the horizon step is
    # calendar metadata.
    panel = make_panel("2025-06-01", days=28)
    origin = panel.index[20 * 48]
    block = recency_features(panel, origin, cfg.horizon)
    demand = panel["demand_mw"]
    np.testing.assert_array_equal(block["horizon_step"], np.arange(cfg.horizon))
    assert block["dev_day"].nunique() == 1
    expected = float(demand.iloc[20 * 48 - 1] - demand.iloc[20 * 48 - 49])
    np.testing.assert_allclose(block["dev_day"].iloc[0], expected, rtol=1e-6)

    # Corrupting demand from the origin onwards must leave the whole
    # origin block untouched: recency features and every demand lag look
    # strictly backwards, so only the targets may change.
    corrupted = panel.copy()
    corrupted.loc[panel.index >= origin, "demand_mw"] += 10_000.0
    origins = pd.DatetimeIndex([origin])
    clean_x, clean_y = stacked_origin_design(panel, cfg, origins)
    dirty_x, dirty_y = stacked_origin_design(corrupted, cfg, origins)
    pd.testing.assert_frame_equal(clean_x, dirty_x)
    assert (dirty_y - clean_y).abs().max() > 1_000.0


def test_bsts_scalers_ignore_data_after_the_fit_window(cfg):
    # Standardisation statistics come from the fit window only: corrupting
    # everything after it must leave the prepared inputs untouched.
    panel = make_panel("2025-06-01", days=28)
    fit_index = panel.index[max(cfg.features.demand_lags) : 20 * 48]
    corrupted = panel.copy()
    later = panel.index[20 * 48 :]
    corrupted.loc[later, "demand_mw"] += 10_000.0
    corrupted.loc[later, "temp_c"] += 30.0

    clean = prepare_inputs(panel, cfg, fit_index)
    dirty = prepare_inputs(corrupted, cfg, fit_index)
    np.testing.assert_array_equal(clean.y, dirty.y)
    np.testing.assert_array_equal(clean.x_mean, dirty.x_mean)
    np.testing.assert_array_equal(clean.x_var, dirty.x_var)
