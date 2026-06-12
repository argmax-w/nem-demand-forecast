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
from nemforecastdemand.data.loaders import load_panel, load_splits
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
    if not (cfg.paths.processed / "panel.parquet").exists():
        pytest.skip("committed splits not present")
    return load_splits(cfg.paths.processed)


def test_training_block_precedes_every_evaluation_point(committed):
    # The core no-leakage guarantee: no training timestamp is at or after
    # any validation or test timestamp, so no fitting target can sit behind
    # an evaluation point and no lag feature of a training row can read one.
    train = committed["train"].index
    for name in ("validation", "test"):
        assert train[-1] < committed[name].index[0]
    assert len(committed["validation"].index.intersection(committed["test"].index)) == 0


def test_validation_and_test_are_both_all_season(committed):
    # Both evaluation sets must span every month of the evaluation year, so
    # selection on validation faces the same seasonal mix as the test set.
    months = {}
    for name in ("validation", "test"):
        market = committed[name].index.tz_convert(MARKET_TZ)
        months[name] = set(pd.PeriodIndex(market, freq="M"))
    assert months["validation"] == months["test"]
    assert len(months["test"]) == 12
    # The holiday effect is learned on the two-year training block, which
    # sees every public holiday; whether a five-day evaluation window lands
    # on one is incidental and not required.
    assert committed["train"]["is_holiday"].any()


def test_validation_and_test_have_no_position_in_month_bias(committed):
    # Each set takes the early window in half its months and the late
    # window in the other half, so neither is biased to the start or end of
    # the month.
    for name in ("validation", "test"):
        market = committed[name].index.tz_convert(MARKET_TZ)
        month = pd.PeriodIndex(market, freq="M").astype(str)
        frame = pd.DataFrame({"ym": month, "day": market.day})
        first_day = frame.groupby("ym")["day"].min()
        early = int((first_day <= 12).sum())
        late = int((first_day >= 19).sum())
        assert early == late == 6


def test_training_demand_brackets_evaluation(committed):
    # Two years of training bracket what evaluation asks, so the test set
    # probes interpolation rather than extrapolation in the target.
    train_demand = committed["train"]["demand_mw"]
    for name in ("validation", "test"):
        demand = committed[name]["demand_mw"]
        inside = demand.between(train_demand.min(), train_demand.max()).mean()
        assert inside > 0.99


def test_perturbation_calibration_uses_only_genuine_forecast_rows(cfg, committed):
    # The early training period predates the forecast archive and carries
    # actuals in the forecast columns; the perturbation calibration must not
    # treat those as zero-error forecasts.
    from nemforecastdemand.models.predict import fit_perturbation_models

    panel = load_panel(cfg.paths.processed)
    models = fit_perturbation_models(panel, committed["train"].index)
    # A genuine day-ahead forecast has non-trivial error, so the fitted
    # per-step scales must be well above zero.
    for model in models.values():
        assert float(np.mean(model.sigma_by_step)) > 0.1


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
