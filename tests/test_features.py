"""Feature engineering checks: degree days, seasonal bases and calendar."""

import numpy as np
import pandas as pd

from nemforecastdemand.features.calendar import (
    fourier_design,
    holiday_flag,
    local_phases,
    periodic_rbf_design,
)
from nemforecastdemand.features.weather import PerturbationModel, degree_days, fit_perturbation


def test_degree_days_band_and_slopes():
    temp = pd.Series(
        [10.0, 16.5, 18.0, 20.5, 25.0],
        index=pd.date_range("2025-06-01", periods=5, freq="30min", tz="UTC"),
    )
    out = degree_days(temp, heating_base=16.5, cooling_base=20.5)
    np.testing.assert_allclose(out["heating_deg"], [6.5, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out["cooling_deg"], [0.0, 0.0, 0.0, 0.0, 4.5])


def test_fourier_design_shape_and_values():
    phase = np.array([0.0, 0.25, 0.5])
    design = fourier_design(phase, harmonics=2, prefix="d")
    assert design.shape == (3, 4)
    np.testing.assert_allclose(design["d_sin1"], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(design["d_cos2"], [1.0, -1.0, 1.0], atol=1e-12)


def test_rbf_design_is_periodic_and_local():
    phase = np.array([0.0, 1.0 - 1e-9, 0.5])
    design = periodic_rbf_design(phase, centres=8, prefix="d")
    assert design.shape == (3, 7)
    # The circle closes: phase 0 and phase 1 give the same basis row.
    np.testing.assert_allclose(design.iloc[0], design.iloc[1], atol=1e-6)
    # Each basis function peaks at its own centre.
    assert design.iloc[2]["d_rbf4"] == design.iloc[2].max()


def test_local_phases_follow_sydney_clock_through_dst():
    # 14:00 UTC is local midnight in winter (AEST), 13:00 UTC in summer (AEDT).
    winter = pd.DatetimeIndex([pd.Timestamp("2025-06-15 14:00", tz="UTC")])
    summer = pd.DatetimeIndex([pd.Timestamp("2026-01-15 13:00", tz="UTC")])
    for index in (winter, summer):
        daily, _ = local_phases(index)
        assert daily[0] == 0.0
    market_midnight_summer = pd.DatetimeIndex([pd.Timestamp("2026-01-15 14:00", tz="UTC")])
    daily, _ = local_phases(market_midnight_summer)
    assert daily[0] == 1.0 / 24.0


def test_holiday_flag_australia_day():
    # 2026-01-26 in Sydney spans 13:00 UTC on the 25th to 13:00 UTC on the 26th.
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-25 20:00", tz="UTC"),
            pd.Timestamp("2026-01-27 00:00", tz="UTC"),
        ]
    )
    flags = holiday_flag(index)
    assert flags.tolist() == [True, False]


def test_perturbation_zero_multiplier_returns_actuals():
    model = PerturbationModel(rho=0.7, sigma_by_step=np.full(48, 1.5))
    actual = np.linspace(10, 20, 48)
    steps = np.arange(48)
    rng = np.random.default_rng(1)
    np.testing.assert_array_equal(model.sample(actual, steps, 0.0, rng), actual)


def test_perturbation_fit_recovers_error_scale():
    rng = np.random.default_rng(2)
    index = pd.date_range("2025-06-01", periods=48 * 200, freq="30min", tz="UTC")
    rho, sigma = 0.8, 2.0
    shocks = rng.standard_normal(len(index))
    errors = np.empty(len(index))
    errors[0] = sigma * shocks[0]
    for i in range(1, len(index)):
        errors[i] = rho * errors[i - 1] + sigma * np.sqrt(1 - rho**2) * shocks[i]
    actual = pd.Series(20.0, index=index)
    forecast = actual + errors
    model = fit_perturbation(actual, forecast)
    assert abs(model.rho - rho) < 0.05
    np.testing.assert_allclose(model.sigma_by_step, sigma, rtol=0.2)


def test_perturbation_nonnegative_clips():
    model = PerturbationModel(rho=0.0, sigma_by_step=np.full(48, 500.0), nonnegative=True)
    actual = np.zeros(48)
    out = model.sample(actual, np.arange(48), 1.0, np.random.default_rng(3))
    assert (out >= 0).all()
