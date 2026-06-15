"""Data gates: poisoned inputs fail hard, nonsense outputs are withheld."""

import numpy as np
import pytest
from tests.conftest import make_panel

from nemforecastdemand import gates


def test_clean_panel_passes_the_input_gate():
    gates.validate_inputs(make_panel("2025-06-01", days=14))


def test_nan_in_inputs_fails_hard():
    panel = make_panel("2025-06-01", days=14)
    panel.iloc[5, panel.columns.get_loc("temp_c")] = np.nan
    with pytest.raises(gates.GateError, match="NaN"):
        gates.validate_inputs(panel)


def test_unphysical_value_fails_hard():
    panel = make_panel("2025-06-01", days=14)
    panel.iloc[10, panel.columns.get_loc("demand_mw")] = -50.0
    with pytest.raises(gates.GateError, match="outside"):
        gates.validate_inputs(panel)


def test_constant_feed_fails_hard():
    panel = make_panel("2025-06-01", days=14)
    panel["temp_c"] = 21.0
    with pytest.raises(gates.GateError, match="constant"):
        gates.validate_inputs(panel)


def test_broken_grid_fails_hard():
    panel = make_panel("2025-06-01", days=14)
    gates.validate_inputs(panel)
    dropped = panel.drop(panel.index[7])  # leaves a one-step hole
    with pytest.raises(gates.GateError, match="grid"):
        gates.validate_inputs(dropped)


def test_sound_forecast_passes_the_output_gate():
    rng = np.random.default_rng(0)
    paths = 8000 + rng.normal(0, 300, (200, 48))
    assert gates.validate_forecast(samples=paths) == []
    gates.check_forecast(samples=paths)


def test_negligible_negative_tail_in_samples_is_tolerated():
    rng = np.random.default_rng(0)
    paths = 8000 + rng.normal(0, 300, (1000, 48))
    paths[0, :] = -2000.0  # one draw in a thousand below zero: a tail, not nonsense
    assert gates.validate_forecast(samples=paths) == []


def test_sample_median_out_of_bounds_is_withheld():
    rng = np.random.default_rng(0)
    paths = 8000 + rng.normal(0, 300, (1000, 48))
    paths[:, 0] = -100.0  # one point's whole distribution is negative
    assert gates.validate_forecast(samples=paths) != []
    with pytest.raises(gates.GateError, match="median"):
        gates.check_forecast(samples=paths)


def test_excess_mass_out_of_bounds_in_samples_is_withheld():
    rng = np.random.default_rng(0)
    paths = 8000 + rng.normal(0, 300, (1000, 48))
    paths[:20, :] = -50.0  # 2% of the mass below zero, but the medians stay sound
    assert gates.validate_forecast(samples=paths) != []
    with pytest.raises(gates.GateError, match="mass"):
        gates.check_forecast(samples=paths)


def test_negative_and_nonfinite_forecasts_are_withheld():
    assert gates.validate_forecast(mean=np.array([7000.0, -10.0])) != []
    assert gates.validate_forecast(mean=np.array([np.nan, 7000.0])) != []
    assert gates.validate_forecast(mean=np.array([7000.0]), sd=np.array([-1.0])) != []


def test_crossing_quantiles_are_withheld():
    levels = np.array([0.1, 0.5, 0.9])[:, None]
    ok = levels * 0 + np.array([6000.0, 7000.0, 8000.0])[:, None]
    assert gates.validate_forecast(quantiles=ok) == []
    crossed = ok[::-1]  # high level holds the low value
    assert gates.validate_forecast(quantiles=crossed) != []
    with pytest.raises(gates.GateError):
        gates.check_forecast(quantiles=crossed)
