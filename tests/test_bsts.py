"""The Kalman prediction machinery: variance decomposition correctness.

The decomposition must be exact, not approximate: with identical draws the
parameter component vanishes and the remaining components reproduce the
closed-form h-step predictive variance; with spread-out draws the four
components must sum to the variance of the simulated predictive paths up
to Monte Carlo error.
"""

import numpy as np
import pytest

from nemforecastdemand.config import BstsConfig, BstsPriors
from nemforecastdemand.models import bsts

HORIZON = 12
N_HIST = 200


@pytest.fixture(scope="module")
def cfg_bsts() -> BstsConfig:
    return BstsConfig(
        train_days=4,
        damped_slope=True,
        obs_family="gaussian",
        heteroskedastic=True,
        variance_daily_harmonics=1,
        variance_use_degree_days=False,
        priors=BstsPriors(
            level_scale=0.1,
            slope_scale=0.01,
            damping_alpha=8.0,
            damping_beta=2.0,
            coef_scale=1.0,
            obs_scale=0.5,
            var_intercept_loc=-1.5,
            var_intercept_scale=0.7,
            var_coef_scale=0.25,
            init_level_scale=1.0,
            init_slope_scale=0.1,
            student_t_df_rate=0.5,
        ),
    )


def make_draws(n_draws: int, n_coefs: int, n_var: int, spread: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = {
        "sigma_level": 0.06,
        "sigma_slope": 0.008,
        "phi": 0.85,
        "gamma0": -1.4,
        "level_init": 0.1,
        "slope_init": 0.0,
    }
    draws = {
        name: np.full(n_draws, value) + spread * rng.normal(0, abs(value) * 0.2 + 0.01, n_draws)
        for name, value in base.items()
    }
    draws["phi"] = np.clip(draws["phi"], 0.5, 0.99)
    draws["sigma_level"] = np.abs(draws["sigma_level"])
    draws["sigma_slope"] = np.abs(draws["sigma_slope"])
    draws["beta"] = np.tile(rng.normal(0, 0.3, n_coefs), (n_draws, 1))
    draws["beta"] += spread * rng.normal(0, 0.05, (n_draws, n_coefs))
    draws["gamma"] = np.tile(rng.normal(0, 0.1, n_var), (n_draws, 1))
    draws["gamma"] += spread * rng.normal(0, 0.02, (n_draws, n_var))
    return {name: value.astype(np.float32) for name, value in draws.items()}


def synthetic_history(seed: int = 1):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, N_HIST).astype(np.float32)
    x = rng.normal(0, 1, (N_HIST, 3)).astype(np.float32)
    z = rng.normal(0, 1, (N_HIST, 2)).astype(np.float32)
    return y, x, z


def horizon_designs(n_origins: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    x_future = rng.normal(0, 1, (n_origins, HORIZON, 3)).astype(np.float32)
    z_future = rng.normal(0, 1, (n_origins, HORIZON, 2)).astype(np.float32)
    return x_future, z_future


def test_identical_draws_have_no_parameter_variance(cfg_bsts):
    draws = make_draws(8, 3, 2, spread=0.0)
    y, x, z = synthetic_history()
    x_future, z_future = horizon_designs(2)
    filtered_mean, filtered_cov = bsts.kalman_filter_states(draws, y, x, z, cfg_bsts)
    positions = np.array([150, 180])
    parts = bsts.decompose_horizon_variance(
        draws, filtered_mean, filtered_cov, positions, x_future, z_future, cfg_bsts
    )
    assert np.allclose(parts["parameter"], 0.0, atol=1e-9)

    # Closed-form check against an independent numpy recursion for the
    # first origin: the components must reproduce the h-step predictive
    # variance exactly when there is a single effective draw.
    phi = float(draws["phi"][0])
    transition = np.array([[1.0, 1.0], [0.0, phi]])
    process = np.diag([float(draws["sigma_level"][0]) ** 2, float(draws["sigma_slope"][0]) ** 2])
    p_state = filtered_cov[0, positions[0] - 1].astype(np.float64)
    p_proc = np.zeros((2, 2))
    expected_state, expected_proc = [], []
    for _ in range(HORIZON):
        p_state = transition @ p_state @ transition.T
        p_proc = transition @ p_proc @ transition.T + process
        expected_state.append(p_state[0, 0])
        expected_proc.append(p_proc[0, 0])
    np.testing.assert_allclose(parts["state"][0], expected_state, rtol=1e-4)
    np.testing.assert_allclose(parts["process"][0], expected_proc, rtol=1e-4)
    log_sigma = np.clip(
        draws["gamma0"][0] + z_future[0] @ draws["gamma"][0].astype(np.float64), -8.0, 3.0
    )
    np.testing.assert_allclose(parts["observation"][0], np.exp(2 * log_sigma), rtol=1e-3)


def test_components_sum_to_simulated_path_variance(cfg_bsts):
    draws = make_draws(3000, 3, 2, spread=1.0, seed=3)
    y, x, z = synthetic_history()
    x_future, z_future = horizon_designs(2)
    filtered_mean, filtered_cov = bsts.kalman_filter_states(draws, y, x, z, cfg_bsts)
    positions = np.array([150, 180])
    parts = bsts.decompose_horizon_variance(
        draws, filtered_mean, filtered_cov, positions, x_future, z_future, cfg_bsts
    )
    total = parts["parameter"] + parts["state"] + parts["process"] + parts["observation"]

    paths = bsts.simulate_horizon_paths(
        draws, filtered_mean, filtered_cov, positions, x_future, z_future, cfg_bsts, seed=7
    )
    simulated = paths.astype(np.float64).var(axis=0)
    np.testing.assert_allclose(total, simulated, rtol=0.12)
