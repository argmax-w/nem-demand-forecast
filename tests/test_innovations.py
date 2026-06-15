"""Innovations-form AR(2): exact likelihood and exact variance decomposition.

The likelihood must match a hand-rolled normal computation exactly (it is a
reparameterisation, not an approximation), and the two-part decomposition
must reproduce the variance of the simulated predictive paths.
"""

import numpy as np
import pytest
from scipy import stats

from nemforecastdemand.config import BstsConfig, BstsPriors
from nemforecastdemand.models import innovations

HORIZON = 12
N_HIST = 200


@pytest.fixture(scope="module")
def cfg_bsts() -> BstsConfig:
    return BstsConfig(
        obs_family="gaussian",
        heteroskedastic=True,
        variance_daily_harmonics=1,
        variance_use_degree_days=False,
        priors=BstsPriors(
            coef_scale=1.0,
            obs_scale=0.5,
            var_intercept_loc=-1.5,
            var_intercept_scale=0.7,
            var_coef_scale=0.25,
            student_t_df_rate=0.5,
            ar_alpha=8.0,
            ar_beta=2.0,
        ),
    )


def make_draws(n_draws: int, n_coefs: int, n_var: int, spread: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    draws = {
        "phi1": np.clip(0.9 + spread * rng.normal(0, 0.03, n_draws), 0.5, 0.98),
        "phi2": np.clip(-0.3 + spread * rng.normal(0, 0.05, n_draws), -0.9, 0.9),
        "gamma0": -1.4 + spread * rng.normal(0, 0.1, n_draws),
        "beta": np.tile(rng.normal(0, 0.3, n_coefs), (n_draws, 1))
        + spread * rng.normal(0, 0.05, (n_draws, n_coefs)),
        "gamma": np.tile(rng.normal(0, 0.1, n_var), (n_draws, 1))
        + spread * rng.normal(0, 0.02, (n_draws, n_var)),
    }
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


def test_log_density_matches_manual_computation(cfg_bsts):
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    y, x, z = synthetic_history()
    params = {
        "phi1": np.float64(0.85),
        "phi2": np.float64(-0.3),
        "beta": np.array([0.2, -0.1, 0.3]),
        "gamma0": np.float64(-1.2),
        "gamma": np.array([0.05, -0.03]),
    }
    model_density, _ = log_density(
        innovations.innovations_model,
        (jnp.asarray(y), jnp.asarray(x), jnp.asarray(z), cfg_bsts),
        {},
        {name: jnp.asarray(value) for name, value in params.items()},
    )

    rho1 = params["phi1"] * (1 - params["phi2"])
    rho2 = params["phi2"]
    sigma = np.exp(np.clip(params["gamma0"] + z.astype(np.float64) @ params["gamma"], -8.0, 3.0))
    e = y.astype(np.float64) - x.astype(np.float64) @ params["beta"]
    var0 = sigma[0] ** 2 * (1 - rho2) / ((1 + rho2) * ((1 - rho2) ** 2 - rho1**2))
    sd0 = np.sqrt(var0)
    lag1 = rho1 / (1 - rho2)
    manual = stats.norm.logpdf(e[0], 0.0, sd0)
    manual += stats.norm.logpdf(e[1], lag1 * e[0], sd0 * np.sqrt(1 - lag1**2))
    manual += stats.norm.logpdf(e[2:], rho1 * e[1:-1] + rho2 * e[:-2], sigma[2:]).sum()
    manual += stats.beta.logpdf(params["phi1"], 8.0, 2.0)
    manual += -np.log(2.0)  # phi2 ~ Uniform(-1, 1)
    manual += stats.norm.logpdf(params["beta"], 0.0, cfg_bsts.priors.coef_scale).sum()
    manual += stats.norm.logpdf(
        params["gamma0"], cfg_bsts.priors.var_intercept_loc, cfg_bsts.priors.var_intercept_scale
    )
    manual += stats.norm.logpdf(params["gamma"], 0.0, cfg_bsts.priors.var_coef_scale).sum()
    np.testing.assert_allclose(float(model_density), manual, rtol=1e-5)


def test_identical_draws_match_closed_form(cfg_bsts):
    draws = make_draws(8, 3, 2, spread=0.0)
    y, x, _ = synthetic_history()
    x_future, z_future = horizon_designs(2)
    positions = np.array([150, 180])
    e_origin = innovations.origin_residuals(draws, y, x, positions)
    assert e_origin.shape == (8, 2, 2)  # (draws, origins, two lags)
    parts = innovations.decompose_horizon_variance(draws, e_origin, x_future, z_future, cfg_bsts)
    assert np.allclose(parts["parameter"], 0.0, atol=1e-9)

    # Independent numpy recursion for the first origin: the innovation
    # component must equal the AR(2) variance accumulated through the impulse
    # response psi.
    rho1 = float(draws["phi1"][0] * (1 - draws["phi2"][0]))
    rho2 = float(draws["phi2"][0])
    psi = [1.0, rho1]
    for _ in range(2, HORIZON):
        psi.append(rho1 * psi[-1] + rho2 * psi[-2])
    psi = np.array(psi)
    log_sigma = np.clip(
        draws["gamma0"][0] + z_future[0] @ draws["gamma"][0].astype(np.float64), -8.0, 3.0
    )
    variance = np.exp(2 * log_sigma)
    expected = [float((psi[: h + 1][::-1] ** 2 * variance[: h + 1]).sum()) for h in range(HORIZON)]
    # The decomposition accumulates in float32, so the tolerance against the
    # float64 recursion sits at the float32 noise floor for the summed series.
    np.testing.assert_allclose(parts["innovation"][0], expected, rtol=2e-3)


def test_components_sum_to_simulated_path_variance(cfg_bsts):
    draws = make_draws(4000, 3, 2, spread=1.0, seed=3)
    y, x, _ = synthetic_history()
    x_future, z_future = horizon_designs(2)
    positions = np.array([150, 180])
    e_origin = innovations.origin_residuals(draws, y, x, positions)
    parts = innovations.decompose_horizon_variance(draws, e_origin, x_future, z_future, cfg_bsts)
    total = parts["parameter"] + parts["innovation"]

    paths = innovations.simulate_horizon_paths(
        draws, e_origin, x_future, z_future, cfg_bsts, seed=7
    )
    simulated = paths.astype(np.float64).var(axis=0)
    np.testing.assert_allclose(total, simulated, rtol=0.1)
