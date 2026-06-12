"""Innovations-form AR(1) regression: the BSTS revision without latent states.

The collapsed BSTS posterior shrank the level innovation to nothing and
pushed every short-memory fluctuation through a heavily damped slope, which
filtering loved and forecasting paid for: slope noise integrates into the
level, so 48-step paths accumulated process variance until the model scored
worse than the weekly naive. The ARIMA baseline points at the repair — its
selected order (1, 0, 1) carries the same short-run autocorrelation in a
stationary error instead of an integrated trend. This module is that
structure on the Bayesian side: the same regression and heteroskedastic
scale as the BSTS, with a stationary AR(1) error in place of the trend,

    e_t = y_t - x_t' beta,
    e_t - rho e_{t-1} ~ Normal(0, sigma_t),

and the first residual anchored at its stationary distribution. Writing the
likelihood on the innovations rather than on latent error states removes
the sequential dependence entirely: residuals come from one matrix product
and the innovations from a shifted difference, so there is no scan, no
Kalman pass and nothing the GPU dislikes. Because the error is first-order
Markov, prediction conditions only on the residual at the forecast origin
— observed exactly, no filtering uncertainty — and the h-step predictive
moments are closed-form in powers of rho.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from nemforecastdemand.config import BstsConfig

HYPER_SITES = ("rho", "beta", "gamma0", "gamma")


def innovations_model(
    y: jnp.ndarray,
    x_mean: jnp.ndarray,
    x_var: jnp.ndarray,
    bsts: BstsConfig,
) -> None:
    """Regression with stationary AR(1) errors, fully vectorised.

    Parameters
    ----------
    y
        Standardised observations, shape ``(T,)``.
    x_mean
        Standardised mean design, shape ``(T, K)``.
    x_var
        Standardised variance design, shape ``(T, J)``.
    bsts
        Structural settings; the heteroskedastic flag and prior scales are
        shared with the state-space formulations.
    """
    priors = bsts.priors
    n_coefs = x_mean.shape[1]

    rho = numpyro.sample("rho", dist.Beta(priors.ar_alpha, priors.ar_beta))
    beta = numpyro.sample("beta", dist.Normal(0.0, priors.coef_scale).expand([n_coefs]).to_event(1))
    if bsts.heteroskedastic:
        gamma0 = numpyro.sample(
            "gamma0", dist.Normal(priors.var_intercept_loc, priors.var_intercept_scale)
        )
        gamma = numpyro.sample(
            "gamma",
            dist.Normal(0.0, priors.var_coef_scale).expand([x_var.shape[1]]).to_event(1),
        )
        # The clip is a numerical guard for early optimisation steps, far
        # outside the region the tight priors allow at convergence.
        sigma = jnp.exp(jnp.clip(gamma0 + x_var @ gamma, -8.0, 3.0))
    else:
        sigma = numpyro.sample("gamma0", dist.HalfNormal(priors.obs_scale)) * jnp.ones(
            x_var.shape[0]
        )

    e = y - x_mean @ beta
    # Stationary anchor for the first residual; with a time-varying scale
    # this uses the scale at the first step, the exact form for the
    # homoskedastic model and a one-observation approximation otherwise.
    numpyro.sample("e_first", dist.Normal(0.0, sigma[0] / jnp.sqrt(1.0 - rho**2)), obs=e[0])
    numpyro.sample("innovations", dist.Normal(rho * e[:-1], sigma[1:]), obs=e[1:])


def _scales_and_regression(
    draws: dict[str, jnp.ndarray],
    x_future: jnp.ndarray,
    xv_future: jnp.ndarray,
    heteroskedastic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-draw horizon regression means and innovation scales, ``(S, O, H)``."""
    regression = jnp.einsum("ohk,sk->soh", x_future, draws["beta"])
    if heteroskedastic:
        log_sigma = draws["gamma0"][:, None, None] + jnp.einsum(
            "ohj,sj->soh", xv_future, draws["gamma"]
        )
        sigma = jnp.exp(jnp.clip(log_sigma, -8.0, 3.0))
    else:
        sigma = draws["gamma0"][:, None, None] * jnp.ones_like(regression)
    return regression, sigma


def _decay_weights(rho: jnp.ndarray, horizon: int) -> jnp.ndarray:
    """Lower-triangular AR carry weights ``W[s, h, j] = rho_s^(h - j)``."""
    steps = jnp.arange(horizon)
    lag = steps[:, None] - steps[None, :]
    return jnp.where(lag >= 0, rho[:, None, None] ** lag, 0.0)


def origin_residuals(
    draws: dict[str, jnp.ndarray],
    y: np.ndarray,
    x: np.ndarray,
    positions: np.ndarray,
) -> jnp.ndarray:
    """Observed regression residual just before each origin, ``(S, O)``.

    ``positions`` indexes the origins in the history arrays; the residual
    at ``position - 1`` is the last realised observation before forecast
    issue, which is all a first-order error needs.
    """
    before = positions - 1
    return jnp.asarray(y)[before][None, :] - jnp.einsum(
        "ok,sk->so", jnp.asarray(x)[before], draws["beta"]
    )


def simulate_horizon_paths(
    draws: dict[str, jnp.ndarray],
    e_origin: jnp.ndarray,
    x_future: np.ndarray,
    xv_future: np.ndarray,
    bsts: BstsConfig,
    seed: int,
    chunk: int = 200,
) -> np.ndarray:
    """Simulate coherent predictive paths for every origin and draw.

    The AR error at step h decomposes exactly into the carried origin
    residual ``rho^(h+1) e_origin`` plus a weighted sum of future
    innovations, so paths are one einsum per chunk rather than a scan.

    Returns
    -------
    numpy.ndarray
        Standardised predictive paths, shape ``(S, O, H)``, float32.
    """
    horizon = x_future.shape[1]
    x_future = jnp.asarray(x_future)
    xv_future = jnp.asarray(xv_future)

    def one_chunk(block: dict[str, jnp.ndarray], e_block: jnp.ndarray, key) -> jnp.ndarray:
        regression, sigma = _scales_and_regression(block, x_future, xv_future, bsts.heteroskedastic)
        weights = _decay_weights(block["rho"], horizon)
        carry = block["rho"][:, None, None] ** jnp.arange(1, horizon + 1) * e_block[:, :, None]
        noise = sigma * jax.random.normal(key, sigma.shape)
        return regression + carry + jnp.einsum("shj,soj->soh", weights, noise)

    chunk_fn = jax.jit(one_chunk)
    n_draws = e_origin.shape[0]
    blocks = []
    for i in range(0, n_draws, chunk):
        block = {name: jnp.asarray(value[i : i + chunk]) for name, value in draws.items()}
        key = jax.random.PRNGKey(seed + i)
        blocks.append(np.asarray(chunk_fn(block, e_origin[i : i + chunk], key), dtype=np.float32))
    return np.concatenate(blocks, axis=0)


def decompose_horizon_variance(
    draws: dict[str, jnp.ndarray],
    e_origin: jnp.ndarray,
    x_future: np.ndarray,
    xv_future: np.ndarray,
    bsts: BstsConfig,
    chunk: int = 200,
) -> dict[str, np.ndarray]:
    """Split the predictive variance at each horizon step into named sources.

    Conditional on one draw the h-step predictive is Gaussian with mean
    ``x'beta + rho^(h+1) e_origin`` and variance ``sum_j rho^(2(h-j))
    sigma_j^2``, so the law of total variance gives exactly two terms:

    - ``parameter``: variance across draws of the conditional mean — all
      the epistemic uncertainty there is, since the origin residual is
      observed rather than filtered (the state term of the trend models is
      structurally zero here);
    - ``innovation``: mean across draws of the accumulated innovation
      variance — the aleatoric noise, playing the roles that process and
      observation variance split between them under the trend models.

    Returns
    -------
    dict of numpy.ndarray
        The two components, each ``(O, H)``, standardised variance units.
    """
    horizon = x_future.shape[1]
    x_future = jnp.asarray(x_future)
    xv_future = jnp.asarray(xv_future)

    def one_chunk(block: dict[str, jnp.ndarray], e_block: jnp.ndarray):
        regression, sigma = _scales_and_regression(block, x_future, xv_future, bsts.heteroskedastic)
        weights = _decay_weights(block["rho"], horizon)
        carry = block["rho"][:, None, None] ** jnp.arange(1, horizon + 1) * e_block[:, :, None]
        conditional_var = jnp.einsum("shj,soj->soh", weights**2, sigma**2)
        return regression + carry, conditional_var

    chunk_fn = jax.jit(one_chunk)
    n_draws = e_origin.shape[0]
    means, variances = [], []
    for i in range(0, n_draws, chunk):
        block = {name: jnp.asarray(value[i : i + chunk]) for name, value in draws.items()}
        mu, var = chunk_fn(block, e_origin[i : i + chunk])
        means.append(np.asarray(mu, dtype=np.float64))
        variances.append(np.asarray(var, dtype=np.float64))
    return {
        "parameter": np.concatenate(means).var(axis=0),
        "innovation": np.concatenate(variances).mean(axis=0),
    }
