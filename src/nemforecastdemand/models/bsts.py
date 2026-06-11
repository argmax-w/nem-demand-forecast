"""The Bayesian structural time-series model and its prediction machinery.

One generative model, defined once and fitted two ways (ADVI and NUTS), so
the comparison isolates the inference algorithm.

Structure, on demand standardised over the fitting window:

- stochastic local linear trend: a random-walk level plus a damped AR(1)
  slope, written as an explicit state-space recursion with ``lax.scan``;
- static seasonal regression on the shared design matrix (local-clock
  seasonal basis, weather, demand lags, holiday);
- heteroskedastic Gaussian observations: the log observation scale is
  linear in a small variance design, so predictive spread follows the
  covariates; a Student-t family is available as a robustness option.

Parameterisation is non-centred throughout: innovations enter as standard
normal draws scaled inside the recursion, which removes the funnel between
innovation scales and states and gives NUTS a tractable geometry. The
variance head's coefficients carry deliberately tight priors because the
exponential link compounds across the design.

Prediction is Rao-Blackwellised. Conditional on the hyperparameters the
model is linear-Gaussian, so rolling-origin forecasts run a Kalman filter
over the history once per posterior draw and then simulate forward over each
horizon, rather than refitting per origin. The simulation produces jointly
coherent 48-step paths, which the energy score in the evaluation needs;
marginal CRPS gets the same draws. Filtering, prediction and simulation are
vectorised over draws (and origins) with ``vmap`` and chunked so device
memory stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from nemforecastdemand.config import BstsConfig, Config
from nemforecastdemand.models.base import build_design, variance_design

HYPER_SITES = (
    "sigma_level",
    "sigma_slope",
    "phi",
    "beta",
    "gamma0",
    "gamma",
    "level_init",
    "slope_init",
)


@dataclass
class BstsInputs:
    """Standardised arrays for fitting and prediction.

    The mean design uses actual weather over history (it has realised by the
    time a forecast is issued); horizon designs are built per variant at
    prediction time. Scalers are stored so everything returns to megawatts.
    """

    index: pd.DatetimeIndex
    y: np.ndarray
    x_mean: np.ndarray
    x_var: np.ndarray
    y_loc: float
    y_scale: float
    x_loc: np.ndarray
    x_scale: np.ndarray
    xv_loc: np.ndarray
    xv_scale: np.ndarray
    columns: list[str]


def prepare_inputs(panel: pd.DataFrame, cfg: Config, fit_index: pd.DatetimeIndex) -> BstsInputs:
    """Standardise the target and designs on the fitting window.

    Parameters
    ----------
    panel
        Full processed panel (contiguous splits concatenated).
    cfg
        Project configuration.
    fit_index
        The window the model is fitted on; scalers come from here only, so
        later data never leaks into the standardisation.
    """
    design = build_design(panel, cfg, weather_source="actual").loc[fit_index]
    if design.isna().any().any():
        raise ValueError("fit window starts inside the demand-lag warmup")
    vdesign = variance_design(panel, cfg, weather_source="actual").loc[fit_index]
    y = panel["demand_mw"].loc[fit_index].to_numpy(dtype=np.float64)

    y_loc, y_scale = float(y.mean()), float(y.std())
    x_loc = design.mean().to_numpy()
    x_scale = design.std().replace(0.0, 1.0).to_numpy()
    xv_loc = vdesign.mean().to_numpy()
    xv_scale = vdesign.std().replace(0.0, 1.0).to_numpy()

    return BstsInputs(
        index=fit_index,
        y=((y - y_loc) / y_scale).astype(np.float32),
        x_mean=((design.to_numpy() - x_loc) / x_scale).astype(np.float32),
        x_var=((vdesign.to_numpy() - xv_loc) / xv_scale).astype(np.float32),
        y_loc=y_loc,
        y_scale=y_scale,
        x_loc=x_loc,
        x_scale=x_scale,
        xv_loc=xv_loc,
        xv_scale=xv_scale,
        columns=list(design.columns),
    )


def transform_design(inputs: BstsInputs, design: pd.DataFrame, vdesign: pd.DataFrame):
    """Standardise out-of-window designs with the stored fit-window scalers."""
    x = ((design.to_numpy() - inputs.x_loc) / inputs.x_scale).astype(np.float32)
    z = ((vdesign.to_numpy() - inputs.xv_loc) / inputs.xv_scale).astype(np.float32)
    return x, z


def bsts_model(
    y: jnp.ndarray | None,
    x_mean: jnp.ndarray,
    x_var: jnp.ndarray,
    bsts: BstsConfig,
) -> None:
    """The NumPyro model shared by ADVI and NUTS.

    Parameters
    ----------
    y
        Standardised observations, shape ``(T,)``, or None for prior draws.
    x_mean
        Standardised mean design, shape ``(T, K)``.
    x_var
        Standardised variance design, shape ``(T, J)``.
    bsts
        Structural settings and priors.
    """
    priors = bsts.priors
    n_steps, n_coefs = x_mean.shape

    sigma_level = numpyro.sample("sigma_level", dist.HalfNormal(priors.level_scale))
    sigma_slope = numpyro.sample("sigma_slope", dist.HalfNormal(priors.slope_scale))
    if bsts.damped_slope:
        phi = numpyro.sample("phi", dist.Beta(priors.damping_alpha, priors.damping_beta))
    else:
        phi = 1.0
    beta = numpyro.sample("beta", dist.Normal(0.0, priors.coef_scale).expand([n_coefs]).to_event(1))
    level_init = numpyro.sample("level_init", dist.Normal(0.0, priors.init_level_scale))
    slope_init = numpyro.sample("slope_init", dist.Normal(0.0, priors.init_slope_scale))

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
        log_sigma = jnp.clip(gamma0 + x_var @ gamma, -8.0, 3.0)
        sigma_obs = jnp.exp(log_sigma)
    else:
        sigma_obs = numpyro.sample("gamma0", dist.HalfNormal(priors.obs_scale))

    z_level = numpyro.sample("z_level", dist.Normal(0.0, 1.0).expand([n_steps]).to_event(1))
    z_slope = numpyro.sample("z_slope", dist.Normal(0.0, 1.0).expand([n_steps]).to_event(1))

    def step(carry, inputs):
        level, slope = carry
        z_l, z_s = inputs
        level = level + slope + sigma_level * z_l
        slope = phi * slope + sigma_slope * z_s
        return (level, slope), level

    _, levels = jax.lax.scan(step, (level_init, slope_init), (z_level, z_slope))
    numpyro.deterministic("level", levels)
    mu = levels + x_mean @ beta

    if bsts.obs_family == "student_t":
        df = 2.0 + numpyro.sample("df_minus_two", dist.Exponential(priors.student_t_df_rate))
        numpyro.sample("y", dist.StudentT(df, mu, sigma_obs), obs=y)
    else:
        numpyro.sample("y", dist.Normal(mu, sigma_obs), obs=y)


def bsts_collapsed_model(
    y: jnp.ndarray,
    x_mean: jnp.ndarray,
    x_var: jnp.ndarray,
    bsts: BstsConfig,
) -> None:
    """The same generative model with the states marginalised analytically.

    Conditional on the hyperparameters the trend is linear-Gaussian, so the
    latent states integrate out exactly through a Kalman filter and the
    marginal likelihood is the product of one-step Gaussian predictives.
    NUTS and ADVI then work in the roughly fifty-dimensional hyperparameter
    space regardless of the data length, which is what makes a full year of
    half hours tractable; the trade is a sequential filter pass inside
    every gradient evaluation. Priors and structure match
    :func:`bsts_model` exactly; the initial state enters through its prior
    rather than as a sampled site.
    """
    if bsts.obs_family != "gaussian":
        raise ValueError("the collapsed likelihood requires Gaussian observations")
    priors = bsts.priors
    n_coefs = x_mean.shape[1]

    sigma_level = numpyro.sample("sigma_level", dist.HalfNormal(priors.level_scale))
    sigma_slope = numpyro.sample("sigma_slope", dist.HalfNormal(priors.slope_scale))
    phi = (
        numpyro.sample("phi", dist.Beta(priors.damping_alpha, priors.damping_beta))
        if bsts.damped_slope
        else 1.0
    )
    beta = numpyro.sample("beta", dist.Normal(0.0, priors.coef_scale).expand([n_coefs]).to_event(1))
    if bsts.heteroskedastic:
        gamma0 = numpyro.sample(
            "gamma0", dist.Normal(priors.var_intercept_loc, priors.var_intercept_scale)
        )
        gamma = numpyro.sample(
            "gamma",
            dist.Normal(0.0, priors.var_coef_scale).expand([x_var.shape[1]]).to_event(1),
        )
        sigma_obs = jnp.exp(jnp.clip(gamma0 + x_var @ gamma, -8.0, 3.0))
    else:
        sigma_obs = numpyro.sample("gamma0", dist.HalfNormal(priors.obs_scale)) * jnp.ones(
            x_var.shape[0]
        )

    transition = jnp.array([[1.0, 1.0], [0.0, phi]])
    process_cov = jnp.diag(jnp.array([sigma_level**2, sigma_slope**2]))
    regression = x_mean @ beta
    mean0 = jnp.zeros(2)
    cov0 = jnp.diag(jnp.array([priors.init_level_scale**2, priors.init_slope_scale**2]))

    def step(carry, inputs):
        mean, cov = carry
        obs, reg, sd = inputs
        mean = transition @ mean
        cov = transition @ cov @ transition.T + process_cov
        innovation = obs - (mean[0] + reg)
        innovation_var = cov[0, 0] + sd**2
        log_density = -0.5 * (
            jnp.log(2.0 * jnp.pi * innovation_var) + innovation**2 / innovation_var
        )
        gain = cov[:, 0] / innovation_var
        mean = mean + gain * innovation
        cov = cov - jnp.outer(gain, cov[0, :])
        return (mean, cov), log_density

    # The unroll trades a little compile time for far fewer sequential
    # kernel launches, which dominates a long scan of tiny updates.
    _, log_densities = jax.lax.scan(step, (mean0, cov0), (y, regression, sigma_obs), unroll=16)
    numpyro.factor("marginal_loglik", log_densities.sum())


def states_from_draws(
    draws: dict[str, np.ndarray], bsts: BstsConfig, chunk: int = 200
) -> np.ndarray:
    """Reconstruct level paths from non-centred draws, chunked over draws.

    Used for the ADVI-versus-NUTS comparison of the latent state posterior.

    Parameters
    ----------
    draws
        Posterior draws with leading draw dimension, containing the
        innovation z sites and hyperparameters.
    bsts
        Structural settings.
    chunk
        Draws per vmapped block, bounding device memory.

    Returns
    -------
    numpy.ndarray
        Level paths, shape ``(draws, T)``, float32.
    """

    def one_path(draw):
        phi = draw["phi"] if bsts.damped_slope else 1.0

        def step(carry, inputs):
            level, slope = carry
            z_l, z_s = inputs
            level = level + slope + draw["sigma_level"] * z_l
            slope = phi * slope + draw["sigma_slope"] * z_s
            return (level, slope), level

        _, levels = jax.lax.scan(
            step, (draw["level_init"], draw["slope_init"]), (draw["z_level"], draw["z_slope"])
        )
        return levels

    path_fn = jax.jit(jax.vmap(one_path))
    n_draws = draws["z_level"].shape[0]
    sites = ["sigma_level", "sigma_slope", "level_init", "slope_init", "z_level", "z_slope"]
    if bsts.damped_slope:
        sites.append("phi")
    blocks = []
    for i in range(0, n_draws, chunk):
        block = {site: jnp.asarray(draws[site][i : i + chunk]) for site in sites}
        blocks.append(np.asarray(path_fn(block), dtype=np.float32))
    return np.concatenate(blocks, axis=0)


def _observation_scale(draw: dict, x_var: jnp.ndarray, heteroskedastic: bool) -> jnp.ndarray:
    if heteroskedastic:
        return jnp.exp(jnp.clip(draw["gamma0"] + x_var @ draw["gamma"], -8.0, 3.0))
    return draw["gamma0"] * jnp.ones(x_var.shape[0])


def kalman_filter_states(
    draws: dict[str, jnp.ndarray],
    y: jnp.ndarray,
    x_mean: jnp.ndarray,
    x_var: jnp.ndarray,
    bsts: BstsConfig,
    chunk: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtered state means and covariances at every step, per draw.

    Runs once per posterior draw over the full history (fit window plus
    everything up to the end of the evaluation period). Forecasts for any
    origin then start from the filtered state just before that origin, so no
    per-origin filtering or refitting is ever needed.

    Parameters
    ----------
    draws
        Hyperparameter draws, leading dimension S.
    y
        Standardised demand over the full history, shape ``(T,)``.
    x_mean, x_var
        Standardised designs over the same index (actual weather: history
        covariates have realised by forecast time).
    bsts
        Structural settings.
    chunk
        Draws per vmapped block.

    Returns
    -------
    tuple of numpy.ndarray
        Filtered means ``(S, T, 2)`` and covariances ``(S, T, 2, 2)``,
        float32, for the state ``[level, slope]`` after the update at each
        step.
    """
    priors = bsts.priors

    def one_draw(draw):
        phi = draw["phi"] if bsts.damped_slope else 1.0
        transition = jnp.array([[1.0, 1.0], [0.0, phi]])
        process_cov = jnp.diag(jnp.array([draw["sigma_level"] ** 2, draw["sigma_slope"] ** 2]))
        sigma_obs = _observation_scale(draw, x_var, bsts.heteroskedastic)
        regression = x_mean @ draw["beta"]

        mean0 = jnp.array([draw["level_init"], draw["slope_init"]])
        # The recursion emits the state after one transition, so the prior
        # for the first observation is the propagated initial state. The
        # initial uncertainty uses the prior scales: posterior draws pin the
        # init sites too, but the filter treats them as the diffuse anchor.
        cov0 = jnp.diag(jnp.array([priors.init_level_scale**2, priors.init_slope_scale**2]))

        def step(carry, inputs):
            mean, cov = carry
            obs, reg, sd = inputs
            mean = transition @ mean
            cov = transition @ cov @ transition.T + process_cov
            resid = obs - (mean[0] + reg)
            innov_var = cov[0, 0] + sd**2
            gain = cov[:, 0] / innov_var
            mean = mean + gain * resid
            cov = cov - jnp.outer(gain, cov[0, :])
            return (mean, cov), (mean, cov)

        _, (means, covs) = jax.lax.scan(step, (mean0, cov0), (y, regression, sigma_obs))
        return means, covs

    filter_fn = jax.jit(jax.vmap(one_draw))
    n_draws = draws["sigma_level"].shape[0]
    mean_blocks, cov_blocks = [], []
    for i in range(0, n_draws, chunk):
        block = {site: jnp.asarray(value[i : i + chunk]) for site, value in draws.items()}
        means, covs = filter_fn(block)
        mean_blocks.append(np.asarray(means, dtype=np.float32))
        cov_blocks.append(np.asarray(covs, dtype=np.float32))
    return np.concatenate(mean_blocks), np.concatenate(cov_blocks)


def simulate_horizon_paths(
    draws: dict[str, jnp.ndarray],
    filtered_mean: np.ndarray,
    filtered_cov: np.ndarray,
    origin_positions: np.ndarray,
    x_future: np.ndarray,
    xv_future: np.ndarray,
    bsts: BstsConfig,
    seed: int,
    chunk: int = 100,
) -> np.ndarray:
    """Simulate coherent predictive paths for every origin and draw.

    For each posterior draw and origin, the state at the origin is sampled
    from its filtered Gaussian and propagated through the transition with
    sampled innovations and observation noise, giving one jointly coherent
    48-step path per (draw, origin) pair: the posterior predictive mixture.

    Parameters
    ----------
    draws
        Hyperparameter draws, leading dimension S.
    filtered_mean, filtered_cov
        Output of :func:`kalman_filter_states` over the full history.
    origin_positions
        Integer positions of each origin in the history index; the filter
        state at ``position - 1`` (data strictly before the origin) seeds
        the simulation.
    x_future, xv_future
        Standardised horizon designs per origin, shapes ``(O, H, K)`` and
        ``(O, H, J)``, built under the chosen weather variant.
    bsts
        Structural settings.
    seed
        Base PRNG seed.
    chunk
        Draws per vmapped block.

    Returns
    -------
    numpy.ndarray
        Standardised predictive paths, shape ``(S, O, H)``, float32.
    """
    horizon = x_future.shape[1]
    start_mean = jnp.asarray(filtered_mean[:, origin_positions - 1, :])
    start_cov = jnp.asarray(filtered_cov[:, origin_positions - 1, :, :])
    x_future = jnp.asarray(x_future)
    xv_future = jnp.asarray(xv_future)

    def one_pair(draw, mean, cov, x_fut, xv_fut, key):
        phi = draw["phi"] if bsts.damped_slope else 1.0
        transition = jnp.array([[1.0, 1.0], [0.0, phi]])
        chol = jnp.linalg.cholesky(cov + 1e-9 * jnp.eye(2))
        sigma_obs = _observation_scale(draw, xv_fut, bsts.heteroskedastic)
        regression = x_fut @ draw["beta"]

        key_init, key_state, key_obs = jax.random.split(key, 3)
        state = mean + chol @ jax.random.normal(key_init, (2,))
        state_noise = jax.random.normal(key_state, (horizon, 2)) * jnp.array(
            [draw["sigma_level"], draw["sigma_slope"]]
        )
        obs_noise = jax.random.normal(key_obs, (horizon,)) * sigma_obs

        def step(carry, inputs):
            current = carry
            noise, reg, eps = inputs
            current = transition @ current + noise
            return current, current[0] + reg + eps

        _, path = jax.lax.scan(step, state, (state_noise, regression, obs_noise))
        return path

    over_origins = jax.vmap(one_pair, in_axes=(None, 0, 0, 0, 0, 0))
    over_draws = jax.jit(jax.vmap(over_origins, in_axes=(0, 0, 0, None, None, 0)))

    n_draws = start_mean.shape[0]
    n_origins = x_future.shape[0]
    blocks = []
    for i in range(0, n_draws, chunk):
        block = {site: jnp.asarray(value[i : i + chunk]) for site, value in draws.items()}
        size = block["sigma_level"].shape[0]
        keys = jax.random.split(jax.random.PRNGKey(seed + i), size * n_origins).reshape(
            size, n_origins, 2
        )
        paths = over_draws(
            block, start_mean[i : i + size], start_cov[i : i + size], x_future, xv_future, keys
        )
        blocks.append(np.asarray(paths, dtype=np.float32))
    return np.concatenate(blocks, axis=0)
