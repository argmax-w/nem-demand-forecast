"""A learned interaction surface: basis-function GP over clock and temperature.

The hand-made interaction columns encode the two non-linearities the EDA
could see. This module lets the data choose the rest of the surface while
keeping every downstream guarantee: the Gaussian process over (time of
day, temperature) is approximated by a truncated spectral basis, periodic
Fourier features in the daily phase crossed with Hilbert-space
eigenfunctions in temperature, so the model stays linear in the weights.
The kernel lives in the weight priors: an amplitude and two lengthscales
set how fast the prior variances decay across the basis, and shrinkage
does the rest. The likelihood, the prediction machinery and the variance
decomposition of the innovations model carry over unchanged because the
basis columns are ordinary design columns and the concatenated weight
vector is exposed as the ``beta`` site.

Domain bounds for the temperature eigenfunctions are fixed constants from
the configuration, not data statistics, so the design stays causal.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from nemforecastdemand.config import BstsConfig, FeatureConfig
from nemforecastdemand.features.calendar import local_phases

GP_HYPER_SITES = ("rho", "beta", "gamma0", "gamma", "amplitude", "ell_time", "ell_temp")


def gp_design(index: pd.DatetimeIndex, temp_c: pd.Series, features: FeatureConfig) -> pd.DataFrame:
    """The spectral basis over (daily phase, temperature), one column each.

    Columns are products of a daily Fourier feature (order ``i``, with
    ``i = 0`` the constant, allowing a smooth temperature main effect) and
    a temperature eigenfunction ``sin(j pi (T - lo) / (hi - lo))``. All
    columns are bounded by construction and named ``gp_*`` so the input
    scalers leave them alone.
    """
    daily, _ = local_phases(index)
    half_range = (features.hsgp_temp_hi - features.hsgp_temp_lo) / 2.0
    centred = np.clip(
        temp_c.to_numpy(dtype=np.float64) - (features.hsgp_temp_lo + half_range),
        -half_range,
        half_range,
    )
    columns: dict[str, np.ndarray] = {}
    for j in range(1, features.hsgp_temp_basis + 1):
        eigen = np.sin(j * np.pi * (centred + half_range) / (2.0 * half_range))
        columns[f"gp_t0_e{j}"] = eigen
        for i in range(1, features.hsgp_time_harmonics + 1):
            columns[f"gp_c{i}_e{j}"] = np.cos(2.0 * np.pi * i * daily) * eigen
            columns[f"gp_s{i}_e{j}"] = np.sin(2.0 * np.pi * i * daily) * eigen
    return pd.DataFrame(columns, index=index)


def gp_metadata(columns: list[str], features: FeatureConfig) -> dict[str, np.ndarray]:
    """Static arrays the model needs: per-GP-column orders and frequencies."""
    gp_columns = [c for c in columns if c.startswith("gp_")]
    time_order = np.array([int(c.split("_")[1][1:]) for c in gp_columns], dtype=np.float64)
    temp_index = np.array([int(c.split("_e")[1]) for c in gp_columns], dtype=np.float64)
    half_range = (features.hsgp_temp_hi - features.hsgp_temp_lo) / 2.0
    return {
        "n_linear": len(columns) - len(gp_columns),
        "time_order": time_order,
        "temp_omega": temp_index * np.pi / (2.0 * half_range),
    }


def innovations_hsgp_model(
    y: jnp.ndarray,
    x_mean: jnp.ndarray,
    x_var: jnp.ndarray,
    gp: dict,
    bsts: BstsConfig,
) -> None:
    """The innovations AR(1) model with the GP surface in its design.

    Identical to :func:`innovations.innovations_model` except that the
    design's trailing GP columns get kernel-structured weight priors: a
    shared amplitude, a squared-exponential decay over daily harmonic
    order (lengthscale in phase units) and the spectral density of a
    squared-exponential kernel over temperature frequency (lengthscale in
    degrees). The combined coefficient vector is exposed as ``beta`` so
    prediction code treats both models identically.
    """
    priors = bsts.priors
    n_linear = gp["n_linear"]

    rho = numpyro.sample("rho", dist.Beta(priors.ar_alpha, priors.ar_beta))
    beta_lin = numpyro.sample(
        "beta_lin", dist.Normal(0.0, priors.coef_scale).expand([n_linear]).to_event(1)
    )
    amplitude = numpyro.sample("amplitude", dist.HalfNormal(0.5))
    ell_time = numpyro.sample("ell_time", dist.LogNormal(np.log(0.10), 0.5))
    ell_temp = numpyro.sample("ell_temp", dist.LogNormal(np.log(8.0), 0.5))
    z_gp = numpyro.sample(
        "z_gp", dist.Normal(0.0, 1.0).expand([gp["time_order"].shape[0]]).to_event(1)
    )
    time_decay = jnp.exp(-2.0 * (jnp.pi * gp["time_order"] * ell_time) ** 2)
    temp_sd = jnp.sqrt(jnp.sqrt(2.0 * jnp.pi) * ell_temp) * jnp.exp(
        -0.25 * ell_temp**2 * gp["temp_omega"] ** 2
    )
    weights = z_gp * amplitude * time_decay * temp_sd
    beta = numpyro.deterministic("beta", jnp.concatenate([beta_lin, weights]))

    if bsts.heteroskedastic:
        gamma0 = numpyro.sample(
            "gamma0", dist.Normal(priors.var_intercept_loc, priors.var_intercept_scale)
        )
        gamma = numpyro.sample(
            "gamma",
            dist.Normal(0.0, priors.var_coef_scale).expand([x_var.shape[1]]).to_event(1),
        )
        sigma = jnp.exp(jnp.clip(gamma0 + x_var @ gamma, -8.0, 3.0))
    else:
        sigma = numpyro.sample("gamma0", dist.HalfNormal(priors.obs_scale)) * jnp.ones(
            x_var.shape[0]
        )

    e = y - x_mean @ beta
    numpyro.sample("e_first", dist.Normal(0.0, sigma[0] / jnp.sqrt(1.0 - rho**2)), obs=e[0])
    numpyro.sample("innovations", dist.Normal(rho * e[:-1], sigma[1:]), obs=e[1:])
