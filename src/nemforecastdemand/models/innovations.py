"""Innovations-form AR(2) regression: the project's BSTS model.

A seasonal regression on the shared design with a stationary AR(2) error and
a heteroskedastic observation scale, fitted by ADVI and NUTS,

    e_t = y_t - x_t' beta,
    e_t - rho1 e_{t-1} - rho2 e_{t-2} ~ Normal(0, sigma_t),

and the first two residuals anchored at their stationary distribution. The
second lag is what an AR(1) error cannot supply: AR(1) carries only the
residual level forward and reverts toward the regression mean at a fixed
rate, so its near-origin slope is fixed by that one number. AR(2) carries
the last two residuals, so the forecast has a level and a slope at the
origin, the curvature the residual diagnostics show (a large negative
lag-2 partial autocorrelation). The coefficients are sampled through their
partial autocorrelations (phi1, phi2 in (-1, 1)), which guarantees
stationarity, with rho1 = phi1 (1 - phi2) and rho2 = phi2.

Writing the likelihood on the innovations rather than on latent error states
removes the sequential dependence entirely: residuals come from one matrix
product and the innovations from shifted differences, so there is no scan,
no Kalman pass and nothing the GPU dislikes. Because the error is second-
order Markov, prediction conditions only on the two residuals at the
forecast origin, both observed exactly, and the h-step predictive moments
are closed form in the AR(2) impulse response.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from nemforecastdemand.config import BstsConfig

HYPER_SITES = ("phi1", "phi2", "beta", "gamma0", "gamma")

# Support of each hyperparameter, for mapping draws to the unconstrained space
# where the variational guide is Gaussian. The lag-1 partial autocorrelation
# lives on the unit interval and the lag-2 on (-1, 1); the regression and the
# heteroskedastic log-scale coefficients are unconstrained.
HYPER_SUPPORTS = {
    "phi1": "unit_interval",
    "phi2": ("interval", -1.0, 1.0),
    "beta": "real",
    "gamma0": "real",
    "gamma": "real",
}


def innovations_model(
    y: jnp.ndarray,
    x_mean: jnp.ndarray,
    x_var: jnp.ndarray,
    bsts: BstsConfig,
) -> None:
    """Regression with a stationary AR(2) error, fully vectorised.

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

    # Partial autocorrelations in (-1, 1) keep the AR(2) stationary for any
    # draw; phi1 is the lag-1 PACF (strongly positive), phi2 the lag-2.
    phi1 = numpyro.sample("phi1", dist.Beta(priors.ar_alpha, priors.ar_beta))
    phi2 = numpyro.sample("phi2", dist.Uniform(-1.0, 1.0))
    rho1 = numpyro.deterministic("rho1", phi1 * (1.0 - phi2))
    rho2 = numpyro.deterministic("rho2", phi2)

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
    # Stationary bivariate anchor for the first two residuals, using the
    # scale at the first step (exact under homoskedasticity, a two-observation
    # approximation otherwise).
    var0 = sigma[0] ** 2 * (1.0 - rho2) / ((1.0 + rho2) * ((1.0 - rho2) ** 2 - rho1**2))
    sd0 = jnp.sqrt(var0)
    lag1_corr = rho1 / (1.0 - rho2)
    numpyro.sample("e0", dist.Normal(0.0, sd0), obs=e[0])
    numpyro.sample(
        "e1", dist.Normal(lag1_corr * e[0], sd0 * jnp.sqrt(1.0 - lag1_corr**2)), obs=e[1]
    )
    numpyro.sample(
        "innovations", dist.Normal(rho1 * e[1:-1] + rho2 * e[:-2], sigma[2:]), obs=e[2:]
    )


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


def _ar2_coeffs(draws: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """AR(2) coefficients ``(rho1, rho2)`` from the sampled partial autocorrelations."""
    phi1, phi2 = draws["phi1"], draws["phi2"]
    return phi1 * (1.0 - phi2), phi2


def _ar2_weights(
    rho1: jnp.ndarray, rho2: jnp.ndarray, horizon: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """AR(2) forecast weights for every draw.

    Returns
    -------
    tuple
        ``A`` and ``B`` of shape ``(S, H)``: the carry coefficients so the
        deterministic error at step ``h`` is ``A[h] e_last + B[h] e_prev``;
        and ``W`` of shape ``(S, H, H)``, the lower-triangular MA weights
        ``W[h, j] = psi_{h-j}`` with ``psi`` the AR(2) impulse response.
    """
    one, zero = jnp.ones_like(rho1), jnp.zeros_like(rho1)
    # Carry coefficients: A_k (on e_last) and B_k (on e_prev), k = 1..H.
    a_terms, b_terms = [rho1], [rho2]  # k = 1
    a_pp, a_p = one, rho1  # A_0, A_1
    b_pp, b_p = zero, rho2  # B_0, B_1
    for _ in range(2, horizon + 1):
        a_pp, a_p = a_p, rho1 * a_p + rho2 * a_pp
        b_pp, b_p = b_p, rho1 * b_p + rho2 * b_pp
        a_terms.append(a_p)
        b_terms.append(b_p)
    A = jnp.stack(a_terms, axis=1)
    B = jnp.stack(b_terms, axis=1)
    # Impulse response psi_0..H-1, psi_0 = 1, psi_1 = rho1.
    psi_terms = [one]
    if horizon > 1:
        psi_terms.append(rho1)
        p_pp, p_p = one, rho1
        for _ in range(2, horizon):
            p_pp, p_p = p_p, rho1 * p_p + rho2 * p_pp
            psi_terms.append(p_p)
    psi = jnp.stack(psi_terms, axis=1)  # (S, H)
    steps = jnp.arange(horizon)
    lag = steps[:, None] - steps[None, :]
    W = jnp.where(lag >= 0, psi[:, jnp.clip(lag, 0, horizon - 1)], 0.0)
    return A, B, W


def _ar2_carry(A: jnp.ndarray, B: jnp.ndarray, e_block: jnp.ndarray) -> jnp.ndarray:
    """Deterministic AR(2) error carried from the two origin residuals, ``(S, O, H)``."""
    return A[:, None, :] * e_block[..., 0][:, :, None] + B[:, None, :] * e_block[..., 1][:, :, None]


def origin_residuals(
    draws: dict[str, jnp.ndarray],
    y: np.ndarray,
    x: np.ndarray,
    positions: np.ndarray,
) -> jnp.ndarray:
    """The two observed residuals just before each origin, ``(S, O, 2)``.

    ``positions`` indexes the origins in the history arrays; ``[..., 0]`` is
    the residual at ``position - 1`` (the last realised observation) and
    ``[..., 1]`` the one at ``position - 2``, which is all a second-order
    error needs.
    """
    y = jnp.asarray(y)
    x = jnp.asarray(x)

    def resid_at(idx: np.ndarray) -> jnp.ndarray:
        return y[idx][None, :] - jnp.einsum("ok,sk->so", x[idx], draws["beta"])

    return jnp.stack([resid_at(positions - 1), resid_at(positions - 2)], axis=-1)


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

    The AR(2) error at step h decomposes exactly into the carried origin
    residuals (``A_h e_last + B_h e_prev``) plus a weighted sum of future
    innovations through the impulse response, so paths are a couple of
    einsums per chunk rather than a scan.

    Returns
    -------
    numpy.ndarray
        Standardised predictive paths, shape ``(S, O, H)``, float32.
    """
    horizon = x_future.shape[1]
    x_future = jnp.asarray(x_future)
    xv_future = jnp.asarray(xv_future)

    def one_chunk(block: dict[str, jnp.ndarray], e_block: jnp.ndarray, key) -> jnp.ndarray:
        rho1, rho2 = _ar2_coeffs(block)
        regression, sigma = _scales_and_regression(block, x_future, xv_future, bsts.heteroskedastic)
        A, B, W = _ar2_weights(rho1, rho2, horizon)
        carry = _ar2_carry(A, B, e_block)
        noise = sigma * jax.random.normal(key, sigma.shape)
        return regression + carry + jnp.einsum("shj,soj->soh", W, noise)

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
    ``x'beta + A_h e_last + B_h e_prev`` and variance ``sum_j psi_{h-j}^2
    sigma_j^2``, so the law of total variance gives exactly two terms:

    - ``parameter``: variance across draws of the conditional mean, all the
      epistemic uncertainty there is, since the origin residuals are observed
      rather than filtered (there is no separate state term);
    - ``innovation``: mean across draws of the accumulated innovation
      variance, the irreducible aleatoric noise.

    Returns
    -------
    dict of numpy.ndarray
        The two components, each ``(O, H)``, standardised variance units.
    """
    horizon = x_future.shape[1]
    x_future = jnp.asarray(x_future)
    xv_future = jnp.asarray(xv_future)

    def one_chunk(block: dict[str, jnp.ndarray], e_block: jnp.ndarray):
        rho1, rho2 = _ar2_coeffs(block)
        regression, sigma = _scales_and_regression(block, x_future, xv_future, bsts.heteroskedastic)
        A, B, W = _ar2_weights(rho1, rho2, horizon)
        carry = _ar2_carry(A, B, e_block)
        conditional_var = jnp.einsum("shj,soj->soh", W**2, sigma**2)
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
