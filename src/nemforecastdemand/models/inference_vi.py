"""ADVI fitting for the BSTS, with the ELBO decomposed as it trains.

Two surrogate families share the training loop: a mean-field Gaussian
(``AutoNormal``) and a full-rank Gaussian (``AutoMultivariateNormal``). Both
operate in the unconstrained space induced by NumPyro's transforms.

The logged decomposition uses ELBO = energy + entropy, where the energy is
the expected log joint under the surrogate (including the Jacobian of the
constraining transforms) and the entropy is the surrogate's differential
entropy, available in closed form for both Gaussian families:

    H = D/2 (1 + log 2 pi) + sum(log sigma_i)            (mean-field)
    H = D/2 (1 + log 2 pi) + sum(log diag(L))            (full-rank)

The ELBO itself is Monte-Carlo estimated with a larger particle count than
the training gradient uses, and the energy follows by subtraction, so the
decomposition adds no extra model evaluations beyond the periodic checkpoint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpyro.infer import SVI, Trace_ELBO, autoguide

from nemforecastdemand.config import ViConfig
from nemforecastdemand.evaluation.diagnostics import ElboTrace
from nemforecastdemand.utils import tree_to_float32

GUIDE_KINDS = ("meanfield", "fullrank")


@dataclass
class ViFit:
    """A fitted surrogate and everything the analysis needs from it."""

    kind: str
    params: dict
    guide: object
    trace: ElboTrace
    timings: dict[str, float] = field(default_factory=dict)
    device: str = ""

    def entropy(self) -> float:
        """Closed-form entropy of the fitted Gaussian surrogate."""
        return float(_gaussian_entropy(self.params))

    def posterior_draws(self, model_fn: Callable, seed: int, n_draws: int) -> dict[str, np.ndarray]:
        """Sample constrained posterior draws from the surrogate."""
        samples = self.guide.sample_posterior(
            jax.random.PRNGKey(seed), self.params, sample_shape=(n_draws,)
        )
        return tree_to_float32({name: np.asarray(value) for name, value in samples.items()})


def make_guide(kind: str, model_fn: Callable) -> autoguide.AutoGuide:
    """Construct the surrogate family.

    The full-rank guide starts with a smaller initial scale: with several
    thousand latent dimensions its early ELBO is dominated by the prior
    volume, and a wide start makes the first optimisation steps erratic.
    """
    if kind == "meanfield":
        return autoguide.AutoNormal(model_fn, init_scale=0.1)
    if kind == "fullrank":
        return autoguide.AutoMultivariateNormal(model_fn, init_scale=0.02)
    raise ValueError(f"unknown guide kind {kind!r}, expected one of {GUIDE_KINDS}")


def _gaussian_entropy(params: dict) -> jnp.ndarray:
    """Entropy of the surrogate, covering both parameter layouts.

    AutoNormal stores one ``{site}_auto_loc`` and ``{site}_auto_scale`` pair
    per latent site; AutoMultivariateNormal packs every latent into a single
    ``auto_loc`` vector with a shared ``auto_scale_tril``.
    """
    if "auto_loc" in params:
        dims = params["auto_loc"].size
        log_scale = jnp.sum(jnp.log(jnp.diagonal(params["auto_scale_tril"])))
    else:
        scales = [value for name, value in params.items() if name.endswith("_auto_scale")]
        dims = sum(scale.size for scale in scales)
        log_scale = sum(jnp.sum(jnp.log(scale)) for scale in scales)
    return 0.5 * dims * (1.0 + jnp.log(2.0 * jnp.pi)) + log_scale


def fit_advi(
    model_fn: Callable,
    kind: str,
    vi: ViConfig,
    seed: int,
) -> ViFit:
    """Fit a surrogate by stochastic gradient ascent on the ELBO.

    Parameters
    ----------
    model_fn
        Zero-argument NumPyro model (data closed over), so every JIT trace
        is argument-free and compiles once.
    kind
        ``meanfield`` or ``fullrank``.
    vi
        Optimisation settings.
    seed
        PRNG seed; the evaluation stream is split from the training stream.

    Returns
    -------
    ViFit
        Fitted parameters, the ELBO decomposition trace and timings with
        compilation separated from optimisation.
    """
    guide = make_guide(kind, model_fn)
    schedule = optax.exponential_decay(
        init_value=vi.learning_rate,
        transition_steps=max(vi.steps // 4, 1),
        decay_rate=0.5,
    )
    optimiser = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    svi = SVI(model_fn, guide, optimiser, Trace_ELBO(num_particles=vi.num_particles))
    eval_elbo = Trace_ELBO(num_particles=vi.eval_particles)

    rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(seed))
    state = svi.init(rng_init)
    update = jax.jit(svi.update)

    def checkpoint(rng, params):
        return -eval_elbo.loss(rng, params, model_fn, guide)

    checkpoint = jax.jit(checkpoint)

    timings: dict[str, float] = {}
    start = time.perf_counter()
    state, _ = update(state)
    jax.block_until_ready(state)
    timings["compile_seconds"] = time.perf_counter() - start

    steps, elbos, entropies = [], [], []
    start = time.perf_counter()
    for step in range(1, vi.steps):
        state, _ = update(state)
        if step % vi.log_every == 0 or step == vi.steps - 1:
            params = svi.get_params(state)
            rng_eval, rng_step = jax.random.split(rng_eval)
            steps.append(step)
            elbos.append(float(checkpoint(rng_step, params)))
            entropies.append(float(_gaussian_entropy(params)))
    jax.block_until_ready(state)
    timings["fit_seconds"] = time.perf_counter() - start
    timings["steps_per_second"] = (vi.steps - 1) / timings["fit_seconds"]

    elbo = np.array(elbos)
    entropy = np.array(entropies)
    trace = ElboTrace(steps=np.array(steps), elbo=elbo, energy=elbo - entropy, entropy=entropy)
    return ViFit(
        kind=kind,
        params=svi.get_params(state),
        guide=guide,
        trace=trace,
        timings=timings,
        device=jax.devices()[0].platform,
    )


def _site_params(params: dict, suffix: str) -> dict[str, jnp.ndarray]:
    return {
        name.removesuffix(suffix): value for name, value in params.items() if name.endswith(suffix)
    }


def surrogate_mass_matrix(fit: ViFit) -> tuple[np.ndarray, np.ndarray, bool]:
    """Surrogate mean and covariance in NumPyro's flat latent ordering.

    NumPyro flattens the unconstrained latent dict with ``ravel_pytree``
    (keys sorted). The mean-field guide already stores per-site parameters,
    so flattening the site dict gives the right order directly; the
    full-rank guide packs latents in its own order, so an index vector is
    pushed through its unpacking to build the permutation.

    Returns
    -------
    tuple
        ``(loc, inverse_mass, dense)``: the flat surrogate mean, the
        surrogate variance vector (mean-field) or covariance matrix
        (full-rank) to be used as the inverse mass matrix, and whether the
        mass matrix is dense.
    """
    if "auto_loc" in fit.params:
        loc_packed = jnp.asarray(fit.params["auto_loc"])
        index_dict = fit.guide._unpack_latent(jnp.arange(loc_packed.size))
        perm, _ = jax.flatten_util.ravel_pytree(index_dict)
        perm = np.asarray(perm, dtype=np.int64)
        tril = np.asarray(fit.params["auto_scale_tril"], dtype=np.float64)
        cov = tril @ tril.T
        return np.asarray(loc_packed)[perm], cov[np.ix_(perm, perm)], True

    locs = _site_params(fit.params, "_auto_loc")
    scales = _site_params(fit.params, "_auto_scale")
    loc_flat, _ = jax.flatten_util.ravel_pytree(locs)
    scale_flat, _ = jax.flatten_util.ravel_pytree(scales)
    return np.asarray(loc_flat), np.asarray(scale_flat) ** 2, False


def sample_unconstrained(fit: ViFit, seed: int, n_draws: int) -> dict[str, np.ndarray]:
    """Draw unconstrained latent dicts from the surrogate, for NUTS inits."""
    key = jax.random.PRNGKey(seed)
    if "auto_loc" in fit.params:
        loc = jnp.asarray(fit.params["auto_loc"])
        eps = jax.random.normal(key, (n_draws, loc.size))
        packed = loc + eps @ jnp.asarray(fit.params["auto_scale_tril"]).T
        unpacked = jax.vmap(fit.guide._unpack_latent)(packed)
        return {name: np.asarray(value) for name, value in unpacked.items()}

    locs = _site_params(fit.params, "_auto_loc")
    scales = _site_params(fit.params, "_auto_scale")
    out = {}
    for name, loc in locs.items():
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, (n_draws, *loc.shape))
        out[name] = np.asarray(loc + eps * scales[name])
    return out
