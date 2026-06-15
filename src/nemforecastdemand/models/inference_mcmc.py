"""NUTS fitting for the BSTS, with diagnostics and the ADVI warm start.

NUTS with multiple chains is treated as the reference posterior. Warmup and
sampling are timed separately (warmup includes XLA compilation, which both
cold and warm runs pay alike), and the sampler's extra fields are kept so
notebooks can show divergences, energies, tree depths and acceptance
statistics alongside the trace plots.

The warm start seeds two things from a fitted ADVI surrogate: the initial
position of every chain, drawn from q, and the inverse mass matrix, taken as
the surrogate covariance (dense, from the full-rank guide) with mass
adaptation frozen. Step-size adaptation stays on; it is cheap and
absorbs any global misscaling of the surrogate. The honest comparison
against the cold run happens at matched quality in the analysis: total
wall-clock to a target bulk ESS with clean R-hat, never raw wall-clock at
mismatched mixing. If the surrogate under-estimates variance, the derived
mass matrix can mis-scale the sampler and the warm run may mix worse or
diverge; that outcome is reported, not hidden.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

from nemforecastdemand.config import NutsConfig
from nemforecastdemand.evaluation.diagnostics import mcmc_summary, sampler_health
from nemforecastdemand.models.inference_vi import (
    ViFit,
    sample_unconstrained,
    surrogate_mass_matrix,
)
from nemforecastdemand.utils import tree_to_float32

EXTRA_FIELDS = ("diverging", "energy", "num_steps", "accept_prob")


@dataclass
class WarmStart:
    """Initialisation derived from a fitted surrogate.

    With ``freeze_mass`` the surrogate covariance becomes the fixed
    inverse mass matrix and mass adaptation is disabled; without it the
    surrogate supplies starting positions only and NUTS adapts its own
    mass matrix, which suits posteriors whose geometry the guide cannot
    capture (hierarchical funnels).
    """

    init_params: dict[str, np.ndarray]
    inverse_mass_matrix: np.ndarray
    dense: bool
    source: str
    freeze_mass: bool = True


@dataclass
class NutsRun:
    """One NUTS run: draws, sampler statistics and timings."""

    posterior: dict[str, np.ndarray]
    extra: dict[str, np.ndarray]
    timings: dict[str, float] = field(default_factory=dict)
    device: str = ""
    settings: dict = field(default_factory=dict)

    def summary(self) -> object:
        """Per-site convergence table (worst R-hat first)."""
        sites = {k: v for k, v in self.posterior.items() if k != "level"}
        return mcmc_summary(sites)

    def health(self, max_tree_depth: int) -> object:
        """Per-chain divergences, E-BFMI and tree-depth saturation."""
        return sampler_health(self.extra, max_tree_depth)


def warm_start_from_vi(fit: ViFit, chains: int, seed: int) -> WarmStart:
    """Build chain initialisations and an inverse mass matrix from ADVI."""
    _, inverse_mass, dense = surrogate_mass_matrix(fit)
    init = sample_unconstrained(fit, seed, chains)
    return WarmStart(
        init_params=init,
        inverse_mass_matrix=inverse_mass,
        dense=dense,
        source=fit.kind,
    )


def fit_nuts(
    model_fn: Callable,
    nuts: NutsConfig,
    seed: int,
    warmup: int | None = None,
    warm_start: WarmStart | None = None,
    progress: bool = True,
) -> NutsRun:
    """Run NUTS, cold or warm started.

    Parameters
    ----------
    model_fn
        Zero-argument NumPyro model with the data closed over.
    nuts
        Sampler settings.
    seed
        PRNG seed.
    warmup
        Override for the number of warmup iterations (the warm-start runs
        use a reduced schedule); defaults to the configured value.
    warm_start
        Optional surrogate-derived initialisation. When given, chains start
        from draws of q, the inverse mass matrix is fixed to the surrogate
        covariance and mass adaptation is disabled. Step-size adaptation
        remains on either way.
    progress
        Show the sampler's progress bar. Vectorised chains share one bar.
        The bar runs the iteration loop on the host instead of one fused
        scan; the per-iteration dispatch overhead is the same on every
        device, so timed comparisons remain like for like.

    Returns
    -------
    NutsRun
        Draws grouped by chain, extra fields, timings and settings.
    """
    num_warmup = nuts.warmup if warmup is None else warmup
    kernel_kwargs: dict = {
        "target_accept_prob": nuts.target_accept,
        "max_tree_depth": nuts.max_tree_depth,
    }
    init_params = None
    if warm_start is not None:
        if warm_start.freeze_mass:
            kernel_kwargs["inverse_mass_matrix"] = jax.numpy.asarray(warm_start.inverse_mass_matrix)
            kernel_kwargs["dense_mass"] = warm_start.dense
            kernel_kwargs["adapt_mass_matrix"] = False
        init_params = {
            name: jax.numpy.asarray(value) for name, value in warm_start.init_params.items()
        }

    kernel = NUTS(model_fn, **kernel_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=nuts.samples,
        num_chains=nuts.chains,
        chain_method=nuts.chain_method,
        progress_bar=progress,
    )

    timings: dict[str, float] = {}
    rng = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    mcmc.warmup(rng, init_params=init_params, collect_warmup=False, extra_fields=EXTRA_FIELDS)
    # JAX dispatches asynchronously, so without an explicit barrier the timer
    # records dispatch rather than computation and the warmup cost leaks into
    # whichever later call happens to block first.
    jax.block_until_ready(mcmc.post_warmup_state)
    timings["warmup_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=EXTRA_FIELDS)
    jax.block_until_ready(mcmc.last_state)
    timings["sample_seconds"] = time.perf_counter() - start

    posterior = tree_to_float32(
        {name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()}
    )
    extra = {
        name: np.asarray(value)
        for name, value in mcmc.get_extra_fields(group_by_chain=True).items()
    }
    return NutsRun(
        posterior=posterior,
        extra=extra,
        timings=timings,
        device=jax.devices()[0].platform,
        settings={
            "warmup": num_warmup,
            "samples": nuts.samples,
            "chains": nuts.chains,
            "chain_method": nuts.chain_method,
            "target_accept": nuts.target_accept,
            "max_tree_depth": nuts.max_tree_depth,
            "warm_start": warm_start.source if warm_start else None,
            "dense_mass": bool(warm_start.dense) if warm_start else False,
        },
    )


def flatten_chains(posterior: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge the chain and draw dimensions for prediction code."""
    return {name: value.reshape(-1, *value.shape[2:]) for name, value in posterior.items()}
