"""Inference diagnostics: ELBO decomposition and MCMC health.

The ELBO splits as ELBO = energy + entropy, where the energy is the expected
log joint under the surrogate and the entropy is the surrogate's own. The
split separates "the surrogate found mass under the model" from "the
surrogate stayed spread out", so a surrogate that under-estimates its
variance shows it here: entropy falls away while the energy keeps rising.

MCMC health wraps ArviZ: split R-hat, bulk and tail ESS, divergence counts,
E-BFMI and tree-depth saturation, summarised per parameter block so a five
thousand dimensional latent state does not drown the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd


@dataclass
class ElboTrace:
    """Logged ELBO decomposition over optimisation steps."""

    steps: np.ndarray
    elbo: np.ndarray
    energy: np.ndarray
    entropy: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Tabulate the trace for plotting."""
        return pd.DataFrame(
            {
                "step": self.steps,
                "elbo": self.elbo,
                "energy": self.energy,
                "entropy": self.entropy,
            }
        ).set_index("step")

    def converged(self, window: int = 20, tolerance: float = 1e-2) -> bool:
        """Crude convergence check: relative ELBO drift over the last window.

        The checkpoints are 64-particle estimates, so their noise floor sits
        well above fractions of a percent; one percent drift between the
        last two windows is the practical plateau for this estimator.
        """
        if len(self.elbo) < 2 * window:
            return False
        recent = self.elbo[-window:].mean()
        previous = self.elbo[-2 * window : -window].mean()
        return abs(recent - previous) / (abs(previous) + 1e-12) < tolerance


def mcmc_summary(posterior: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-parameter convergence summary from raw chain draws.

    Parameters
    ----------
    posterior
        Mapping from site name to draws of shape ``(chains, draws, ...)``.

    Returns
    -------
    pandas.DataFrame
        Worst-case split R-hat and smallest bulk and tail ESS per site, with
        trailing dimensions reduced, so vector sites report their weakest
        element.
    """
    idata = az.from_dict({"posterior": posterior})
    rhat = az.rhat(idata)
    bulk = az.ess(idata, method="bulk")
    tail = az.ess(idata, method="tail")
    rows = []
    for name in posterior:
        rows.append(
            {
                "site": name,
                "size": int(np.prod(posterior[name].shape[2:], initial=1)),
                "max_rhat": float(rhat[name].max()),
                "min_bulk_ess": float(bulk[name].min()),
                "min_tail_ess": float(tail[name].min()),
            }
        )
    return pd.DataFrame(rows).set_index("site").sort_values("max_rhat", ascending=False)


def sampler_health(extra_fields: dict[str, np.ndarray], max_tree_depth: int) -> pd.DataFrame:
    """Sampler-level health per chain.

    Parameters
    ----------
    extra_fields
        NUTS extra fields with leading chain dimension: ``diverging``
        (bool), ``energy`` and ``num_steps``.
    max_tree_depth
        The configured tree-depth cap; a step count of ``2**depth - 1``
        means the tree saturated.

    Returns
    -------
    pandas.DataFrame
        Divergence counts, E-BFMI and tree-depth saturation rate per chain.
    """
    diverging = np.asarray(extra_fields["diverging"])
    energy = np.asarray(extra_fields["energy"], dtype=np.float64)
    num_steps = np.asarray(extra_fields["num_steps"])

    energy_diff = np.diff(energy, axis=-1)
    ebfmi = energy_diff.var(axis=-1) / energy.var(axis=-1)
    saturated = num_steps >= 2**max_tree_depth - 1
    return pd.DataFrame(
        {
            "divergences": diverging.sum(axis=-1).astype(int),
            "e_bfmi": ebfmi,
            "tree_depth_saturation": saturated.mean(axis=-1),
        },
        index=pd.RangeIndex(diverging.shape[0], name="chain"),
    )


def time_to_target_ess(
    warmup_seconds: float,
    sampling_seconds: float,
    achieved_bulk_ess: float,
    target_bulk_ess: float,
) -> float:
    """Wall-clock to reach a target bulk ESS, extrapolating sampling time.

    ESS accrues roughly linearly in post-warmup draws, so sampling time is
    scaled by the ESS shortfall while warmup is a fixed cost. This is the
    quality-adjusted quantity used for the cold-versus-warm comparison;
    quoting raw wall-clock would reward a warmup too short to mix.
    """
    if achieved_bulk_ess <= 0:
        return float("inf")
    return warmup_seconds + sampling_seconds * target_bulk_ess / achieved_bulk_ess
