"""Posterior divergences for adjudicating one inference against another.

The variational guides are Gaussian in the model's unconstrained coordinates,
so a surrogate is compared with its NUTS reference there: the guide then
carries no approximation error of its own and only the reference is summarised
by its first two moments. Working in the constrained space would approximate
both sides as Gaussian and inflate the divergence with that double error.

Two estimators of the Kullback-Leibler divergence are provided. The closed
form for two Gaussians is the workhorse, justified by near-Gaussian marginals
(small skew and excess kurtosis). A nearest-neighbour estimator that assumes
nothing about the shape is kept for a low-dimensional cross-check, so the
Gaussian assumption is validated rather than asserted. A finite-sample floor,
the divergence between disjoint draws of the same posterior, sets the scale
below which a measured gap is estimation noise. All divergences are in nats.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def to_unconstrained(values: np.ndarray, support: str | tuple) -> np.ndarray:
    """Map constrained draws to the unconstrained line via the model's bijection.

    Mirrors the transforms NumPyro applies internally: the identity on the
    real line, a log for positive sites, a logit for the unit interval and a
    rescaled logit for a bounded interval.

    Parameters
    ----------
    values
        Constrained draws, any shape.
    support
        ``"real"``, ``"positive"``, ``"unit_interval"`` or the tuple
        ``("interval", lo, hi)``.
    """
    values = np.asarray(values, dtype=np.float64)
    if support == "real":
        return values
    if support == "positive":
        return np.log(values)
    if support == "unit_interval":
        return np.log(values) - np.log1p(-values)
    if isinstance(support, tuple) and support[0] == "interval":
        _, lo, hi = support
        scaled = (values - lo) / (hi - lo)
        return np.log(scaled) - np.log1p(-scaled)
    raise ValueError(f"unknown support {support!r}")


def stack_unconstrained(
    draws: dict[str, np.ndarray], supports: dict[str, str | tuple]
) -> tuple[np.ndarray, list[str]]:
    """Stack named draws into one unconstrained design matrix.

    Parameters
    ----------
    draws
        Mapping from site name to draws of shape ``(N,)`` or ``(N, K)``;
        chains must already be flattened into the leading axis.
    supports
        Mapping from site name to its support; iteration order fixes the
        column order of the result.

    Returns
    -------
    tuple
        The matrix of shape ``(N, D)`` and the list of ``D`` column names,
        vector sites expanded as ``site[k]``.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    for site, support in supports.items():
        block = to_unconstrained(draws[site], support)
        if block.ndim == 1:
            columns.append(block[:, None])
            names.append(site)
        else:
            columns.append(block)
            names.extend(f"{site}[{k}]" for k in range(block.shape[1]))
    return np.concatenate(columns, axis=1), names


def gaussian_kl_1d(
    mean_p: np.ndarray, sd_p: np.ndarray, mean_q: np.ndarray, sd_q: np.ndarray
) -> np.ndarray:
    """KL(P || Q) between univariate Gaussians, vectorised over the inputs."""
    return np.log(sd_q / sd_p) + (sd_p**2 + (mean_p - mean_q) ** 2) / (2.0 * sd_q**2) - 0.5


def gaussian_kl(
    mean_p: np.ndarray,
    cov_p: np.ndarray,
    mean_q: np.ndarray,
    cov_q: np.ndarray,
    ridge: float = 0.0,
) -> float:
    """KL(P || Q) between multivariate Gaussians.

    Parameters
    ----------
    mean_p, cov_p
        Mean and covariance of P, the distribution sampled from (the surrogate
        in the surrogate-against-reference reading).
    mean_q, cov_q
        Mean and covariance of Q, the reference.
    ridge
        Optional jitter added to both diagonals before the solves, for the
        rare ill-conditioned covariance.
    """
    mean_p = np.asarray(mean_p, dtype=np.float64)
    mean_q = np.asarray(mean_q, dtype=np.float64)
    cov_p = np.asarray(cov_p, dtype=np.float64)
    cov_q = np.asarray(cov_q, dtype=np.float64)
    dim = mean_p.shape[0]
    if ridge:
        eye = np.eye(dim)
        cov_p = cov_p + ridge * eye
        cov_q = cov_q + ridge * eye
    diff = mean_q - mean_p
    trace_term = np.trace(np.linalg.solve(cov_q, cov_p))
    maha = diff @ np.linalg.solve(cov_q, diff)
    _, logdet_q = np.linalg.slogdet(cov_q)
    _, logdet_p = np.linalg.slogdet(cov_p)
    return float(0.5 * (logdet_q - logdet_p - dim + trace_term + maha))


def marginal_kl(samples_p: np.ndarray, samples_q: np.ndarray) -> np.ndarray:
    """Per-coordinate Gaussian KL(P || Q) from samples, ignoring correlation.

    The sum over coordinates is the divergence of the best independent
    approximation, a lower bound on the joint divergence that isolates which
    marginals the surrogate misses.
    """
    samples_p = np.asarray(samples_p, dtype=np.float64)
    samples_q = np.asarray(samples_q, dtype=np.float64)
    return gaussian_kl_1d(
        samples_p.mean(axis=0),
        samples_p.std(axis=0, ddof=1),
        samples_q.mean(axis=0),
        samples_q.std(axis=0, ddof=1),
    )


def joint_gaussian_kl(samples_p: np.ndarray, samples_q: np.ndarray, ridge: float = 1e-8) -> float:
    """Full-covariance Gaussian KL(P || Q) from samples."""
    samples_p = np.asarray(samples_p, dtype=np.float64)
    samples_q = np.asarray(samples_q, dtype=np.float64)
    return gaussian_kl(
        samples_p.mean(axis=0),
        np.cov(samples_p, rowvar=False),
        samples_q.mean(axis=0),
        np.cov(samples_q, rowvar=False),
        ridge=ridge,
    )


def knn_kl(samples_p: np.ndarray, samples_q: np.ndarray, k: int = 5) -> float:
    """Nearest-neighbour estimator of KL(P || Q) after Wang, Kulkarni, Verdu.

    Assumes nothing about the shape of either distribution, so agreement with
    the Gaussian closed form in low dimension certifies the Gaussian reading.
    Unreliable in high dimension, where the Gaussian form is used instead.
    """
    p = np.atleast_2d(np.asarray(samples_p, dtype=np.float64))
    q = np.atleast_2d(np.asarray(samples_q, dtype=np.float64))
    n, dim = p.shape
    m = q.shape[0]
    if n < k + 1 or m < k:
        raise ValueError("too few samples for the requested neighbour count")
    rho = cKDTree(p).query(p, k + 1)[0][:, k]  # k-th neighbour within P, self excluded
    nu = cKDTree(q).query(p, k)[0][:, k - 1]  # k-th neighbour in Q
    tiny = np.finfo(np.float64).tiny
    return float(dim * np.mean(np.log((nu + tiny) / (rho + tiny))) + np.log(m / (n - 1)))


def gaussian_kl_noise_floor(
    samples: np.ndarray,
    n_p: int,
    n_q: int,
    reps: int = 20,
    seed: int = 0,
    ridge: float = 1e-8,
) -> dict[str, float]:
    """Finite-sample floor: joint Gaussian KL between disjoint draws of one set.

    Two disjoint subsamples of sizes ``n_p`` and ``n_q`` come from the same
    distribution, so their measured divergence is pure estimation noise. A
    measured surrogate gap below this floor is not distinguishable from the
    reference at these sample sizes.

    Returns
    -------
    dict
        ``mean`` and ``std`` of the floor over ``reps`` random splits.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if n_p + n_q > samples.shape[0]:
        raise ValueError("n_p + n_q exceeds the number of available draws")
    values = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        order = rng.permutation(samples.shape[0])
        part_p = samples[order[:n_p]]
        part_q = samples[order[n_p : n_p + n_q]]
        values.append(joint_gaussian_kl(part_p, part_q, ridge=ridge))
    values = np.asarray(values)
    return {"mean": float(values.mean()), "std": float(values.std())}
