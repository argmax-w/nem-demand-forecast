"""Scoring rules for point and probabilistic forecasts.

The continuous ranked probability score (CRPS) is the primary metric. Two
estimators are provided: the analytic closed form for a Gaussian predictive,
used by the classical baseline, and the sample-based energy form

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|,    X, X' ~ F,

used for the Bayesian posterior predictives and computed streaming over
chunks of draws so device and host memory stay bounded. The two are
unit-tested against each other in the Gaussian limit.

All scoring functions are negatively oriented (smaller is better) and fully
vectorised over forecast steps.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def crps_gaussian(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of a Gaussian predictive.

    Parameters
    ----------
    y
        Observations, any broadcastable shape.
    mean, sd
        Predictive parameters, broadcastable against ``y``.

    Returns
    -------
    numpy.ndarray
        CRPS per observation, in the units of ``y``.
    """
    z = (np.asarray(y) - mean) / sd
    pit_term = z * (2.0 * stats.norm.cdf(z) - 1.0)
    return sd * (pit_term + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def crps_samples(y: np.ndarray, draws: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Energy-form CRPS from predictive samples, streaming over draws.

    The pairwise term is accumulated chunk against chunk, so peak memory is
    O(chunk * chunk + chunk * T) rather than O(S * S) for S draws. Draws are
    promoted to float64 for the accumulation only.

    Parameters
    ----------
    y
        Observations, shape ``(T,)``.
    draws
        Predictive samples, shape ``(S, T)``.
    chunk
        Draws per block.

    Returns
    -------
    numpy.ndarray
        CRPS per step, shape ``(T,)``.
    """
    y = np.asarray(y, dtype=np.float64)
    n_draws = draws.shape[0]

    abs_error = np.zeros_like(y)
    pairwise = np.zeros_like(y)
    for i in range(0, n_draws, chunk):
        block = np.asarray(draws[i : i + chunk], dtype=np.float64)
        abs_error += np.abs(block - y).sum(axis=0)
        for j in range(0, n_draws, chunk):
            other = np.asarray(draws[j : j + chunk], dtype=np.float64)
            pairwise += np.abs(block[:, None, :] - other[None, :, :]).sum(axis=(0, 1))
    return abs_error / n_draws - 0.5 * pairwise / (n_draws * n_draws)


def energy_score(y: np.ndarray, draws: np.ndarray, chunk: int = 256) -> float:
    """Energy score of the joint predictive path.

    The multivariate generalisation of CRPS over the whole horizon,

        ES(F, y) = E||X - y|| - 0.5 E||X - X'||,

    sensitive to the temporal coherence of sampled paths where per-step CRPS
    is not.

    Parameters
    ----------
    y
        Observed path, shape ``(T,)``.
    draws
        Sampled paths, shape ``(S, T)``.
    chunk
        Draws per block.

    Returns
    -------
    float
        The energy score for this path.
    """
    y = np.asarray(y, dtype=np.float64)
    n_draws = draws.shape[0]

    error_term = 0.0
    pair_term = 0.0
    for i in range(0, n_draws, chunk):
        block = np.asarray(draws[i : i + chunk], dtype=np.float64)
        error_term += np.linalg.norm(block - y, axis=1).sum()
        for j in range(0, n_draws, chunk):
            other = np.asarray(draws[j : j + chunk], dtype=np.float64)
            pair_term += np.linalg.norm(block[:, None, :] - other[None, :, :], axis=2).sum()
    return error_term / n_draws - 0.5 * pair_term / (n_draws * n_draws)


def crps_from_quantiles(
    y: np.ndarray, quantile_forecasts: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """CRPS approximated from a grid of predictive quantiles.

    Uses the identity CRPS = 2 integral of the pinball loss over quantile
    levels, evaluated by the trapezoidal rule with the open tails closed at
    levels 0 and 1 (where the pinball loss of any finite forecast tends to
    zero). Used for quantile-only forecasters such as the gradient-boosted
    benchmark; accuracy against the analytic Gaussian form is unit-tested.

    Parameters
    ----------
    y
        Observations, shape ``(T,)``.
    quantile_forecasts
        Forecast quantiles, shape ``(Q, T)``, non-crossing.
    quantiles
        The quantile levels, shape ``(Q,)``, increasing.

    Returns
    -------
    numpy.ndarray
        Approximate CRPS per step, shape ``(T,)``.
    """
    y = np.asarray(y)[None, :]
    q = np.asarray(quantiles)[:, None]
    diff = y - np.asarray(quantile_forecasts)
    pinball = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    levels = np.concatenate([[0.0], np.asarray(quantiles), [1.0]])
    curve = np.vstack([np.zeros(y.shape[1]), pinball, np.zeros(y.shape[1])])
    return 2.0 * np.trapezoid(curve, levels, axis=0)


def log_score_gaussian(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Negative log predictive density of a Gaussian predictive."""
    return -stats.norm.logpdf(np.asarray(y), loc=mean, scale=sd)


def log_score_samples(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Negative log predictive density from samples via a mixture fit.

    Each step's predictive is summarised by a Gaussian matched to the draw
    mean and variance. For near-Gaussian posterior predictives this is a
    stable, low-variance estimate; a kernel density over a thousand draws
    would add noise without changing conclusions. The approximation is named
    in the notebooks wherever the score is reported.
    """
    mean = draws.mean(axis=0)
    sd = draws.std(axis=0, ddof=1)
    return log_score_gaussian(y, mean, sd)


def pinball_loss(
    y: np.ndarray, quantile_forecasts: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Pinball loss per quantile, averaged over steps.

    Parameters
    ----------
    y
        Observations, shape ``(T,)``.
    quantile_forecasts
        Forecast quantiles, shape ``(Q, T)``.
    quantiles
        The quantile levels, shape ``(Q,)``.

    Returns
    -------
    numpy.ndarray
        Mean loss per quantile level, shape ``(Q,)``.
    """
    y = np.asarray(y)[None, :]
    q = np.asarray(quantiles)[:, None]
    diff = y - np.asarray(quantile_forecasts)
    return np.where(diff >= 0, q * diff, (q - 1.0) * diff).mean(axis=1)


def mae(y: np.ndarray, point: np.ndarray) -> float:
    """Mean absolute error of a point forecast."""
    return float(np.mean(np.abs(np.asarray(y) - point)))


def rmse(y: np.ndarray, point: np.ndarray) -> float:
    """Root mean squared error of a point forecast."""
    return float(np.sqrt(np.mean((np.asarray(y) - point) ** 2)))


def mase(y: np.ndarray, point: np.ndarray, naive_abs_error: float) -> float:
    """Mean absolute scaled error.

    Parameters
    ----------
    y, point
        Observations and point forecasts.
    naive_abs_error
        Mean absolute error of the seasonal-naive benchmark on the same
        targets, the conventional scaling base.
    """
    return mae(y, point) / naive_abs_error


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a central predictive interval."""
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))


def paired_bootstrap_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap for a mean score difference between two models.

    Scores are paired by forecast origin (one mean score per origin), so
    resampling origins respects the strong within-day correlation of errors,
    in the spirit of a Diebold-Mariano comparison.

    Parameters
    ----------
    scores_a, scores_b
        Per-origin mean scores for the two models, shape ``(N,)``.
    n_boot
        Bootstrap resamples.
    seed
        Seed for the resampling.

    Returns
    -------
    dict
        Mean difference (a minus b), the central 95% interval and the
        two-sided p-value for a zero difference.
    """
    diff = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[indices].mean(axis=1)
    p_value = 2.0 * min(np.mean(means > 0.0), np.mean(means < 0.0))
    return {
        "difference": float(diff.mean()),
        "lower95": float(np.quantile(means, 0.025)),
        "upper95": float(np.quantile(means, 0.975)),
        "p_value": float(min(p_value, 1.0)),
    }
