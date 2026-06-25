"""Calibration diagnostics: PIT and reliability of central intervals.

A probabilistic forecast is calibrated when the observation looks like a
draw from the predictive. The probability integral transform (PIT) makes
that testable: PIT values from a calibrated forecast are uniform on [0, 1].
A hump in the PIT histogram means the predictive is too wide, a U shape too
narrow and a tilt means bias.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def pit_gaussian(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """PIT values under a Gaussian predictive."""
    return stats.norm.cdf((np.asarray(y) - mean) / sd)


def pit_samples(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Randomised PIT from predictive samples.

    The empirical CDF of a finite sample puts mass on its atoms, so the PIT
    is randomised uniformly within each atom, the standard treatment that
    restores exact uniformity under calibration.

    Parameters
    ----------
    y
        Observations, shape ``(T,)``.
    draws
        Predictive samples, shape ``(S, T)``.

    Returns
    -------
    numpy.ndarray
        PIT values in [0, 1], shape ``(T,)``.
    """
    y = np.asarray(y)
    below = (draws < y[None, :]).mean(axis=0)
    at = (draws == y[None, :]).mean(axis=0)
    rng = np.random.default_rng(0)
    return below + rng.uniform(size=y.shape) * (at + 1.0 / draws.shape[0])


def pit_histogram(pit: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of PIT values normalised to density 1 under uniformity."""
    counts, edges = np.histogram(np.clip(pit, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    density = counts / counts.sum() * bins
    return density, edges


def rb_pit(
    y: np.ndarray, draw_mean: np.ndarray, draw_sd: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    """Rao-Blackwellised PIT: the per-draw-averaged Gaussian CDF at each ``y``.

    For a per-draw Gaussian predictive the calibrated PIT is the averaged
    mixture CDF ``u_i = mean_j Phi((y_i - mu_ij) / sigma_ij)``, integrating
    out the observation noise analytically rather than ranking sampled
    replicates. Lower variance than :func:`pit_samples` and free of its
    randomisation.

    Parameters
    ----------
    y
        Observations, shape ``(N,)``.
    draw_mean, draw_sd
        Per-draw predictive mean and standard deviation, shape ``(S, N)``.
    chunk
        Observations processed per block, to bound the ``(S, chunk)``
        temporary.

    Returns
    -------
    numpy.ndarray
        PIT values in [0, 1], shape ``(N,)``.
    """
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(draw_mean, dtype=np.float64)
    sd = np.asarray(draw_sd, dtype=np.float64)
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        z = (y[None, sl] - mean[:, sl]) / sd[:, sl]
        out[sl] = stats.norm.cdf(z).mean(axis=0)
    return out


def averaged_gaussian_pdf(
    y_grid: np.ndarray, draw_mean: np.ndarray, draw_sd: np.ndarray, chunk: int = 128
) -> np.ndarray:
    """Per-draw-averaged Gaussian-mixture density on a grid.

    The Rao-Blackwellised predictive density ``f_t(y) = mean_j N(y; mu_tj,
    sigma_tj)`` for every series step ``t``, evaluated on a shared ``y_grid``
    and streamed over draws.

    Parameters
    ----------
    y_grid
        Evaluation points, shape ``(G,)``.
    draw_mean, draw_sd
        Per-draw moments, shape ``(S, T)``.
    chunk
        Draws per block.

    Returns
    -------
    numpy.ndarray
        Density field, shape ``(T, G)``.
    """
    mean = np.asarray(draw_mean, dtype=np.float64)
    sd = np.asarray(draw_sd, dtype=np.float64)
    n_draws, n_steps = mean.shape
    acc = np.zeros((n_steps, y_grid.shape[0]), dtype=np.float64)
    for i in range(0, n_draws, chunk):
        mu = mean[i : i + chunk][:, :, None]
        sg = sd[i : i + chunk][:, :, None]
        z = (y_grid[None, None, :] - mu) / sg
        acc += (np.exp(-0.5 * z * z) / (sg * np.sqrt(2.0 * np.pi))).sum(axis=0)
    return acc / n_draws


def averaged_gaussian_cdf(
    y_grid: np.ndarray, draw_mean: np.ndarray, draw_sd: np.ndarray, chunk: int = 128
) -> np.ndarray:
    """Per-draw-averaged Gaussian-mixture CDF on a grid, shape ``(T, G)``."""
    mean = np.asarray(draw_mean, dtype=np.float64)
    sd = np.asarray(draw_sd, dtype=np.float64)
    n_draws, n_steps = mean.shape
    acc = np.zeros((n_steps, y_grid.shape[0]), dtype=np.float64)
    for i in range(0, n_draws, chunk):
        mu = mean[i : i + chunk][:, :, None]
        sg = sd[i : i + chunk][:, :, None]
        acc += stats.norm.cdf((y_grid[None, None, :] - mu) / sg).sum(axis=0)
    return acc / n_draws


def predictive_band_from_moments(
    probs: np.ndarray,
    draw_mean: np.ndarray,
    draw_sd: np.ndarray,
    n_grid: int = 512,
    pad: float = 5.0,
) -> np.ndarray:
    """Predictive quantiles by inverting the per-draw-averaged CDF.

    The standing rule's route to a band: invert the averaged mixture CDF by
    interpolation on a ``y_grid`` rather than taking empirical quantiles of
    sampled replicates.

    Parameters
    ----------
    probs
        Quantile levels, shape ``(P,)``.
    draw_mean, draw_sd
        Per-draw moments, shape ``(S, T)``.
    n_grid
        Resolution of the grid the averaged CDF is inverted on.
    pad
        Grid half-width in draw standard deviations beyond the extreme means.

    Returns
    -------
    numpy.ndarray
        Quantiles, shape ``(P, T)``.
    """
    mean = np.asarray(draw_mean, dtype=np.float64)
    sd = np.asarray(draw_sd, dtype=np.float64)
    lo = float((mean - pad * sd).min())
    hi = float((mean + pad * sd).max())
    y_grid = np.linspace(lo, hi, n_grid)
    cdf = averaged_gaussian_cdf(y_grid, mean, sd)
    out = np.empty((len(probs), mean.shape[1]), dtype=np.float64)
    for t in range(mean.shape[1]):
        out[:, t] = np.interp(probs, cdf[t], y_grid)
    return out


def pit_ecdf_difference(pit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The PIT empirical CDF minus the uniform reference.

    Returns the sorted PIT values and ``F_n(u) - u``, the difference plotted
    against the simultaneous band. A flat line at zero is perfect calibration.
    A curve that climbs above zero on the left before falling is over-confident
    (the predictive too narrow); the mirror image, a dip then a rise, is
    over-dispersed (too wide).
    """
    x = np.sort(np.clip(np.asarray(pit, dtype=np.float64), 0.0, 1.0))
    n = x.size
    ecdf = np.arange(1, n + 1, dtype=np.float64) / n
    return x, ecdf - x


def ecdf_uniform_band(n: int, alpha: float = 0.05) -> float:
    """Half-width of the simultaneous 1-alpha band for a uniform ECDF.

    The two-sided Kolmogorov-Smirnov critical value at sample size ``n``: a
    calibrated PIT keeps its whole ECDF-difference curve inside ``+/- band``
    with probability ``1 - alpha``, so the band controls family-wise error
    across the entire curve rather than point by point.
    """
    return float(stats.kstwo.ppf(1.0 - alpha, n))


def stratified_pit(pit: np.ndarray, strata: np.ndarray) -> dict[object, np.ndarray]:
    """Split PIT values by stratum label, preserving label order of appearance.

    Marginal uniformity can hide conditional miscalibration where opposing
    errors in different regions cancel, so the calibration is re-read within
    each stratum (a lead-time band or an hour-of-day band).
    """
    pit = np.asarray(pit)
    strata = np.asarray(strata)
    labels = list(dict.fromkeys(strata.tolist()))
    return {label: pit[strata == label] for label in labels}


def reliability_table(
    y: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Observed frequency below each forecast quantile.

    For a calibrated forecast the observed frequency matches the nominal
    level; the gap is the reliability error plotted in the notebooks.
    """
    y = np.asarray(y)[None, :]
    return (y <= np.asarray(quantile_forecasts)).mean(axis=1)
