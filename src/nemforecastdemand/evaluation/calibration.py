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
