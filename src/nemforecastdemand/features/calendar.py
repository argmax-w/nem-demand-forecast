"""Calendar features on the UTC grid.

Demand follows the local clock, not the market clock: NSW observes daylight
saving while NEM market time is fixed AEST, so in UTC (or market time) the
daily shape shifts by an hour twice a year. All phases here are therefore
computed from local Sydney clock time, which keeps the seasonal design stable
across DST transitions while the data remain on a regular UTC grid. Notebook
01 demonstrates the shift and its resolution.

Two seasonal basis families are provided over the same phases: trigonometric
(Fourier) harmonics and periodic Gaussian radial basis functions. The RBF
basis trades the harmonics' global smoothness for localised bumps, which can
track sharp features such as the morning ramp with fewer effective degrees of
freedom; the two are compared on the validation set in notebook 02.
"""

from __future__ import annotations

import holidays as holidays_lib
import numpy as np
import pandas as pd

from nemforecastdemand.config import FeatureConfig

LOCAL_TZ = "Australia/Sydney"


def local_phases(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Daily and weekly phase in [0, 1) from the local Sydney clock.

    Parameters
    ----------
    index
        UTC timestamps.

    Returns
    -------
    tuple of numpy.ndarray
        ``(daily, weekly)`` phases, where 0 is local midnight and the week
        starts at local Monday midnight.
    """
    local = index.tz_convert(LOCAL_TZ)
    daily = (local.hour.to_numpy() + local.minute.to_numpy() / 60.0) / 24.0
    weekly = (local.dayofweek.to_numpy() + daily) / 7.0
    return daily, weekly


def fourier_design(phase: np.ndarray, harmonics: int, prefix: str) -> pd.DataFrame:
    """Fourier basis over a unit-period phase.

    Parameters
    ----------
    phase
        Phase values in [0, 1).
    harmonics
        Number of harmonics K, giving 2K columns.
    prefix
        Column name prefix.

    Returns
    -------
    pandas.DataFrame
        Columns ``{prefix}_sin{k}`` and ``{prefix}_cos{k}`` for k = 1..K.
    """
    k = np.arange(1, harmonics + 1)
    angles = 2.0 * np.pi * np.outer(phase, k)
    columns = {}
    for i, order in enumerate(k):
        columns[f"{prefix}_sin{order}"] = np.sin(angles[:, i])
        columns[f"{prefix}_cos{order}"] = np.cos(angles[:, i])
    return pd.DataFrame(columns)


def periodic_rbf_design(phase: np.ndarray, centres: int, prefix: str) -> pd.DataFrame:
    """Periodic Gaussian RBF basis over a unit-period phase.

    Centres are equally spaced on the circle and the bandwidth equals the
    centre spacing, a standard choice that overlaps neighbouring bumps
    enough for smooth interpolation. The final centre is dropped because the
    bumps sum to a near-constant function, which would be confounded with
    the model's level.

    Parameters
    ----------
    phase
        Phase values in [0, 1).
    centres
        Number of centres before the identifiability drop.
    prefix
        Column name prefix.

    Returns
    -------
    pandas.DataFrame
        Columns ``{prefix}_rbf{j}`` for j = 0..centres - 2.
    """
    grid = np.arange(centres) / centres
    distance = np.abs(phase[:, None] - grid[None, :])
    distance = np.minimum(distance, 1.0 - distance)
    bandwidth = 1.0 / centres
    basis = np.exp(-0.5 * (distance / bandwidth) ** 2)
    return pd.DataFrame(
        {f"{prefix}_rbf{j}": basis[:, j] for j in range(centres - 1)},
    )


def seasonal_design(index: pd.DatetimeIndex, features: FeatureConfig) -> pd.DataFrame:
    """Build the daily and weekly seasonal design for the configured basis.

    Parameters
    ----------
    index
        UTC timestamps.
    features
        Basis family and size settings.

    Returns
    -------
    pandas.DataFrame
        Seasonal regressors indexed like ``index``.
    """
    daily, weekly = local_phases(index)
    if features.seasonal_basis == "fourier":
        blocks = [
            fourier_design(daily, features.daily_harmonics, "daily"),
            fourier_design(weekly, features.weekly_harmonics, "weekly"),
        ]
    elif features.seasonal_basis == "rbf":
        blocks = [
            periodic_rbf_design(daily, features.daily_rbf_centres, "daily"),
            periodic_rbf_design(weekly, features.weekly_rbf_centres, "weekly"),
        ]
    else:
        raise ValueError(f"unknown seasonal basis {features.seasonal_basis!r}")
    design = pd.concat(blocks, axis=1)
    design.index = index
    return design


def holiday_flag(index: pd.DatetimeIndex) -> pd.Series:
    """NSW public holiday indicator, evaluated on local dates.

    Parameters
    ----------
    index
        UTC timestamps.

    Returns
    -------
    pandas.Series
        Boolean series named ``is_holiday`` indexed like ``index``.
    """
    local = index.tz_convert(LOCAL_TZ)
    years = range(local[0].year, local[-1].year + 1) if len(local) else []
    nsw = holidays_lib.country_holidays("AU", subdiv="NSW", years=years)
    dates = pd.Series(local.date, index=index)
    return dates.isin(set(nsw.keys())).rename("is_holiday")
