"""Shared figure helpers.

Data are stored and modelled in UTC; every axis shown to a reader is AEST
(market time), so the display conversion lives here and nowhere else. Local
Sydney clock time appears only where the DST discussion needs it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISPLAY_TZ = "Australia/Brisbane"
LOCAL_TZ = "Australia/Sydney"

_PALETTE = {
    "demand": "#1f5673",
    "temperature": "#c44536",
    "irradiance": "#e8a13a",
    "forecast": "#5d8aa8",
    "accent": "#7a4988",
}


def setup_style() -> None:
    """Apply the project's matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (11, 4),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
        }
    )


def palette(name: str) -> str:
    """Colour for a named series family."""
    return _PALETTE[name]


# One canonical colour per model and fitting method, used consistently across
# every notebook. The full-rank ADVI fit is the headline BSTS, so "BSTS" and
# "BSTS-ADVI-FR" share a colour. "observed" is reserved (black) and is not a
# model.
MODEL_COLOURS = {
    "observed": "#000000",
    "seasonal naive": "#9a9a9a",
    "ARIMA": "#c44536",
    "LightGBM": "#2e7d32",
    "BART": "#7a4988",
    "BSTS": "#1f5673",
    "BSTS-ADVI-FR": "#1f5673",
    "BSTS-NUTS": "#e8a13a",
}


def model_colour(name: str) -> str:
    """Canonical colour for a model-and-fitting-method label."""
    return MODEL_COLOURS[name]


def model_shades(name: str, n: int) -> list[str]:
    """``n`` distinguishable shades of a model's colour, light to dark.

    For showing one model at several lead or issue times without leaving its
    colour family.
    """
    import matplotlib.colors as mcolors

    base = np.array(mcolors.to_rgb(MODEL_COLOURS[name]))
    factors = np.linspace(0.55, -0.4, n)  # > 0 lightens toward white, < 0 darkens
    shades = []
    for f in factors:
        rgb = base + (1.0 - base) * f if f >= 0 else base * (1.0 + f)
        shades.append(mcolors.to_hex(np.clip(rgb, 0.0, 1.0)))
    return shades


def display_index(index: pd.DatetimeIndex, tz: str = DISPLAY_TZ) -> pd.DatetimeIndex:
    """Convert a UTC index to the display timezone."""
    return index.tz_convert(tz)


def format_date_axis(ax: plt.Axes) -> None:
    """Concise date ticks for long spans."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def clock_profile(
    series: pd.Series,
    tz: str = DISPLAY_TZ,
    quantiles: tuple[float, float] = (0.1, 0.9),
) -> pd.DataFrame:
    """Median and quantile band of a series by half hour of the clock day.

    Parameters
    ----------
    series
        Half-hourly series on a UTC index.
    tz
        Clock used for the grouping: market time by default, local Sydney
        time for the DST comparison.
    quantiles
        Lower and upper band quantiles.

    Returns
    -------
    pandas.DataFrame
        Indexed by hour-of-day fraction, columns ``median``, ``lower`` and
        ``upper``.
    """
    clock = series.index.tz_convert(tz)
    hour = clock.hour + clock.minute / 60.0
    grouped = series.groupby(hour)
    return pd.DataFrame(
        {
            "median": grouped.median(),
            "lower": grouped.quantile(quantiles[0]),
            "upper": grouped.quantile(quantiles[1]),
        }
    )


def plot_clock_profile(
    ax: plt.Axes,
    series: pd.Series,
    tz: str,
    label: str,
    colour: str,
) -> None:
    """Draw a clock-day profile with its quantile band."""
    profile = clock_profile(series, tz=tz)
    ax.plot(profile.index, profile["median"], label=label, color=colour)
    ax.fill_between(profile.index, profile["lower"], profile["upper"], alpha=0.15, color=colour)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("hour of day")


def fan_chart(
    ax: plt.Axes,
    index: pd.DatetimeIndex,
    mean: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
    colour: str = _PALETTE["demand"],
    label: str | None = None,
) -> None:
    """Draw central predictive bands and the median over a horizon.

    Accepts either a Gaussian predictive (``mean`` and ``sd``) or predictive
    ``samples`` of shape ``(S, T)``; bands are exact quantiles either way.
    """
    from scipy import stats

    times = display_index(index)
    for level in sorted(levels, reverse=True):
        tail = (1.0 - level) / 2.0
        if samples is not None:
            lower = np.quantile(samples, tail, axis=0)
            upper = np.quantile(samples, 1.0 - tail, axis=0)
        else:
            z = stats.norm.ppf(1.0 - tail)
            lower, upper = mean - z * sd, mean + z * sd
        ax.fill_between(times, lower, upper, color=colour, alpha=0.18, lw=0)
    centre = np.median(samples, axis=0) if samples is not None else mean
    ax.plot(times, centre, color=colour, lw=1.4, label=label)
    format_date_axis(ax)


def horizon_curve(
    ax: plt.Axes,
    scores: np.ndarray,
    label: str,
    colour: str,
) -> None:
    """Mean score by forecast step, from per-origin scores ``(O, H)``."""
    steps = (np.arange(scores.shape[1]) + 1) / 2.0
    ax.plot(steps, scores.mean(axis=0), color=colour, label=label)
    ax.set_xlabel("lead time (hours)")


def confidence_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 2.0,
    **kwargs,
) -> None:
    """Draw an n-standard-deviation covariance ellipse for a 2D Gaussian.

    Parameters
    ----------
    ax
        Target axis.
    mean
        Centre, shape ``(2,)``.
    cov
        Covariance, shape ``(2, 2)``.
    n_std
        Radius in standard deviations; 2 covers about 86% of the mass.
    """
    from matplotlib.patches import Ellipse

    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(values)
    ax.add_patch(
        Ellipse(xy=tuple(mean), width=width, height=height, angle=angle, fill=False, **kwargs)
    )


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> Path:
    """Save a figure for the README, returning its path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path
