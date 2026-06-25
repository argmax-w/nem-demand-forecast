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
from matplotlib import patheffects

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


# --- Appendix B conventions -------------------------------------------------
# Each visual role rides a fixed channel so role, parameter identity, chain
# identity and diagnostic status never collide. Black is reserved for observed
# data; the predictive density field is a warm map for the posterior and a cool
# map for the prior; diagnostic status is its own green/amber/red axis.

# Categorical palette for parameter identity (Okabe-Ito), drawn from hues
# outside the predictive-band families so a parameter never reads as a ribbon.
PARAM_COLOURS = [
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#999999",
    "#E69F00",
    "#D55E00",
    "#F0E442",
    "#CC79A7",
]
# Fixed chain palette for per-chain plots (trace, rank); orthogonal to the rest.
CHAIN_COLOURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

DATA_COLOUR = "#000000"  # observed data only
DATA_EDGE = "#ffffff"  # marker/line casing in the background colour
DATA_EDGE_WIDTH = 1.6  # points; reads over the darkest part of a ribbon

# Prior vs posterior of the same parameter: same colour, prior dashed and faded,
# posterior solid and full, so contraction reads from linestyle not hue.
PRIOR_STYLE = dict(linestyle="--", alpha=0.55, linewidth=1.4)
POSTERIOR_STYLE = dict(linestyle="-", alpha=1.0, linewidth=1.8)

# Density-field maps: warm for the posterior predictive, cool for the prior.
POSTERIOR_CMAP = "OrRd"
PRIOR_CMAP = "Purples"
RIBBON_ALPHA = 0.85  # below 1 so the paper lightens the peak and data stay legible
CI_LINE_STYLE = ":"  # dotted central-interval lines, distinct from solid/dashed

# Diagnostic traffic lights: a separate reserved axis, never reused above.
STATUS_COLOURS = {"green": "#1a9850", "amber": "#fee08b", "red": "#d73027"}


def sequential_cmap(base_colour: str, name: str = "seq"):
    """A white-to-``base_colour`` sequential colormap for a density field.

    Each model's predictive density rides its own colour, so the heat map
    carries model identity while low density fades to the white background and
    the peak reaches the deep base colour, as Appendix B asks of the ribbon.
    """
    import matplotlib.colors as mcolors

    base = mcolors.to_rgb(base_colour)
    mid = tuple(0.5 + 0.5 * c for c in base)  # a tinted midpoint so the ramp is not washed out
    return mcolors.LinearSegmentedColormap.from_list(name, [(1.0, 1.0, 1.0), mid, base])


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


def density_ribbon(
    ax: plt.Axes,
    index: pd.DatetimeIndex,
    *,
    draw_mean: np.ndarray | None = None,
    draw_sd: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    cmap=POSTERIOR_CMAP,
    line_colour=None,
    ci_levels: tuple[float, ...] = (0.5, 0.9),
    y_grid: np.ndarray | None = None,
    n_grid: int = 220,
    median: bool = True,
    label: str | None = None,
    observed_label: str | None = "observed",
    density_label: str | None = "predictive density",
    interval_label: str | None = None,
) -> None:
    """Predictive density field as a heat map, with dotted central-interval lines.

    The Appendix B predictive ribbon: at each horizon step the predictive
    density is drawn as colour intensity, so multimodality, skew and tail
    thickness survive where a mean-and-band summary would discard them. The
    central intervals in ``ci_levels`` (50% and 90% by default) are overlaid as
    dotted lines, and any ``observed`` path in reserved black with a white
    casing. The field is normalised per step to its own peak, so every hour
    stays legible under the heteroskedastic scale; the dotted lines carry the
    absolute width.

    Provide exactly one predictive form:

    - ``draw_mean`` and ``draw_sd`` of shape ``(S, T)``: per-draw Gaussian
      moments, averaged analytically into the mixture density and inverted for
      the bands (the Rao-Blackwellised route, used for the BSTS);
    - ``mean`` and ``sd`` of shape ``(T,)``: a single Gaussian predictive (the
      analytic ARIMA);
    - ``samples`` of shape ``(S, T)``: predictive draws, with a per-step
      Gaussian kernel density for the field and empirical quantiles for the
      bands.

    ``index`` may be a ``DatetimeIndex`` (drawn against a market-time axis) or
    a plain numeric array (for horizon-step or hours-from-origin axes).
    """
    from nemforecastdemand.evaluation import calibration

    is_dates = isinstance(index, pd.DatetimeIndex)
    times = mdates.date2num(display_index(index)) if is_dates else np.asarray(index, dtype=float)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if line_colour is None:
        line_colour = cmap_obj(0.92)
    if interval_label is None:
        interval_label = " and ".join(f"{round(lv * 100)}" for lv in ci_levels) + "% interval"
    casing = [patheffects.withStroke(linewidth=2.0, foreground="white")]

    if draw_mean is not None:
        dm = np.asarray(draw_mean, dtype=np.float64)
        ds = np.asarray(draw_sd, dtype=np.float64)
    elif mean is not None:
        dm = np.asarray(mean, dtype=np.float64)[None, :]
        ds = np.asarray(sd, dtype=np.float64)[None, :]
    else:
        dm = ds = None

    if y_grid is None:
        if dm is not None:
            lo, hi = float((dm - 4.0 * ds).min()), float((dm + 4.0 * ds).max())
        else:
            lo, hi = float(np.min(samples)), float(np.max(samples))
        if observed is not None:
            lo, hi = min(lo, float(np.min(observed))), max(hi, float(np.max(observed)))
        pad = 0.04 * (hi - lo)
        y_grid = np.linspace(lo - pad, hi + pad, n_grid)

    if dm is not None:
        field = calibration.averaged_gaussian_pdf(y_grid, dm, ds)
    else:
        from scipy.stats import gaussian_kde

        field = np.stack([gaussian_kde(samples[:, t])(y_grid) for t in range(samples.shape[1])])
    peak = field.max(axis=1, keepdims=True)
    field = np.where(peak > 0.0, field / peak, 0.0)
    ax.pcolormesh(
        times,
        y_grid,
        field.T,
        cmap=cmap_obj,
        shading="gouraud",
        alpha=RIBBON_ALPHA,
        rasterized=True,
    )
    if density_label is not None:
        # an off-canvas swatch so the legend names the density field itself
        ax.plot(
            [], [], marker="s", ms=9, ls="", color=cmap_obj(0.6), alpha=0.9, label=density_label
        )

    for level in ci_levels:
        probs = np.array([0.5 - level / 2.0, 0.5 + level / 2.0])
        if dm is not None:
            band = calibration.predictive_band_from_moments(probs, dm, ds)
        else:
            band = np.quantile(samples, probs, axis=0)
        for row in band:
            ax.plot(times, row, ls=CI_LINE_STYLE, lw=1.1, color=line_colour, path_effects=casing)
    if interval_label is not None and len(ci_levels) > 0:
        # an off-canvas dotted line so the legend names what the dotted lines mean
        ax.plot([], [], color=line_colour, ls=CI_LINE_STYLE, lw=1.1, label=interval_label)

    if median:
        if dm is not None:
            centre = calibration.predictive_band_from_moments(np.array([0.5]), dm, ds)[0]
        else:
            centre = np.median(samples, axis=0)
        ax.plot(times, centre, lw=1.5, color=line_colour, path_effects=casing, label=label)

    if observed is not None:
        ax.plot(
            times,
            np.asarray(observed),
            color=DATA_COLOUR,
            lw=1.6,
            path_effects=[patheffects.withStroke(linewidth=3.2, foreground=DATA_EDGE)],
            label=observed_label,
        )
    if is_dates:
        format_date_axis(ax)


def status_table_figure(ax: plt.Axes, table: pd.DataFrame) -> None:
    """Render a diagnostic traffic-light table on an axis.

    ``table`` is indexed by diagnostic with ``value`` and ``status`` columns,
    as :func:`~nemforecastdemand.evaluation.diagnostics.diagnostic_status_table`
    returns. The status cell carries the reserved green/amber/red.
    """
    ax.axis("off")

    def fmt(name: str, value: float) -> str:
        if "ESS" in name or "divergences" in name:
            return f"{round(value)}"
        return f"{value:.3f}"

    cells = [[name, fmt(name, row.value), row.status.upper()] for name, row in table.iterrows()]
    tbl = ax.table(
        cellText=cells,
        colLabels=["diagnostic", "value", "status"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.set_text_props(weight="bold")
        elif c == 2:
            status = cells[r - 1][2].lower()
            cell.set_facecolor(STATUS_COLOURS[status])
            cell.get_text().set_color("black" if status == "amber" else "white")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f5f5f5")


def rank_plot(
    ax: plt.Axes,
    draws: np.ndarray,
    colours: list[str],
    bins: int = 20,
) -> None:
    """Rank histogram by chain for one parameter.

    Ranks are taken across the pooled draws and split back by chain; under
    convergence each chain's ranks are uniform, so departures from the flat
    reference line flag between-chain discrepancies more reliably than a trace.

    Parameters
    ----------
    draws
        Per-chain draws for one parameter, shape ``(chains, draws)``.
    colours
        One colour per chain.
    bins
        Rank-histogram bins.
    """
    from scipy import stats

    draws = np.asarray(draws)
    n_chains, n_per = draws.shape
    ranks = stats.rankdata(draws.reshape(-1)).reshape(draws.shape)
    edges = np.linspace(0.0, n_chains * n_per, bins + 1)
    for chain in range(n_chains):
        ax.hist(ranks[chain], bins=edges, histtype="step", lw=1.2, color=colours[chain])
    ax.axhline(n_per / bins, color="0.5", ls="--", lw=0.8)
    ax.set_yticks([])
    ax.set_xticks([])


def pass_fail_hband(
    ax: plt.Axes,
    band: float,
    label_pass: str | None = "inside 95% simultaneous band",
    label_fail: str | None = "outside (miscalibrated)",
) -> None:
    """Shade a horizontal +/- band: inside pale green (pass), outside pale red (fail).

    For the PIT-ECDF-difference plots, where a calibrated curve stays within the
    simultaneous band. Call after the curves are drawn, so the y-limits are set.
    """
    lo, hi = ax.get_ylim()
    ax.axhspan(-band, band, color=STATUS_COLOURS["green"], alpha=0.10, zorder=0, label=label_pass)
    ax.axhspan(band, hi, color=STATUS_COLOURS["red"], alpha=0.07, zorder=0, label=label_fail)
    ax.axhspan(lo, -band, color=STATUS_COLOURS["red"], alpha=0.07, zorder=0)
    ax.set_ylim(lo, hi)


def pass_fail_diagonal(
    ax: plt.Axes,
    half_width: float,
    label_pass: str | None = "inside 95% simultaneous band",
    label_fail: str | None = "outside (miscalibrated)",
) -> None:
    """Shade a corridor of +/- ``half_width`` around the ``y = x`` diagonal on the
    unit square: inside pale green (pass), outside pale red (fail), for a coverage
    curve of empirical against nominal."""
    q = np.linspace(0.0, 1.0, 200)
    lower, upper = np.clip(q - half_width, 0.0, 1.0), np.clip(q + half_width, 0.0, 1.0)
    ax.fill_between(
        q, lower, upper, color=STATUS_COLOURS["green"], alpha=0.10, zorder=0, label=label_pass
    )
    ax.fill_between(
        q, upper, 1.0, color=STATUS_COLOURS["red"], alpha=0.07, zorder=0, label=label_fail
    )
    ax.fill_between(q, 0.0, lower, color=STATUS_COLOURS["red"], alpha=0.07, zorder=0)


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
