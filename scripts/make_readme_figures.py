"""Render the two headline figures the README opens on.

The first figure is the BSTS-NUTS reference posterior (the warm-started,
full-rank w300 run) shown on one test day as a calibrated fan against the
observed. The second contrasts two ways of drawing whole-day scenarios for
an example day: the BSTS coherent paths on the left, and on the right the only
thing LightGBM's per-step quantile heads allow, one independent draw per half
hour, which assumes the 48 timestamps are independent and comes out jagged.

Run after the fits, e.g. as the last line of ``scripts/run_all.sh``:

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemforecastdemand.config import load_config
from nemforecastdemand.evaluation.metrics import crps_samples
from nemforecastdemand.plotting import (
    DISPLAY_TZ,
    MODEL_COLOURS,
    density_ribbon,
    save_figure,
    sequential_cmap,
    setup_style,
)
from nemforecastdemand.utils import load_artifact

# The two days the README leads on, chosen by hand off the test set:
#  - the fan day is a high-demand winter Saturday the model forecasts tightly,
#  - the scenario day has the classic twin peaks, so the spread between coherent
#    draws is wide enough to read. It stands in for a generic example day: its
#    clock and features are given, its outcome is not, so no date is shown.
# Both start at 00:00 AEST, so the horizon is one clean calendar day.
FAN_ORIGIN = 20
SCENARIO_ORIGIN = 17
N_SCENARIOS = 30


def _clock(origin: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """AEST timestamps for the half-hourly steps after a UTC origin."""
    steps = origin + pd.to_timedelta((np.arange(horizon) + 1) * 30, unit="m")
    return steps.tz_convert(DISPLAY_TZ)


def _time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=DISPLAY_TZ))
    ax.set_xlabel("time of day (AEST)")
    ax.set_ylabel("demand (MW)")


def main() -> None:
    setup_style()
    cfg = load_config()
    ar, _ = load_artifact(cfg.paths.artifacts / "bsts_innovations_nuts_warm_fullrank_w300")
    paths, y = ar["forecast_paths"], ar["y_test"]  # (S, O, H), (O, H)
    origins = pd.DatetimeIndex(ar["origins_test"].astype("datetime64[us]")).tz_localize("UTC")
    horizon = paths.shape[2]
    colour = MODEL_COLOURS["BSTS-NUTS"]

    gbdt, gbdt_meta = load_artifact(cfg.paths.artifacts / "gbdt")
    quantiles = gbdt["forecast_quantiles"]  # (O, Q, H), LightGBM's per-step marginals
    levels = np.array(gbdt_meta["quantile_levels"])

    # --- 1. A calibrated day-ahead density field against the observed ---------
    i = FAN_ORIGIN
    times = _clock(origins[i], horizon)
    day = times[0].strftime("%a %d %b %Y")
    crps = crps_samples(y[i], paths[:, i, :]).mean()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    density_ribbon(
        ax,
        times,
        draw_mean=ar["forecast_path_mean"][:, i, :],
        draw_sd=ar["forecast_path_sd"][:, i, :],
        observed=y[i],
        cmap=sequential_cmap(colour),
        label="median forecast",
    )
    _time_axis(ax)
    ax.set_title(f"Day-ahead BSTS forecast for NSW1 demand, {day}")
    ax.annotate(
        f"CRPS {crps:.0f} MW",
        (0.985, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="0.3",
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save_figure(fig, "readme_forecast_fan", cfg.paths.figures)

    # --- 2. Coherent BSTS paths vs LightGBM's independent marginals ----------
    # An example day with no date: the BSTS draws whole coherent days; the only
    # path LightGBM can offer is one independent draw per half hour from its
    # per-step quantiles, which assumes the 48 timestamps are independent.
    j = SCENARIO_ORIGIN
    times = _clock(origins[j], horizon)
    lg_colour = MODEL_COLOURS["LightGBM"]

    order = np.argsort(levels)  # np.interp needs the quantile levels ascending
    lv, qj = levels[order], quantiles[j][order]  # (Q,), (Q, H)
    rng = np.random.default_rng(cfg.seed)
    u = rng.uniform(size=(N_SCENARIOS, horizon))  # an independent level per step
    marginal = np.stack([np.interp(u[:, h], lv, qj[:, h]) for h in range(horizon)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    axes[0].plot(times, paths[:N_SCENARIOS, j, :].T, color=colour, alpha=0.32, lw=0.9)
    axes[1].plot(times, marginal.T, color=lg_colour, alpha=0.32, lw=0.9)
    axes[0].set_title("BSTS coherent scenarios: each line a whole plausible day")
    axes[1].set_title("LightGBM marginals sampled independently: jagged, no joint law")
    for ax in axes:
        _time_axis(ax)
    axes[1].set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "readme_coherent_traces", cfg.paths.figures)

    aest = origins.tz_convert(DISPLAY_TZ)
    print(f"fan day      idx {i}  {aest[i].strftime('%Y-%m-%d')} AEST  CRPS {crps:.0f} MW")
    print(f"scenario day idx {j}  {N_SCENARIOS} coherent vs {N_SCENARIOS} marginal draws")


if __name__ == "__main__":
    main()
