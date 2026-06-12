# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (nem-demand-forecast)
#     language: python
#     name: nem-demand-forecast
# ---

# %% [markdown]
# # 01. Exploratory analysis and data cleansing
#
# This notebook examines the raw inputs, verifies that timestamps and joins
# are exactly right, justifies the cleansing decisions and writes the
# processed train, validation and test splits consumed by every model
# notebook. `scripts/build_dataset.py` mirrors the build steps headlessly.
#
# The data are NSW1 operational demand actuals from NEMWeb (half-hourly),
# ERA5 reanalysis weather and archived ECMWF IFS day-ahead forecasts from
# Open-Meteo (hourly, interpolated onto the half-hourly grid). Everything is
# stored and modelled in UTC with period-start timestamps; plots are shown in
# AEST, the NEM market time. Weather data by
# [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from nemforecastdemand.config import load_config
from nemforecastdemand.data import weather
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.features.calendar import holiday_flag
from nemforecastdemand.features.preprocessing import build_panel, half_hourly_grid
from nemforecastdemand.plotting import (
    DISPLAY_TZ,
    LOCAL_TZ,
    display_index,
    format_date_axis,
    palette,
    plot_clock_profile,
    save_figure,
    setup_style,
)
from nemforecastdemand.splits import chronological_split, split_summary

setup_style()
cfg = load_config()

# %%
demand = (
    pl.read_parquet(cfg.paths.interim / "demand.parquet").to_pandas().set_index("ts")["demand_mw"]
)
era5 = weather.load_raw(cfg.paths.raw / "weather" / "era5.parquet")
forecast = weather.load_raw(cfg.paths.raw / "weather" / "forecast.parquet")
demand.describe().to_frame().T

# %% [markdown]
# ## Timestamp and timezone audit
#
# Three conventions interact here and each is a classic source of silent
# misalignment, so they are verified rather than assumed.
#
# 1. AEMO publishes half hours stamped with their **ending** time in market
#    time (AEST, a fixed UTC+10 with no daylight saving). The data layer
#    shifts these to period-start stamps and converts to UTC.
# 2. Open-Meteo series were requested in UTC directly.
# 3. NSW behaviour follows the **local Sydney clock**, which moves against
#    both UTC and market time at daylight-saving transitions.
#
# First, the grid itself: the demand series should cover every half hour of
# the window exactly once.

# %%
grid = half_hourly_grid(demand.index[0], demand.index[-1])
audit = pd.DataFrame(
    {
        "expected half hours": [len(grid)],
        "present": [len(demand)],
        "missing": [len(grid.difference(demand.index))],
        "duplicates": [int(demand.index.duplicated().sum())],
        "first (AEST)": [display_index(demand.index[:1])[0]],
        "last (AEST)": [display_index(demand.index[-1:])[0]],
    }
)
audit

# %% [markdown]
# The grid is complete and duplicate-free, and the window runs from market
# midnight to market midnight as intended.
#
# ### The daily shape moves with the local clock, not the market clock
#
# Market time has no daylight saving but Sydney does. If demand is profiled
# against the market clock, the entire daily shape should jump by an hour
# when DST begins. The four weeks either side of the October 2025 transition
# make the point cleanly, and re-profiling against the local clock should
# remove the jump. This matters for modelling: seasonal features built on the
# market clock would smear the morning and evening ramps for half the year,
# so every calendar feature in this project is computed from the local clock.

# %%
before_dst = demand.loc["2025-09-07":"2025-10-04"]
after_dst = demand.loc["2025-10-06":"2025-11-02"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, tz, title in (
    (axes[0], DISPLAY_TZ, "market clock (AEST)"),
    (axes[1], LOCAL_TZ, "local Sydney clock"),
):
    plot_clock_profile(ax, before_dst, tz, "four weeks before DST", palette("demand"))
    plot_clock_profile(ax, after_dst, tz, "four weeks after DST", palette("accent"))
    ax.set_title(title)
axes[0].set_ylabel("demand (MW)")
axes[1].legend()
fig.suptitle("Daylight saving shifts the daily shape on the market clock only", y=1.04)
save_figure(fig, "dst_daily_shape", cfg.paths.figures)
plt.show()

# %% [markdown]
# The hour-wide displacement on the left collapses on the right, confirming
# both the timezone handling and the choice of clock for seasonal features.
#
# ### Cross-checking the demand-weather join
#
# A timezone slip in either series would shift the demand-weather
# relationship by hours, so the join is verified two ways.
#
# **Temperature.** Correlating the raw series would mostly measure the phase
# difference between two daily cycles (temperature peaks mid-afternoon,
# demand peaks early evening), so the scan uses **anomalies**: each summer
# series minus its own mean daily profile. A hot spell should lift demand
# within a few hours, with demand trailing temperature (buildings and
# behaviour integrate heat), so the peak should sit a little above zero lag
# and nowhere near the ten hours a UTC-versus-AEST slip would produce.
#
# **Irradiance.** Sunlight obeys astronomy: direct normal irradiance must be
# zero through the local night and peak near local solar noon. This is an
# exact, demand-free check of the weather clock.

# %%
summer = slice("2025-12-01", "2026-02-28")
temp_half_hourly = (
    era5["temperature_2m"].reindex(era5.index.union(grid)).interpolate("time").reindex(grid)
)
dni_half_hourly = (
    era5["direct_normal_irradiance"]
    .reindex(era5.index.union(grid))
    .interpolate("time")
    .reindex(grid)
)


def daily_anomaly(series: pd.Series) -> pd.Series:
    local_clock = series.index.tz_convert(LOCAL_TZ)
    step = local_clock.hour * 2 + local_clock.minute // 30
    return series - series.groupby(step).transform("mean")


demand_anomaly = daily_anomaly(demand.loc[summer])
temp_anomaly = daily_anomaly(temp_half_hourly.loc[summer])
lags = np.arange(-12, 13)
ccf = pd.Series([demand_anomaly.corr(temp_anomaly.shift(int(k))) for k in lags], index=lags / 2.0)

local_hour = dni_half_hourly.index.tz_convert(LOCAL_TZ).hour + (
    dni_half_hourly.index.tz_convert(LOCAL_TZ).minute / 60
)
dni_profile = dni_half_hourly.groupby(local_hour).mean()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ccf.index, ccf, color=palette("temperature"))
axes[0].axvline(0, color="grey", lw=0.8)
axes[0].set_xlabel("lag (hours, positive = demand trails temperature)")
axes[0].set_ylabel("anomaly correlation")
axes[0].set_title("Summer demand follows temperature anomalies")
axes[1].plot(dni_profile.index, dni_profile, color=palette("irradiance"))
axes[1].set_xlabel("local Sydney hour")
axes[1].set_ylabel("mean DNI (W/m2)")
axes[1].set_xticks(np.arange(0, 25, 3))
axes[1].set_title("Irradiance obeys the local sun")
plt.show()

night = dni_half_hourly[
    np.isin(dni_half_hourly.index.tz_convert(LOCAL_TZ).hour, [22, 23, 0, 1, 2, 3])
]
print(
    f"temperature anomaly correlation peaks at {ccf.idxmax():+.1f} h; "
    f"DNI profile peaks at {dni_profile.idxmax():.1f} local, night-time max {night.max():.1f} W/m2"
)

# %% [markdown]
# The anomaly correlation peaks three hours after the temperature anomaly,
# the right sign and size for thermal inertia and far from the ten-hour
# displacement a timezone slip would produce. Irradiance is identically zero
# at night and peaks within an hour of local solar noon, pinning the weather
# clock exactly. (The demand-irradiance anomaly correlation is deliberately
# not used here: sunshine suppresses demand through rooftop PV immediately
# but raises it through heat a few hours later, and the competing channels
# cancel near zero lag. That same confounding is why temperature and both
# irradiance components all enter the feature set together.)
#
# ## What drives NSW1 demand
#
# The window covers a full year plus a repeated autumn, so both seasons'
# behaviour is visible. Demand carries a strong daily cycle, a weekly cycle
# from working patterns, a U-shaped temperature response (heating below
# roughly 17 C, cooling above roughly 20 C) and midday suppression from
# rooftop solar.

# %%
daily_mean = demand.resample("1D").mean()
daily_temp = temp_half_hourly.resample("1D").mean()

fig, ax = plt.subplots()
ax.plot(display_index(daily_mean.index), daily_mean, color=palette("demand"), lw=1.0)
ax.set_ylabel("daily mean demand (MW)", color=palette("demand"))
twin = ax.twinx()
twin.plot(
    display_index(daily_temp.index), daily_temp, color=palette("temperature"), lw=0.8, alpha=0.6
)
twin.set_ylabel("daily mean temperature (C)", color=palette("temperature"))
twin.grid(False)
format_date_axis(ax)
ax.set_title("Winter heating and summer cooling both lift demand")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 5))
hb = ax.hexbin(temp_half_hourly, demand, gridsize=60, cmap="cividis", mincnt=1)
ax.set_xlabel("temperature (C)")
ax.set_ylabel("demand (MW)")
ax.set_title("The U-shaped temperature response")
fig.colorbar(hb, label="half hours")
save_figure(fig, "demand_temperature", cfg.paths.figures)
plt.show()

# %% [markdown]
# The trough of the U sits around 17 to 20 C, supporting the configured
# heating and cooling bases (16.5 and 20.5 C) for the degree-day features.
# The vertical spread at any given temperature is the daily cycle, which the
# seasonal basis carries. Dew point enters the feature set alongside
# dry-bulb temperature because humid summer evenings keep air conditioning
# running at temperatures that would otherwise need none.
#
# Rooftop solar is the other defining feature of the modern NEM load shape.
# Splitting days by midday irradiance shows the hollowed-out middle of the
# day on sunny days; this is why direct and diffuse irradiance enter the
# feature set alongside temperature.

# %%
local = demand.index.tz_convert(LOCAL_TZ)
midday = (local.hour >= 10) & (local.hour < 14)
midday_dni = dni_half_hourly[midday].groupby(local[midday].date).mean()
spring = midday_dni.loc[
    (pd.to_datetime(midday_dni.index) >= "2025-09-01")
    & (pd.to_datetime(midday_dni.index) < "2025-12-01")
]
sunny = demand[np.isin(local.date, spring.nlargest(25).index)]
overcast = demand[np.isin(local.date, spring.nsmallest(25).index)]

fig, ax = plt.subplots(figsize=(7, 4))
plot_clock_profile(ax, sunny, LOCAL_TZ, "25 sunniest spring days", palette("irradiance"))
plot_clock_profile(ax, overcast, LOCAL_TZ, "25 most overcast spring days", palette("demand"))
ax.set_ylabel("demand (MW)")
ax.set_title("Rooftop solar hollows out sunny middays")
ax.legend()
save_figure(fig, "duck_curve", cfg.paths.figures)
plt.show()

# %% [markdown]
# The weekly cycle and the public holiday effect close out the calendar
# structure: weekends and holidays drop the working-hours load, and holidays
# behave like Sundays regardless of weekday, which motivates the holiday
# indicator in the design matrix.

# %%
local_dow = demand.index.tz_convert(LOCAL_TZ).dayofweek
day_type = pd.Series(
    np.select(
        [holiday_flag(demand.index).to_numpy(), local_dow >= 5],
        ["holiday", "weekend"],
        default="weekday",
    ),
    index=demand.index,
)

fig, ax = plt.subplots(figsize=(7, 4))
for kind, colour in (
    ("weekday", palette("demand")),
    ("weekend", palette("forecast")),
    ("holiday", palette("accent")),
):
    plot_clock_profile(ax, demand[day_type == kind], DISPLAY_TZ, kind, colour)
ax.set_ylabel("demand (MW)")
ax.set_title("Holidays behave like weekends")
ax.legend()
plt.show()

# %% [markdown]
# ## How good are the day-ahead weather forecasts?
#
# The headline evaluation feeds the models the ECMWF IFS forecast exactly as
# issued one day earlier, so its error against the ERA5 ground truth is part
# of the story. Two caveats are acknowledged here rather than hidden. First,
# coefficients are trained on ERA5 actuals while forecasts come from a
# different model with its own climatology, a mild train/serve mismatch;
# bias-correcting the forecast onto ERA5 is noted as future work. Second,
# ERA5 itself is reanalysis, not station truth. The residual structure below
# also calibrates the perturbation sweep used in notebook 05.

# %%
panel, report = build_panel(
    pl.read_parquet(cfg.paths.interim / "demand.parquet"), era5, forecast, cfg
)
residuals = pd.DataFrame(
    {
        "temperature (C)": panel["temp_fc_c"] - panel["temp_c"],
        "dew point (C)": panel["dew_fc_c"] - panel["dew_c"],
        "DNI (W/m2)": panel["dni_fc_wm2"] - panel["dni_wm2"],
    }
)
residuals.describe().loc[["mean", "std", "min", "max"]].T

# %%
hour = panel.index.tz_convert(DISPLAY_TZ).hour + panel.index.tz_convert(DISPLAY_TZ).minute / 60
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, column in zip(axes, residuals.columns, strict=True):
    grouped = residuals[column].groupby(hour)
    ax.plot(grouped.mean().index, grouped.mean(), color=palette("forecast"), label="bias")
    ax.plot(grouped.std().index, grouped.std(), color=palette("temperature"), label="spread")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"day-ahead {column} error by hour (AEST)")
    ax.set_xlabel("hour of day")
axes[0].legend()
plt.show()

# %% [markdown]
# Temperature errors are roughly half a degree at night and around a degree
# in the afternoon; irradiance errors concentrate in daylight hours as cloud
# timing is the hard part. Both are small enough to leave clear weather
# signal and large enough that perfect-foresight evaluation would flatter
# every model, which is why both variants are scored in notebook 05.
#
# ## Cleansing
#
# The panel build applies three guarded repairs, each counted and surfaced:
# interpolation of demand gaps up to one hour, a Hampel screen (rolling
# weekly median, eight scaled MADs) for telemetry spikes and interpolation of
# short weather-forecast gaps with an actuals fallback for missing archive
# runs. The thresholds are deliberately conservative: genuine demand peaks
# are signal, not faults, and this window arrives clean from NEMWeb.

# %%
report.as_frame()

# %% [markdown]
# Nothing needed repair in this window. The screens stay in the pipeline
# because reruns over future windows inherit them, and the zero counts are
# themselves evidence the upstream feeds are curated.
#
# ## Chronological splits
#
# Roughly 70/15/15 by whole market days, no shuffling: train teaches the
# models, validation selects the ARIMA order and seasonal basis and test is
# touched only by the final rolling-origin evaluation. The splits are written
# as parquet and committed, so every later notebook starts from identical,
# schema-validated data.

# %%
splits = chronological_split(panel.index, cfg.splits.train, cfg.splits.validation)
cfg.paths.processed.mkdir(parents=True, exist_ok=True)
for name, index in splits.items():
    panel.loc[index].to_parquet(cfg.paths.processed / f"{name}.parquet")
load_splits(cfg.paths.processed)
split_summary(splits)

# %% [markdown]
# ## Trigonometric or radial seasonal basis?
#
# The daily and weekly cycles can be encoded as Fourier harmonics or as
# periodic Gaussian radial basis functions over the same local-clock phases,
# with matched column counts so model size stays fixed. Before any time-series
# machinery, a plain least-squares regression of demand on the full design
# (basis, weather, lags, holiday) settles which basis carries the seasonal
# shape better as a predictor: fit on train, score on validation, no
# leakage. The full ARIMA refits the comparison in notebook 02 as
# confirmation.

# %%
from dataclasses import replace

from nemforecastdemand.models.base import build_design

y_all = panel["demand_mw"].astype(np.float64)
basis_rows = {}
basis_errors = {}
for basis in ("fourier", "rbf"):
    basis_cfg = replace(cfg, features=replace(cfg.features, seasonal_basis=basis))
    design = build_design(panel, basis_cfg, weather_source="actual")
    train_design = design.loc[splits["train"]].dropna()
    val_design = design.loc[splits["validation"]]
    train_matrix = np.column_stack([np.ones(len(train_design)), train_design.to_numpy()])
    val_matrix = np.column_stack([np.ones(len(val_design)), val_design.to_numpy()])
    coef, *_ = np.linalg.lstsq(train_matrix, y_all.loc[train_design.index].to_numpy(), rcond=None)
    errors = y_all.loc[val_design.index] - val_matrix @ coef
    basis_errors[basis] = errors
    basis_rows[basis] = {
        "design columns": design.shape[1],
        "validation MAE (MW)": float(errors.abs().mean()),
        "validation RMSE (MW)": float(np.sqrt((errors**2).mean())),
    }
pd.DataFrame(basis_rows).T

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for basis, colour in (("fourier", palette("demand")), ("rbf", palette("accent"))):
    hourly_mae = (
        basis_errors[basis]
        .abs()
        .groupby(basis_errors[basis].index.tz_convert(LOCAL_TZ).hour)
        .mean()
    )
    ax.plot(hourly_mae.index, hourly_mae, label=basis, color=colour)
ax.set_xlabel("local Sydney hour")
ax.set_ylabel("validation MAE (MW)")
ax.set_title("The two bases are near-indistinguishable as predictors")
ax.legend()
plt.show()

# %% [markdown]
# The bases land within noise of each other at matched size, including
# through the sharp morning ramp where the localised bumps were expected to
# help. The trigonometric basis is retained as the default for its exact
# periodicity and interpretability; the RBF basis remains one configuration
# switch away, and notebook 02 repeats this comparison inside the full
# ARIMA model.
#
# ## What stays non-linear after the linear design
#
# A design that enters every model linearly should be checked for the
# interactions it cannot represent. Fitting the additive design on train
# and tabulating the residual mean by time-of-day block and temperature
# band gives a direct signature: a purely additive temperature response
# would leave these cells near zero.

# %%
plain_cfg = replace(cfg, features=replace(cfg.features, interaction_harmonics=0))
design = build_design(panel, plain_cfg, weather_source="actual")
train_design = design.loc[splits["train"]].dropna()
train_matrix = np.column_stack([np.ones(len(train_design)), train_design.to_numpy()])
coef, *_ = np.linalg.lstsq(train_matrix, y_all.loc[train_design.index].to_numpy(), rcond=None)
resid = pd.Series(
    y_all.loc[train_design.index].to_numpy() - train_matrix @ coef, index=train_design.index
)

local_resid = resid.index.tz_convert(LOCAL_TZ)
signature = pd.DataFrame(
    {
        "resid": resid.to_numpy(),
        "block": pd.cut(
            local_resid.hour,
            [0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            right=False,
        ),
        "temp": pd.cut(
            train_design["temp_c"],
            [-5, 10, 15, 20, 25, 45],
            labels=["<10C", "10-15C", "15-20C", "20-25C", ">25C"],
        ).to_numpy(),
        "weekend": local_resid.dayofweek >= 5,
        "hour": local_resid.hour,
    }
)
pivot = signature.pivot_table(
    values="resid", index="block", columns="temp", aggfunc="mean", observed=True
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im = axes[0].imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-450, vmax=450, aspect="auto")
axes[0].set_xticks(range(pivot.shape[1]), pivot.columns)
axes[0].set_yticks(range(pivot.shape[0]), pivot.index)
fig.colorbar(im, ax=axes[0], label="mean residual (MW)")
axes[0].set_title("Temperature response flips sign by time of day")

weekend_gap = signature.pivot_table(values="resid", index="hour", columns="weekend", aggfunc="mean")
axes[1].plot(weekend_gap.index, weekend_gap[False], color=palette("demand"), label="weekday")
axes[1].plot(weekend_gap.index, weekend_gap[True], color=palette("accent"), label="weekend")
axes[1].set_xlabel("local Sydney hour")
axes[1].set_ylabel("mean residual (MW)")
axes[1].set_title("The weekend morning ramp the weekly basis misses")
axes[1].legend()
fig.tight_layout()
save_figure(fig, "nonlinearity_signature", cfg.paths.figures)
plt.show()

# %% [markdown]
# Two clean interaction signatures: hot evenings sit hundreds of megawatts
# above the additive fit while hot nights sit below it (the
# air-conditioning load follows the clock, not just the thermometer), and
# weekend mornings run far from the weekday profile. Both are still
# linear in parameters once the right columns exist: degree days and a
# weekend flag interacted with a small daily Fourier basis. The size of
# that basis is selected on validation, fitted on train only.

# %%
interaction_rows = {}
for n_ix in (0, 2, 3):
    ix_cfg = replace(cfg, features=replace(cfg.features, interaction_harmonics=n_ix))
    design = build_design(panel, ix_cfg, weather_source="actual")
    train_design = design.loc[splits["train"]].dropna()
    val_design = design.loc[splits["validation"]]
    train_matrix = np.column_stack([np.ones(len(train_design)), train_design.to_numpy()])
    val_matrix = np.column_stack([np.ones(len(val_design)), val_design.to_numpy()])
    coef, *_ = np.linalg.lstsq(train_matrix, y_all.loc[train_design.index].to_numpy(), rcond=None)
    errors = y_all.loc[val_design.index] - val_matrix @ coef
    interaction_rows[f"{n_ix} interaction harmonics"] = {
        "design columns": design.shape[1],
        "validation MAE (MW)": float(errors.abs().mean()),
        "validation RMSE (MW)": float(np.sqrt((errors**2).mean())),
    }
pd.DataFrame(interaction_rows).T

# %% [markdown]
# Two harmonics capture the gain; more buys nothing. The interaction block
# is enabled in the configuration, so every model downstream (classical,
# boosted, Bayesian) sees the same enriched design.
#
# ## Summary
#
# - The half-hourly grid is complete, duplicate-free and verified against
#   the interval-ending-to-period-start and AEST-to-UTC conversions.
# - Daily structure follows the local Sydney clock through daylight saving,
#   so all seasonal features are built on local-clock phases.
# - Lagged correlations place the demand-weather join within a couple of
#   hours of zero, ruling out timezone slips.
# - Demand shows daily, weekly and holiday structure, a U-shaped temperature
#   response around a 17 to 20 C comfort band and solar suppression of
#   midday load, motivating the configured feature set.
# - The additive design leaves two interaction signatures in its train
#   residuals (temperature response varying by time of day, a distinct
#   weekend morning ramp); degree-day and weekend interactions with two
#   daily harmonics close them, selected on validation.
# - Day-ahead forecast errors are material but modest, quantified here to
#   calibrate the perturbation sweep.
# - Train, validation and test splits are written, validated and committed.
