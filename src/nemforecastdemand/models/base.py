"""Forecaster interface, the shared design matrix and the naive benchmark.

Every model consumes the same design matrix so that comparisons isolate the
model and the inference, not the features. The design combines the seasonal
basis (local-clock phases), weather terms (temperature, degree days, direct
and diffuse irradiance), lagged demand and the holiday indicator.

Weather enters through one of three variants:

- ``forecast``: the archived day-ahead forecast as issued (the headline);
- ``actual``: ERA5 perfect foresight, an explicit upper bound;
- perturbed: actuals plus a calibrated correlated error, for the robustness
  sweep, supplied through ``overrides``.

Demand lags look backwards only and both configured lags (48 and 336 half
hours) stay behind the forecast origin across a 48-step horizon, so designs
for future steps never touch future demand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nemforecastdemand.config import Config
from nemforecastdemand.features.calendar import (
    LOCAL_TZ,
    fourier_design,
    local_phases,
    seasonal_design,
)
from nemforecastdemand.features.weather import degree_days
from nemforecastdemand.splits import horizon_index

WEATHER_SOURCES = {"actual": "", "forecast": "_fc"}


@dataclass
class Forecast:
    """One probabilistic forecast: 48 half hours from a single origin.

    Exactly one distributional representation is set: ``sd`` (Gaussian
    predictive, the classical models), ``samples`` (posterior predictive
    draws, the Bayesian models) or ``quantile_values`` with their levels
    (the gradient-boosted quantile heads). ``mean`` is the point forecast,
    the predictive median for quantile forecasters.
    """

    origin: pd.Timestamp
    index: pd.DatetimeIndex
    mean: np.ndarray
    sd: np.ndarray | None = None
    samples: np.ndarray | None = None
    quantile_levels: np.ndarray | None = None
    quantile_values: np.ndarray | None = None


def weather_columns(panel: pd.DataFrame, source: str) -> pd.DataFrame:
    """Select the actual or forecast weather columns under canonical names.

    Returned as float64 so perturbation overrides assign losslessly.
    """
    suffix = WEATHER_SOURCES[source]
    return pd.DataFrame(
        {
            "temp_c": panel[f"temp{suffix}_c"],
            "dew_c": panel[f"dew{suffix}_c"],
            "dni_wm2": panel[f"dni{suffix}_wm2"],
            "dhi_wm2": panel[f"dhi{suffix}_wm2"],
        }
    ).astype(np.float64)


def build_design(
    panel: pd.DataFrame,
    cfg: Config,
    weather_source: str = "actual",
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the model design matrix over the panel index.

    Parameters
    ----------
    panel
        The processed panel (or a contiguous concatenation of splits).
    cfg
        Project configuration.
    weather_source
        ``actual`` or ``forecast``.
    overrides
        Optional replacement weather values (canonical column names) for a
        subset of the index, used by the perturbation sweep. Overrides are
        applied before degree days are computed.

    Returns
    -------
    pandas.DataFrame
        Float64 design aligned to ``panel.index``. Rows within the longest
        demand lag of the panel start are NaN; callers slice them away.
    """
    weather = weather_columns(panel, weather_source)
    if overrides is not None:
        weather.loc[overrides.index, overrides.columns] = overrides

    degrees = degree_days(weather["temp_c"], cfg.weather.heating_base, cfg.weather.cooling_base)
    blocks = [
        seasonal_design(panel.index, cfg.features),
        weather,
        degrees,
        pd.DataFrame(
            {f"demand_lag{lag}": panel["demand_mw"].shift(lag) for lag in cfg.features.demand_lags}
        ),
        panel["is_holiday"].astype(np.float64).to_frame(),
    ]
    if cfg.features.interaction_harmonics > 0:
        # The train-split residuals show the temperature response flipping
        # sign by time of day and a weekend-specific morning ramp, so the
        # degree days and the weekend flag interact with a small daily
        # basis. Everything stays linear in parameters and pointwise.
        daily, _ = local_phases(panel.index)
        basis = fourier_design(daily, cfg.features.interaction_harmonics, "ix")
        basis.index = panel.index
        weekend = (panel.index.tz_convert(LOCAL_TZ).dayofweek >= 5).astype(np.float64)
        interactions = {"is_weekend": weekend}
        for column in basis.columns:
            interactions[f"cooling_{column}"] = degrees["cooling_deg"] * basis[column]
            interactions[f"heating_{column}"] = degrees["heating_deg"] * basis[column]
            interactions[f"weekend_{column}"] = weekend * basis[column]
        blocks.append(pd.DataFrame(interactions, index=panel.index))
    if cfg.features.hsgp_time_harmonics > 0 and cfg.features.hsgp_temp_basis > 0:
        from nemforecastdemand.models.hsgp import gp_design

        blocks.append(gp_design(panel.index, weather["temp_c"], cfg.features))
    return pd.concat(blocks, axis=1).astype(np.float64)


def recency_features(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Origin-anchored features carrying the AR(1) information set to trees.

    The time-series models condition on the residual at the forecast origin
    through their error dynamics; a direct regression sees one row per
    target time and cannot. These three columns close that gap: how far the
    last observed half hour sits above its day-ago and week-ago values, and
    how many steps ahead the target is, from which trees can learn a
    decaying correction. Everything is computed strictly before the origin.
    """
    index = horizon_index(origin, horizon)
    demand = panel["demand_mw"]
    last = origin - pd.Timedelta("30min")
    step = pd.Timedelta("30min")
    return pd.DataFrame(
        {
            "horizon_step": np.arange(horizon, dtype=np.float64),
            "dev_day": float(demand.loc[last] - demand.loc[last - 48 * step]),
            "dev_week": float(demand.loc[last] - demand.loc[last - 336 * step]),
        },
        index=index,
    )


def stacked_origin_design(
    panel: pd.DataFrame,
    cfg: Config,
    origins: pd.DatetimeIndex,
    weather_source: str = "actual",
) -> tuple[pd.DataFrame, pd.Series]:
    """Training rows for direct regressors, one 48-step block per origin.

    Each block is the shared design over the origin's horizon plus the
    origin-anchored recency columns, so training mirrors the operational
    setting exactly: timestamps covered by both daily origins appear twice
    with different recency values, once per issue time.
    """
    design = build_design(panel, cfg, weather_source=weather_source)
    blocks, targets = [], []
    for origin in origins:
        index = horizon_index(origin, cfg.horizon)
        block = pd.concat([design.loc[index], recency_features(panel, origin, cfg.horizon)], axis=1)
        blocks.append(block)
        targets.append(panel["demand_mw"].loc[index])
    return pd.concat(blocks), pd.concat(targets).astype(np.float64)


def variance_design(
    panel: pd.DataFrame, cfg: Config, weather_source: str = "actual"
) -> pd.DataFrame:
    """Design for the log observation scale: small daily basis plus degree days.

    Kept deliberately low-dimensional. The variance head multiplies through
    an exponential link, so every extra degree of freedom here costs more
    geometry than it buys fit.
    """
    daily, _ = local_phases(panel.index)
    blocks = [fourier_design(daily, cfg.bsts.variance_daily_harmonics, "vdaily")]
    if cfg.bsts.variance_use_degree_days:
        weather = weather_columns(panel, weather_source)
        blocks.append(
            degree_days(
                weather["temp_c"], cfg.weather.heating_base, cfg.weather.cooling_base
            ).reset_index(drop=True)
        )
    design = pd.concat(blocks, axis=1)
    design.index = panel.index
    return design.astype(np.float64)


class Forecaster(ABC):
    """Interface every model implements: fit once, forecast from any origin."""

    name: str

    @abstractmethod
    def fit(self, panel: pd.DataFrame, fit_index: pd.DatetimeIndex) -> Forecaster:
        """Fit on the given index of the panel, returning self."""

    @abstractmethod
    def forecast(
        self,
        panel: pd.DataFrame,
        origin: pd.Timestamp,
        weather_source: str = "forecast",
        overrides: pd.DataFrame | None = None,
    ) -> Forecast:
        """Forecast the 48 half hours starting at ``origin``.

        The model may condition on everything in ``panel`` strictly before
        ``origin`` plus the covariates over the horizon under the chosen
        weather variant.
        """


def perturbation_overrides(
    panel: pd.DataFrame,
    index: pd.DatetimeIndex,
    models: dict[str, object],
    multiplier: float,
    seed: int,
) -> pd.DataFrame:
    """Perturbed weather covariates for one forecast horizon.

    Each variable's fitted error model draws one correlated path; the seed
    is derived from the origin so the sweep is reproducible and each origin
    sees an independent error realisation.
    """
    market = index.tz_convert("Australia/Brisbane")
    steps = (market.hour * 2 + market.minute // 30).to_numpy()
    origin_token = int(index[0].value) % (2**31)
    overrides = {}
    for i, (column, model) in enumerate(models.items()):
        rng = np.random.default_rng([seed, origin_token, i, round(multiplier * 10)])
        actual = panel.loc[index, column].to_numpy(dtype=np.float64)
        overrides[column] = model.sample(actual, steps, multiplier, rng)
    return pd.DataFrame(overrides, index=index)


def run_variants(
    forecaster: Forecaster,
    panel: pd.DataFrame,
    origins: pd.DatetimeIndex,
    perturbations: dict[str, object],
    multipliers: tuple[float, ...],
    seed: int,
) -> dict[str, list[Forecast]]:
    """Forecast every origin under every weather-input variant.

    Variants: ``forecast`` (archived day-ahead forecast, the headline),
    ``actual`` (ERA5 perfect foresight, a disclosed upper bound) and
    ``perturb_{m}`` for each sweep multiplier, where actuals carry a
    calibrated correlated error of m times the measured magnitude.
    """
    variants: dict[str, list[Forecast]] = {"forecast": [], "actual": []}
    for multiplier in multipliers:
        if multiplier > 0:
            variants[f"perturb_{multiplier:g}"] = []

    for count, origin in enumerate(origins, start=1):
        index = horizon_index(origin, 48)
        variants["forecast"].append(forecaster.forecast(panel, origin, "forecast"))
        variants["actual"].append(forecaster.forecast(panel, origin, "actual"))
        for multiplier in multipliers:
            if multiplier == 0:
                continue
            overrides = perturbation_overrides(panel, index, perturbations, multiplier, seed)
            variants[f"perturb_{multiplier:g}"].append(
                forecaster.forecast(panel, origin, "actual", overrides=overrides)
            )
        if count % 25 == 0 or count == len(origins):
            print(f"  forecasts: {count}/{len(origins)} origins", flush=True)
    return variants


class SeasonalNaive(Forecaster):
    """Same half hour last week, with a Gaussian band from training errors.

    The benchmark every model must beat. The predictive standard deviation
    is the per-half-hour-of-week standard deviation of weekly-naive errors
    on the fitting window, so even the naive model is probabilistically
    honest rather than a bare point forecast.
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 336) -> None:
        self.season = season
        self._sd_by_step: pd.Series | None = None
        self._train_mae: float | None = None

    def fit(self, panel: pd.DataFrame, fit_index: pd.DatetimeIndex) -> SeasonalNaive:
        demand = panel["demand_mw"].astype(np.float64)
        errors = (demand - demand.shift(self.season)).loc[fit_index].dropna()
        market = errors.index.tz_convert("Australia/Brisbane")
        step = market.dayofweek * 48 + market.hour * 2 + market.minute // 30
        self._sd_by_step = errors.groupby(step).std()
        self._train_mae = float(errors.abs().mean())
        return self

    @property
    def train_mae(self) -> float:
        """Mean absolute weekly-naive error on the fit window, the MASE base."""
        if self._train_mae is None:
            raise RuntimeError("fit the model first")
        return self._train_mae

    def forecast(
        self,
        panel: pd.DataFrame,
        origin: pd.Timestamp,
        weather_source: str = "forecast",
        overrides: pd.DataFrame | None = None,
    ) -> Forecast:
        if self._sd_by_step is None:
            raise RuntimeError("fit the model first")
        index = horizon_index(origin, 48)
        mean = panel["demand_mw"].reindex(index - pd.Timedelta("30min") * self.season)
        market = index.tz_convert("Australia/Brisbane")
        step = market.dayofweek * 48 + market.hour * 2 + market.minute // 30
        sd = self._sd_by_step.reindex(step).to_numpy()
        return Forecast(
            origin=origin,
            index=index,
            mean=mean.to_numpy(dtype=np.float64),
            sd=sd.astype(np.float64),
        )
