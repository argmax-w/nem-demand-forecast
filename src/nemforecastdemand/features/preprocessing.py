"""Resampling, alignment, imputation and outlier handling for the panel.

Everything here operates on UTC period-start indices and is vectorised:
interpolation runs on whole frames, outlier detection uses rolling windows
and no row-wise Python appears anywhere. The cleansing decisions (gap limits,
Hampel threshold) are surfaced and justified in notebook 01.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from nemforecastdemand.config import Config
from nemforecastdemand.features.calendar import holiday_flag
from nemforecastdemand.gates import bounds_for

#: Mapping from Open-Meteo variable names to panel column stems.
VARIABLE_STEMS = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dew_c",
    "direct_normal_irradiance": "dni_wm2",
    "diffuse_radiation": "dhi_wm2",
    "apparent_temperature": "apptemp_c",
    "shortwave_radiation": "ghi_wm2",
    "wind_speed_10m": "wind_kmh",
}

#: Irradiance columns are non-negative and physically zero when the sun is
#: below the horizon; the cleaner enforces both rather than smearing
#: interpolated values across sunrise.
IRRADIANCE_STEMS = ("dni_wm2", "dhi_wm2", "ghi_wm2")
#: Other non-negative weather columns.
NONNEGATIVE_STEMS = ("wind_kmh",)


@dataclass
class CleansingReport:
    """Counts of every repair made while building the panel."""

    demand_missing: int = 0
    demand_interpolated: int = 0
    demand_outliers: int = 0
    weather_interpolated: dict[str, int] = field(default_factory=dict)
    forecast_fallback: dict[str, int] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        """Tabulate the report for display in the notebook."""
        rows = [
            ("demand_mw", "missing on grid", self.demand_missing),
            ("demand_mw", "interpolated", self.demand_interpolated),
            ("demand_mw", "outliers replaced", self.demand_outliers),
        ]
        rows += [(col, "interpolated", n) for col, n in self.weather_interpolated.items()]
        rows += [(col, "filled from actuals", n) for col, n in self.forecast_fallback.items()]
        return pd.DataFrame(rows, columns=["column", "repair", "count"])


def half_hourly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Build the full UTC half-hourly period-start grid, both ends inclusive."""
    return pd.date_range(start, end, freq="30min", tz="UTC", name="ts")


def hampel_flags(series: pd.Series, window: int = 336, k: float = 8.0) -> pd.Series:
    """Flag outliers with a rolling median absolute deviation filter.

    Parameters
    ----------
    series
        The series to screen.
    window
        Rolling window width in half hours, centred. A week by default, wide
        enough that a hot afternoon is judged against comparable periods.
    k
        Threshold in scaled MAD units. Demand has heavy daily structure, so
        the threshold is deliberately loose: this screens telemetry faults,
        not genuine peaks.

    Returns
    -------
    pandas.Series
        Boolean flags aligned to the input.
    """
    rolling = series.rolling(window, center=True, min_periods=window // 4)
    median = rolling.median()
    mad = (series - median).abs().rolling(window, center=True, min_periods=window // 4).median()
    threshold = k * 1.4826 * mad
    return (series - median).abs() > threshold


def interpolate_to_grid(
    hourly: pd.DataFrame, grid: pd.DatetimeIndex, extrapolate: bool = True
) -> pd.DataFrame:
    """Interpolate hourly weather onto the half-hourly grid.

    Time-based linear interpolation on the union of the two indices, then a
    reindex. Open-Meteo hourly radiation is a preceding-hour mean rather than
    an instantaneous value; treating it as instantaneous at the stamp is an
    approximation that is immaterial for regression features.

    With ``extrapolate`` the leading and trailing edges are filled with the
    nearest value (needed for the actuals, padded only a couple of days
    beyond the window). For the forecasts it is left False so the period
    before the forecast archive begins stays missing and is filled from
    actuals downstream rather than back-extrapolated as a constant.
    """
    union = hourly.index.union(grid)
    direction = "both" if extrapolate else "forward"
    return hourly.reindex(union).interpolate(method="time", limit_direction=direction).reindex(grid)


def solar_elevation_deg(index: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    """Solar elevation in degrees for each UTC timestamp (NOAA approximation).

    Used only to mark when the sun is below the horizon, so irradiance can be
    forced to its physical zero rather than carrying an interpolation artefact.
    """
    local = index.tz_convert("UTC")
    frac_hour = local.hour + local.minute / 60.0
    gamma = 2.0 * np.pi / 365.0 * (local.dayofyear - 1 + (frac_hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    true_solar_min = frac_hour * 60.0 + eqtime + 4.0 * longitude
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)
    lat = np.radians(latitude)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(cos_zenith, -1.0, 1.0)))


def impute_demand(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fill demand gaps and return the filled series with an imputed mask.

    Short gaps (up to one hour) are linearly interpolated; longer gaps take
    the same half hour one week, then two weeks, earlier, because demand is
    far more periodic than linear; anything still missing falls back to time
    interpolation. The mask flags every filled position.
    """
    missing = series.isna()
    filled = series.interpolate(method="time", limit=2)
    for lag in (336, 672):  # one week, two weeks of half hours
        if filled.isna().any():
            filled = filled.fillna(filled.shift(lag))
    if filled.isna().any():
        filled = filled.interpolate(method="time", limit_direction="both")
    return filled, missing


def build_panel(
    demand: pl.DataFrame,
    era5: pd.DataFrame,
    forecast: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, CleansingReport]:
    """Assemble the aligned half-hourly panel from raw inputs.

    Parameters
    ----------
    demand
        Output of :func:`nemforecastdemand.data.aemo.load_demand`.
    era5, forecast
        Raw hourly weather frames on UTC indices.
    cfg
        Project configuration.

    Returns
    -------
    tuple
        The panel (UTC half-hourly grid, float32 numerics) and a report of
        every repair applied.
    """
    report = CleansingReport()

    series = demand.to_pandas().set_index("ts")["demand_mw"]
    grid = half_hourly_grid(series.index[0], series.index[-1])
    series = series.reindex(grid)
    report.demand_missing = int(series.isna().sum())
    series, missing = impute_demand(series)
    report.demand_interpolated = int(missing.sum())
    if series.isna().any():
        raise ValueError("demand has gaps too long to fill; inspect the raw archives")

    flags = hampel_flags(series)
    report.demand_outliers = int(flags.sum())
    series = series.mask(flags).interpolate(method="time", limit_direction="both")
    # Demand cells that are not genuine observations; the AR anchor and lags
    # read this, so the output gate refuses forecasts issued from a run of it.
    demand_imputed = (missing | flags).to_numpy()

    panel = pd.DataFrame({"demand_mw": series})

    night = solar_elevation_deg(grid, cfg.grid_point.latitude, cfg.grid_point.longitude) <= 0.0

    def make_physical(frame: pd.DataFrame) -> pd.DataFrame:
        """Clip weather to physical bounds and zero irradiance at night."""
        for column in frame.columns:
            bounds = bounds_for(column)
            if bounds is not None:
                frame[column] = frame[column].clip(*bounds)
            if column.replace("_fc", "") in IRRADIANCE_STEMS:
                frame.loc[night, column] = 0.0
        return frame

    actuals = make_physical(interpolate_to_grid(era5.rename(columns=VARIABLE_STEMS), grid))
    for column in actuals:
        panel[column] = actuals[column]

    forecast_stems = {}
    for name, stem in VARIABLE_STEMS.items():
        head, tail = stem.split("_", 1)
        forecast_stems[f"{name}_previous_day{cfg.weather.lead_days}"] = f"{head}_fc_{tail}"
    fc = forecast.rename(columns=forecast_stems)
    gaps_before = fc.isna().sum()
    fc = fc.interpolate(method="time", limit=8)
    fc_grid = interpolate_to_grid(fc, grid, extrapolate=False)
    for column in fc_grid:
        report.weather_interpolated[column] = int(gaps_before.get(column, 0))
        remaining = fc_grid[column].isna()
        if remaining.any():
            # A missing archived run would have been covered operationally by
            # an earlier run; substituting actuals is a small, counted and
            # slightly optimistic repair.
            actual_column = column.replace("_fc", "")
            fc_grid.loc[remaining, column] = panel.loc[remaining, actual_column]
        report.forecast_fallback[column] = int(remaining.sum())
    fc_grid = make_physical(fc_grid)
    for column in fc_grid:
        panel[column] = fc_grid[column]

    panel = panel.astype(np.float32)
    panel["is_holiday"] = holiday_flag(grid).to_numpy()
    panel["demand_imputed"] = demand_imputed
    return panel, report
