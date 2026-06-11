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

#: Mapping from Open-Meteo variable names to panel column stems.
VARIABLE_STEMS = {
    "temperature_2m": "temp_c",
    "direct_normal_irradiance": "dni_wm2",
    "diffuse_radiation": "dhi_wm2",
}


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


def interpolate_to_grid(hourly: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate hourly weather onto the half-hourly grid.

    Time-based linear interpolation on the union of the two indices, then a
    reindex. Open-Meteo hourly radiation is a preceding-hour mean rather than
    an instantaneous value; treating it as instantaneous at the stamp is an
    approximation that is immaterial for regression features.
    """
    union = hourly.index.union(grid)
    return hourly.reindex(union).interpolate(method="time", limit_direction="both").reindex(grid)


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
    series = series.interpolate(method="time", limit=2)
    report.demand_interpolated = report.demand_missing - int(series.isna().sum())
    if series.isna().any():
        raise ValueError("demand has gaps longer than one hour; inspect the raw archives")

    flags = hampel_flags(series)
    report.demand_outliers = int(flags.sum())
    series = series.mask(flags).interpolate(method="time")

    panel = pd.DataFrame({"demand_mw": series})

    actuals = interpolate_to_grid(era5.rename(columns=VARIABLE_STEMS), grid)
    for column in actuals:
        panel[column] = actuals[column]

    forecast_stems = {
        f"{name}_previous_day{cfg.weather.lead_days}": f"{stem.split('_')[0]}_fc_{stem.split('_', 1)[1]}"
        for name, stem in VARIABLE_STEMS.items()
    }
    fc = forecast.rename(columns=forecast_stems)
    gaps_before = fc.isna().sum()
    fc = fc.interpolate(method="time", limit=8)
    fc_grid = interpolate_to_grid(fc, grid)
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
        panel[column] = fc_grid[column]

    panel = panel.astype(np.float32)
    panel["is_holiday"] = holiday_flag(grid).to_numpy()
    return panel, report
