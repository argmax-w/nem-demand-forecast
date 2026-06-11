"""Download Open-Meteo weather for the configured grid point.

Fetches hourly ERA5 reanalysis actuals and archived ACCESS-G operational
forecasts (previous-runs, fixed one-day lead) into ``data/raw/weather``. The
window is padded by two days on each side so timezone conversion and
half-hourly interpolation have no edge effects.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from nemforecastdemand.config import load_config
from nemforecastdemand.data import weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = date.fromisoformat(cfg.window.start) - timedelta(days=2)
    end = date.fromisoformat(cfg.window.end) + timedelta(days=2)
    point = cfg.grid_point
    raw_dir = cfg.paths.raw / "weather"

    actuals = weather.fetch_era5(
        point.latitude,
        point.longitude,
        start,
        end,
        cfg.weather.variables,
        cfg.weather.actuals_model,
    )
    weather.save_raw(actuals, raw_dir / "era5.parquet")
    print(f"ERA5 actuals: {len(actuals)} hours, {actuals.index[0]} to {actuals.index[-1]}")

    forecast = weather.fetch_forecast(
        point.latitude,
        point.longitude,
        start,
        end,
        cfg.weather.variables,
        cfg.weather.forecast_model,
        cfg.weather.lead_days,
    )
    weather.save_raw(forecast, raw_dir / "forecast.parquet")
    missing = int(forecast.isna().any(axis=1).sum())
    print(
        f"{cfg.weather.forecast_model} previous-day{cfg.weather.lead_days} forecasts: "
        f"{len(forecast)} hours, {missing} with gaps"
    )


if __name__ == "__main__":
    main()
