"""Open-Meteo weather loaders: ERA5 actuals and archived operational forecasts.

One provider supplies both sides of the train/serve story. Ground truth is
ERA5 reanalysis from the Historical Weather API (``era5_seamless``, which
appends the preliminary ERA5T tail for recent months). Forecasts as issued
come from the Previous Runs API, which archives operational model output at
fixed lead-time offsets; with ``previous_day1`` each timestamp carries the
value predicted for it by the run initialised one day earlier, so using it
for day-ahead covariates introduces no look-ahead.

Open-Meteo data are CC BY 4.0 and free for non-commercial use. Attribution
lives in ``data/README.md`` and the project README.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
_CHUNK_DAYS = 180
_RETRIES = 4


def _get_json(url: str, params: dict) -> dict:
    """GET with simple exponential backoff for transient failures."""
    for attempt in range(_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == _RETRIES - 1:
                raise
            time.sleep(2.0**attempt)
    raise AssertionError("unreachable")


def _chunks(start: date, end: date) -> list[tuple[date, date]]:
    spans = []
    lower = start
    while lower <= end:
        upper = min(lower + timedelta(days=_CHUNK_DAYS - 1), end)
        spans.append((lower, upper))
        upper_next = upper + timedelta(days=1)
        lower = upper_next
    return spans


def _fetch_hourly(
    url: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    hourly: list[str],
    model: str,
) -> pd.DataFrame:
    """Fetch hourly series in UTC, chunked to keep responses modest."""
    frames = []
    for lower, upper in _chunks(start, end):
        payload = _get_json(
            url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": lower.isoformat(),
                "end_date": upper.isoformat(),
                "hourly": ",".join(hourly),
                "models": model,
                "timezone": "UTC",
            },
        )
        block = pd.DataFrame(payload["hourly"])
        block["time"] = pd.to_datetime(block["time"], utc=True)
        frames.append(block.set_index("time"))
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def fetch_era5(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    variables: tuple[str, ...],
    model: str = "era5_seamless",
) -> pd.DataFrame:
    """Fetch hourly ERA5 reanalysis actuals.

    Parameters
    ----------
    latitude, longitude
        Grid point of interest.
    start, end
        Inclusive UTC date range.
    variables
        Open-Meteo hourly variable names, for example ``("temperature_2m",)``.
    model
        Reanalysis model identifier.

    Returns
    -------
    pandas.DataFrame
        Hourly frame indexed by UTC timestamp, one column per variable.
    """
    return _fetch_hourly(ARCHIVE_API, latitude, longitude, start, end, list(variables), model)


def fetch_forecast(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    variables: tuple[str, ...],
    model: str = "bom_access_global",
    lead_days: int = 1,
) -> pd.DataFrame:
    """Fetch archived operational forecasts at a fixed lead-time offset.

    Parameters
    ----------
    latitude, longitude
        Grid point of interest.
    start, end
        Inclusive UTC date range of forecast valid times.
    variables
        Base hourly variable names; the API suffix for the lead is appended.
    model
        Forecast model identifier, by default the Bureau's ACCESS-G.
    lead_days
        Previous-runs offset in days. One day matches the day-ahead task.

    Returns
    -------
    pandas.DataFrame
        Hourly frame indexed by UTC valid time, one column per variable named
        ``{variable}_previous_day{lead_days}``. Gaps appear as NaN when a
        model run is missing from the archive.
    """
    hourly = [f"{variable}_previous_day{lead_days}" for variable in variables]
    return _fetch_hourly(PREVIOUS_RUNS_API, latitude, longitude, start, end, hourly, model)


def save_raw(frame: pd.DataFrame, path: Path) -> None:
    """Write a raw weather frame to parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw weather frame written by :func:`save_raw`."""
    return pd.read_parquet(path)
