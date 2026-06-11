"""Chronological splits and rolling forecast origins.

Splits are cut at market-day boundaries with no shuffling, so train always
precedes validation precedes test and no half hour appears twice. Forecast
origins are the configured market-clock issue times (00:00 and 12:00 AEST);
each origin forecasts the next 48 half hours, and an origin only qualifies
when its full horizon and its longest demand lag both lie inside the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemforecastdemand.data.loaders import MARKET_TZ, SPLIT_NAMES


def market_day_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Timestamps in ``index`` that fall on a market-day boundary."""
    market = index.tz_convert(MARKET_TZ)
    return index[(market.hour == 0) & (market.minute == 0)]


def chronological_split(
    index: pd.DatetimeIndex,
    train: float,
    validation: float,
) -> dict[str, pd.DatetimeIndex]:
    """Split a half-hourly index chronologically at market-day boundaries.

    Parameters
    ----------
    index
        The full UTC half-hourly grid.
    train, validation
        Fractions of the window; the remainder is the test set. Cut points
        snap to the nearest market-day boundary so every split starts at
        00:00 market time.

    Returns
    -------
    dict
        ``{"train": ..., "validation": ..., "test": ...}`` index slices,
        disjoint and contiguous.
    """
    days = market_day_starts(index)
    n_days = len(days)
    train_days = round(train * n_days)
    validation_days = round(validation * n_days)
    first_validation = days[train_days]
    first_test = days[train_days + validation_days]
    return {
        "train": index[index < first_validation],
        "validation": index[(index >= first_validation) & (index < first_test)],
        "test": index[index >= first_test],
    }


def rolling_origins(
    scoring_index: pd.DatetimeIndex,
    history_index: pd.DatetimeIndex,
    origin_times: tuple[str, ...],
    horizon: int,
    max_lag: int = 0,
) -> pd.DatetimeIndex:
    """Qualifying forecast origins inside a scoring split.

    Parameters
    ----------
    scoring_index
        The split being scored, for example the test set.
    history_index
        The full panel index, used to check lag availability behind the
        origin.
    origin_times
        Market-clock issue times, for example ``("00:00", "12:00")``. The
        origin timestamp is the first forecast step: a forecast issued at
        12:00 covers the periods starting 12:00 through 11:30 next day.
    horizon
        Steps per forecast.
    max_lag
        Longest demand lag used as a feature, in steps.

    Returns
    -------
    pandas.DatetimeIndex
        Origins whose full horizon lies inside the scoring split and whose
        lagged history lies inside the panel.
    """
    market = scoring_index.tz_convert(MARKET_TZ)
    clock = market.strftime("%H:%M")
    candidates = scoring_index[np.isin(clock, origin_times)]

    last_step = candidates + pd.Timedelta("30min") * (horizon - 1)
    lag_start = candidates - pd.Timedelta("30min") * max_lag
    valid = last_step.isin(scoring_index) & lag_start.isin(history_index)
    return candidates[valid]


def horizon_index(origin: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """The half-hourly target index for one forecast origin."""
    return pd.date_range(origin, periods=horizon, freq="30min", tz="UTC")


def split_summary(splits: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Tabulate split extents for display, in market time."""
    rows = []
    for name in SPLIT_NAMES:
        index = splits[name]
        market = index.tz_convert(MARKET_TZ)
        rows.append(
            {
                "split": name,
                "first": market[0],
                "last": market[-1],
                "half_hours": len(index),
                "days": round(len(index) / 48, 1),
            }
        )
    return pd.DataFrame(rows).set_index("split")
