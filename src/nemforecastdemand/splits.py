"""Season-blocked splits and rolling forecast origins.

Everything strictly before the evaluation start is one contiguous training
block, so the time-series likelihoods fit on an unbroken series and no
fitting target ever sits after an evaluation point. The evaluation year is
carved into monthly pairs of fixed day windows, and a balanced seeded draw
sends one window of each month to validation and the other to test, so both
sets span every season and share the same position-in-month distribution.
Forecast origins are the configured market-clock issue times (00:00 and
12:00 AEST); each forecasts the next 48 half hours, and an origin qualifies
only when its full horizon and its longest demand lag lie inside the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemforecastdemand.config import Splits
from nemforecastdemand.data.loaders import MARKET_TZ, SPLIT_NAMES


def market_day_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Timestamps in ``index`` that fall on a market-day boundary."""
    market = index.tz_convert(MARKET_TZ)
    return index[(market.hour == 0) & (market.minute == 0)]


def _eval_months(market: pd.DatetimeIndex, eval_start: pd.Timestamp) -> list[tuple[int, int]]:
    """Distinct (year, month) pairs in the evaluation period, in order."""
    after = market[market >= eval_start]
    periods = pd.PeriodIndex(after, freq="M").unique()
    return [(p.year, p.month) for p in periods]


def season_blocked_split(index: pd.DatetimeIndex, splits: Splits) -> dict[str, pd.DatetimeIndex]:
    """Split a half-hourly index into a training block and monthly eval blocks.

    Parameters
    ----------
    index
        The full UTC half-hourly grid.
    splits
        Split configuration: evaluation start, the two day windows and the
        seed for the balanced validation/test assignment.

    Returns
    -------
    dict
        ``{"train": ..., "validation": ..., "test": ...}`` index slices.
        ``train`` is contiguous; ``validation`` and ``test`` are unions of
        disjoint monthly windows. Evaluation-year days outside the windows
        belong to no split.
    """
    market = index.tz_convert(MARKET_TZ)
    eval_start = pd.Timestamp(splits.eval_start, tz=MARKET_TZ)
    train_mask = market < eval_start

    early_lo, early_hi = splits.early_window
    late_lo, late_hi = splits.late_window
    day = market.day
    # Only months whose data fully covers both windows take part, so a
    # partial month at the end of the record is left out rather than
    # unbalancing the validation/test assignment.
    months = [
        (year, month)
        for (year, month) in _eval_months(market, eval_start)
        if ((market.year == year) & (market.month == month) & (day == early_lo)).any()
        and ((market.year == year) & (market.month == month) & (day == late_hi)).any()
    ]
    # Balanced assignment: validation takes the early window in exactly half
    # the months (rounded down) and the late window in the rest, so neither
    # set is biased towards the start or the end of the month.
    rng = np.random.default_rng(splits.seed)
    val_early = np.zeros(len(months), dtype=bool)
    val_early[: len(months) // 2] = True
    val_early = rng.permutation(val_early)

    val_mask = np.zeros(len(index), dtype=bool)
    test_mask = np.zeros(len(index), dtype=bool)
    for (year, month), v_early in zip(months, val_early, strict=True):
        in_month = (market.year == year) & (market.month == month)
        early = in_month & (day >= early_lo) & (day <= early_hi)
        late = in_month & (day >= late_lo) & (day <= late_hi)
        val_window, test_window = (early, late) if v_early else (late, early)
        val_mask |= np.asarray(val_window)
        test_mask |= np.asarray(test_window)

    return {
        "train": index[train_mask],
        "validation": index[val_mask],
        "test": index[test_mask],
    }


def split_labels(index: pd.DatetimeIndex, splits: dict[str, pd.DatetimeIndex]) -> pd.Series:
    """Per-timestamp split label, ``none`` for evaluation days in no window."""
    labels = pd.Series("none", index=index, dtype="object")
    for name in SPLIT_NAMES:
        labels.loc[splits[name]] = name
    return labels


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
