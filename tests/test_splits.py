"""Split integrity: no leakage, day-boundary cuts, qualifying origins."""

import pandas as pd

from nemforecastdemand.data.loaders import MARKET_TZ
from nemforecastdemand.splits import (
    chronological_split,
    horizon_index,
    market_day_starts,
    rolling_origins,
)


def make_index(days: int) -> pd.DatetimeIndex:
    start = pd.Timestamp("2025-06-01", tz=MARKET_TZ).tz_convert("UTC")
    return pd.date_range(start, periods=days * 48, freq="30min")


def test_chronological_split_is_disjoint_contiguous_and_ordered():
    index = make_index(40)
    splits = chronological_split(index, train=0.7, validation=0.15)
    union = splits["train"].union(splits["validation"]).union(splits["test"])
    assert union.equals(index)
    assert splits["train"][-1] < splits["validation"][0] < splits["test"][0]
    for name in ("validation", "test"):
        market = splits[name][0].tz_convert(MARKET_TZ)
        assert (market.hour, market.minute) == (0, 0)
    assert abs(len(splits["train"]) / len(index) - 0.7) < 1 / 40
    assert abs(len(splits["validation"]) / len(index) - 0.15) < 1 / 40


def test_rolling_origins_respect_horizon_and_lags():
    index = make_index(10)
    splits = chronological_split(index, train=0.5, validation=0.2)
    origins = rolling_origins(
        splits["test"],
        history_index=index,
        origin_times=("00:00", "12:00"),
        horizon=48,
        max_lag=336,
    )
    market = origins.tz_convert(MARKET_TZ)
    assert set(zip(market.hour, market.minute, strict=True)) <= {(0, 0), (12, 0)}
    # Three test days support two half-day origins each, minus the final
    # 12:00 whose horizon would run past the end of the split.
    assert len(origins) == 5
    for origin in origins:
        steps = horizon_index(origin, 48)
        assert steps.isin(splits["test"]).all()
        assert (origin - pd.Timedelta("30min") * 336) >= index[0]


def test_market_day_starts_finds_midnights():
    index = make_index(3)
    starts = market_day_starts(index)
    assert len(starts) == 3
    assert (starts.tz_convert(MARKET_TZ).hour == 0).all()


def test_horizon_index_shape():
    steps = horizon_index(pd.Timestamp("2025-06-05 14:00", tz="UTC"), 48)
    assert len(steps) == 48
    assert steps[-1] - steps[0] == pd.Timedelta("23.5h")
