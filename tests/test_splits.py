"""Split integrity: contiguous train, all-season blocks, qualifying origins."""

import pandas as pd

from nemforecastdemand.config import Splits
from nemforecastdemand.data.loaders import MARKET_TZ
from nemforecastdemand.splits import (
    horizon_index,
    market_day_starts,
    rolling_origins,
    season_blocked_split,
    split_labels,
)


def make_index(start_market: str, days: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_market, tz=MARKET_TZ).tz_convert("UTC")
    return pd.date_range(start, periods=days * 48, freq="30min")


def make_splits_cfg(seed: int = 7) -> Splits:
    return Splits(
        eval_start="2025-01-01",
        early_window=(8, 12),
        late_window=(19, 23),
        seed=seed,
    )


def test_season_blocked_split_is_clean_and_all_season():
    # Train through December 2024, then four evaluation months.
    index = make_index("2024-09-01", days=270)
    splits = season_blocked_split(index, make_splits_cfg())
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))

    # Training block strictly precedes every evaluation timestamp.
    assert train[-1] < validation[0]
    assert train[-1] < test[0]
    eval_start = pd.Timestamp("2025-01-01", tz=MARKET_TZ)
    assert (train.tz_convert(MARKET_TZ) < eval_start).all()

    # Validation and test never overlap.
    assert len(validation.intersection(test)) == 0

    # Both sets span every evaluation month, one window each.
    market = index.tz_convert(MARKET_TZ)
    months = set(pd.PeriodIndex(market[market >= eval_start], freq="M").unique())
    for block in (validation, test):
        covered = set(pd.PeriodIndex(block.tz_convert(MARKET_TZ), freq="M").unique())
        assert covered == months

    # Every window falls inside one of the two configured day ranges.
    for block in (validation, test):
        days = block.tz_convert(MARKET_TZ).day
        in_early = (days >= 8) & (days <= 12)
        in_late = (days >= 19) & (days <= 23)
        assert (in_early | in_late).all()


def test_validation_slot_assignment_is_balanced():
    # An even number of complete evaluation months must split exactly half
    # early, half late for validation (and the complement for test). The
    # span below ends in July, so January to June are the six complete
    # evaluation months and the partial July is excluded.
    index = make_index("2024-09-01", days=305)
    splits = season_blocked_split(index, make_splits_cfg())
    market = splits["validation"].tz_convert(MARKET_TZ)
    frame = pd.DataFrame({"ym": pd.PeriodIndex(market, freq="M").astype(str), "day": market.day})
    first_day = frame.groupby("ym")["day"].min()
    early = (first_day <= 12).sum()
    late = (first_day >= 19).sum()
    assert early == late


def test_split_labels_cover_panel_without_overlap():
    index = make_index("2024-09-01", days=270)
    splits = season_blocked_split(index, make_splits_cfg())
    labels = split_labels(index, splits)
    assert labels.index.equals(index)
    assert set(labels.unique()) <= {"train", "validation", "test", "none"}
    for name in ("train", "validation", "test"):
        assert (labels.loc[splits[name]] == name).all()


def test_rolling_origins_respect_horizon_and_lags():
    index = make_index("2025-06-01", days=10)
    scoring = index[index.tz_convert(MARKET_TZ).day <= 7]
    origins = rolling_origins(
        scoring,
        history_index=index,
        origin_times=("00:00", "12:00"),
        horizon=48,
        max_lag=336,
    )
    market = origins.tz_convert(MARKET_TZ)
    assert set(zip(market.hour, market.minute, strict=True)) <= {(0, 0), (12, 0)}
    for origin in origins:
        steps = horizon_index(origin, 48)
        assert steps.isin(scoring).all()
        assert (origin - pd.Timedelta("30min") * 336) >= index[0]


def test_market_day_starts_finds_midnights():
    index = make_index("2025-06-01", days=3)
    starts = market_day_starts(index)
    assert len(starts) == 3
    assert (starts.tz_convert(MARKET_TZ).hour == 0).all()


def test_horizon_index_shape():
    steps = horizon_index(pd.Timestamp("2025-06-05 14:00", tz="UTC"), 48)
    assert len(steps) == 48
    assert steps[-1] - steps[0] == pd.Timedelta("23.5h")
