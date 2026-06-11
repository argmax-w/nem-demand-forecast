"""Shared fixtures: synthetic panels on the UTC half-hourly grid."""

import numpy as np
import pandas as pd
import pytest

MARKET_TZ = "Australia/Brisbane"


def make_panel(start_market: str, days: int, seed: int = 0) -> pd.DataFrame:
    """Build a schema-conforming synthetic panel.

    Parameters
    ----------
    start_market
        First market-time day, ISO format.
    days
        Number of whole market days.
    seed
        Seed for the noise.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_market, tz=MARKET_TZ).tz_convert("UTC")
    index = pd.date_range(start, periods=days * 48, freq="30min", name="ts")
    step = np.arange(len(index))
    daily = np.sin(2 * np.pi * step / 48)
    temp = 18 + 6 * daily + rng.normal(0, 1, len(index))
    sun = np.maximum(0, 800 * np.sin(2 * np.pi * (step % 48) / 48 - np.pi / 2))
    frame = pd.DataFrame(
        {
            "demand_mw": 7000 + 900 * daily + rng.normal(0, 80, len(index)),
            "temp_c": temp,
            "dni_wm2": sun,
            "dhi_wm2": sun * 0.3,
            "temp_fc_c": temp + rng.normal(0, 0.8, len(index)),
            "dni_fc_wm2": np.maximum(sun + rng.normal(0, 40, len(index)), 0),
            "dhi_fc_wm2": np.maximum(sun * 0.3 + rng.normal(0, 15, len(index)), 0),
        },
        index=index,
    ).astype(np.float32)
    frame["is_holiday"] = False
    return frame


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel("2025-06-01", days=28)
