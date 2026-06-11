"""Loaders and schema checks for the processed half-hourly panel.

The processed splits are small parquet files committed to the repository so
results are reproducible without credentials. Every consumer goes through
:func:`load_split`, which enforces the schema before anything touches a model.

Storage convention: all indices are UTC period-start timestamps. AEST exists
only at the display layer and local Sydney clock time only inside calendar
feature construction.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

STORAGE_TZ = "UTC"
MARKET_TZ = "Australia/Brisbane"
SPLIT_NAMES = ("train", "validation", "test")

#: Required columns of the processed panel and their dtypes. The ``_fc``
#: columns hold the day-ahead forecast issued one day earlier.
PANEL_SCHEMA: dict[str, str] = {
    "demand_mw": "float32",
    "temp_c": "float32",
    "dew_c": "float32",
    "dni_wm2": "float32",
    "dhi_wm2": "float32",
    "temp_fc_c": "float32",
    "dew_fc_c": "float32",
    "dni_fc_wm2": "float32",
    "dhi_fc_wm2": "float32",
    "is_holiday": "bool",
}


def validate_panel(frame: pd.DataFrame, name: str = "panel") -> None:
    """Validate a processed panel against the project schema.

    Parameters
    ----------
    frame
        Panel indexed by UTC period-start timestamps.
    name
        Label used in error messages.

    Raises
    ------
    ValueError
        On any schema, index or range violation.
    """
    missing = set(PANEL_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    for column, dtype in PANEL_SCHEMA.items():
        if str(frame[column].dtype) != dtype:
            raise ValueError(
                f"{name}: column {column} has dtype {frame[column].dtype}, expected {dtype}"
            )
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != STORAGE_TZ:
        raise ValueError(f"{name}: index must be a DatetimeIndex in {STORAGE_TZ}")
    deltas = np.diff(index.to_numpy())
    if len(frame) > 1 and not (deltas == np.timedelta64(30, "m")).all():
        raise ValueError(f"{name}: index is not a strict half-hourly grid")
    numeric = frame.drop(columns="is_holiday")
    if numeric.isna().any().any():
        raise ValueError(f"{name}: numeric columns contain missing values")
    if (frame["demand_mw"] <= 0).any():
        raise ValueError(f"{name}: demand must be positive")
    temps = numeric[["temp_c", "temp_fc_c", "dew_c", "dew_fc_c"]]
    if ((temps < -25) | (temps > 55)).any().any():
        raise ValueError(f"{name}: temperatures outside a plausible range")
    irradiance = numeric[["dni_wm2", "dhi_wm2", "dni_fc_wm2", "dhi_fc_wm2"]]
    if ((irradiance < -1e-3) | (irradiance > 1500)).any().any():
        raise ValueError(f"{name}: irradiance outside a plausible range")


def load_split(name: str, processed_dir: Path) -> pd.DataFrame:
    """Load and validate one processed split.

    Parameters
    ----------
    name
        One of ``train``, ``validation`` or ``test``.
    processed_dir
        Directory holding the committed parquet splits.

    Returns
    -------
    pandas.DataFrame
        The validated panel for the split.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}, expected one of {SPLIT_NAMES}")
    frame = pd.read_parquet(processed_dir / f"{name}.parquet")
    validate_panel(frame, name)
    return frame


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all three splits and check they are contiguous and disjoint."""
    splits = {name: load_split(name, processed_dir) for name in SPLIT_NAMES}
    for earlier, later in pairwise(SPLIT_NAMES):
        gap = splits[later].index[0] - splits[earlier].index[-1]
        if gap != pd.Timedelta("30min"):
            raise ValueError(f"splits {earlier} and {later} are not contiguous (gap {gap})")
    return splits
