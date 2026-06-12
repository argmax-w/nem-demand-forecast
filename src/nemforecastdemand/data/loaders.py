"""Loaders and schema checks for the processed half-hourly panel.

The processed data are two committed parquet files: ``panel.parquet``, the
full contiguous half-hourly panel, and ``split_labels.parquet``, one label
per timestamp marking the training block and the monthly validation and test
windows. Storing the panel whole keeps it on a strict grid that the
time-series models and lag features need, while the labels carve out the
season-blocked splits, which are deliberately not contiguous.

Storage convention: all indices are UTC period-start timestamps. AEST exists
only at the display layer and local Sydney clock time only inside calendar
feature construction.
"""

from __future__ import annotations

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


def validate_panel(frame: pd.DataFrame, name: str = "panel", require_grid: bool = True) -> None:
    """Validate a processed panel against the project schema.

    Parameters
    ----------
    frame
        Panel indexed by UTC period-start timestamps.
    name
        Label used in error messages.
    require_grid
        Enforce a strict half-hourly grid. True for the full panel; False
        for the season-blocked splits, which are deliberately not
        contiguous.

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
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{name}: index must be strictly increasing")
    if require_grid:
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


def load_panel(processed_dir: Path) -> pd.DataFrame:
    """Load and validate the full contiguous half-hourly panel."""
    frame = pd.read_parquet(processed_dir / "panel.parquet")
    validate_panel(frame, "panel", require_grid=True)
    return frame


def load_split_labels(processed_dir: Path) -> pd.Series:
    """Load the per-timestamp split labels aligned to the panel index."""
    return pd.read_parquet(processed_dir / "split_labels.parquet")["split"]


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the three season-blocked splits as slices of the full panel.

    The panel is validated as a strict grid; each split is validated for
    schema and ordering but not contiguity, since validation and test are
    unions of disjoint monthly windows. The training block strictly precedes
    every evaluation timestamp, which is the core no-leakage guarantee.
    """
    panel = load_panel(processed_dir)
    labels = load_split_labels(processed_dir).reindex(panel.index)
    splits = {}
    for name in SPLIT_NAMES:
        frame = panel.loc[labels == name]
        validate_panel(frame, name, require_grid=False)
        splits[name] = frame
    if splits["train"].index[-1] >= splits["validation"].index[0]:
        raise ValueError("training block must strictly precede the evaluation windows")
    if splits["train"].index[-1] >= splits["test"].index[0]:
        raise ValueError("training block must strictly precede the evaluation windows")
    return splits
