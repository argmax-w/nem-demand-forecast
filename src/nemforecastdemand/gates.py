"""Data gates: refuse to fit on poisoned inputs, refuse to report nonsense.

Two automatic checks bracket every model. The input gate runs before each fit
(and before any forecast design is built) and fails hard: bad data never
reaches a model. The output gate runs on every forecast produced and withholds
anything implausible rather than reporting it. Both draw their physical limits
from one table, ``PHYSICAL_BOUNDS``, which the imputation in
:mod:`nemforecastdemand.features.preprocessing` also clips to, so the bounds
the gate enforces and the bounds the cleaner repairs to can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Physical limits per canonical column stem (forecast ``_fc`` variants share
#: them). Generous hard limits that only a telemetry fault would breach, not
#: operational extremes: NSW demand has never approached 20 GW, temperatures
#: stay well inside the range, irradiance and wind are non-negative.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "demand_mw": (0.0, 20000.0),
    "temp_c": (-15.0, 55.0),
    "dew_c": (-25.0, 35.0),
    "apptemp_c": (-25.0, 60.0),
    "dni_wm2": (0.0, 1200.0),
    "dhi_wm2": (0.0, 800.0),
    "ghi_wm2": (0.0, 1400.0),
    "wind_kmh": (0.0, 160.0),
}

#: Columns that must vary; a constant series here signals a stuck feed.
_NONCONSTANT = set(PHYSICAL_BOUNDS)

#: A continuous predictive law for a bounded quantity always places a sliver of
#: mass past a hard physical limit; up to this fraction of sample draws may sit
#: out of bounds before the forecast is judged implausible rather than tailed.
SAMPLE_TAIL_TOLERANCE = 0.01


class GateError(ValueError):
    """Raised when input data fails a hard gate, before any model sees it."""


def bounds_for(column: str) -> tuple[float, float] | None:
    """Physical bounds for a panel column, matching forecast variants."""
    return PHYSICAL_BOUNDS.get(column.replace("_fc", ""))


def validate_inputs(
    panel: pd.DataFrame, index: pd.DatetimeIndex | None = None, *, name: str = "panel"
) -> None:
    """Fail hard if the data going into a fit is unfit to model.

    Checks every column for numeric dtype, finiteness (no NaN or inf), and
    physical bounds; checks that bounded columns are not stuck constant; and
    checks that the time index is a complete, strictly increasing half-hourly
    grid. Raises :class:`GateError` listing every violation at once.
    """
    frame = panel if index is None else panel.loc[index]
    issues: list[str] = []

    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        issues.append("index is not a tz-aware DatetimeIndex")
    else:
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            issues.append("index is not strictly increasing")
        diffs = idx[1:] - idx[:-1]
        if len(diffs) and not (diffs == pd.Timedelta("30min")).all():
            issues.append("index is not a complete 30-minute grid")

    for col in frame.columns:
        series = frame[col]
        if series.dtype == bool:
            continue  # boolean flags (holiday, imputed) have no bounds
        if not pd.api.types.is_numeric_dtype(series):
            issues.append(f"{col}: non-numeric dtype {series.dtype}")
            continue
        values = series.to_numpy(dtype=np.float64)
        n_nan = int(np.isnan(values).sum())
        if n_nan:
            issues.append(f"{col}: {n_nan} NaN")
        if np.isinf(values).any():
            issues.append(f"{col}: contains inf")
        bounds = bounds_for(col)
        if bounds is not None and np.isfinite(values).any():
            lo, hi = bounds
            below, above = int((values < lo).sum()), int((values > hi).sum())
            if below or above:
                issues.append(f"{col}: {below + above} outside [{lo:g}, {hi:g}]")
            if col.replace("_fc", "") in _NONCONSTANT and np.nanstd(values) == 0.0:
                issues.append(f"{col}: constant (stuck feed?)")

    if issues:
        raise GateError(f"input gate failed for {name}:\n  " + "\n  ".join(issues))


def validate_forecast(
    mean: np.ndarray | None = None,
    *,
    sd: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    quantiles: np.ndarray | None = None,
    name: str = "forecast",
) -> list[str]:
    """Return the reasons a forecast should be withheld, empty if it is sound.

    Whatever representation is supplied is checked for finiteness, for demand
    within physical bounds, and for internal consistency: a positive predictive
    scale, and quantile levels that do not cross. The caller withholds any
    forecast with a non-empty list rather than reporting it.

    Reported point summaries (``mean``, ``quantiles``) must sit inside the
    bounds outright. A bundle of ``samples`` (draws on the first axis) is judged
    on what it would report, not on each draw: every point's predictive median
    must be physical and at most :data:`SAMPLE_TAIL_TOLERANCE` of the mass may
    fall out of bounds. A heteroskedastic Gaussian night trough leaves a handful
    of draws below zero, which is a tail and not nonsense; a wholesale-broken
    forecast moves the median or the bulk out of bounds and is withheld.
    """
    lo, hi = PHYSICAL_BOUNDS["demand_mw"]
    issues: list[str] = []

    def check_block(values: np.ndarray, label: str) -> None:
        if not np.isfinite(values).all():
            issues.append(f"{name} {label}: non-finite values")
            return
        if (values < lo).any() or (values > hi).any():
            issues.append(f"{name} {label}: outside [{lo:g}, {hi:g}] MW")

    if mean is not None:
        check_block(np.asarray(mean, dtype=np.float64), "mean")
    if sd is not None:
        sd = np.asarray(sd, dtype=np.float64)
        if not np.isfinite(sd).all() or (sd <= 0).any():
            issues.append(f"{name} sd: not strictly positive")
    if samples is not None:
        samples = np.asarray(samples, dtype=np.float64)
        if not np.isfinite(samples).all():
            issues.append(f"{name} samples: non-finite values")
        else:
            median = np.median(samples, axis=0)
            if (median < lo).any() or (median > hi).any():
                issues.append(f"{name} samples: median outside [{lo:g}, {hi:g}] MW")
            out_of_bounds = float(np.mean((samples < lo) | (samples > hi)))
            if out_of_bounds > SAMPLE_TAIL_TOLERANCE:
                issues.append(
                    f"{name} samples: {out_of_bounds:.1%} of mass outside [{lo:g}, {hi:g}] MW"
                )
    if quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        check_block(quantiles, "quantiles")
        # quantiles are stacked level-first; sorting must not change them
        if np.diff(quantiles, axis=0).min(initial=0.0) < -1e-6:
            issues.append(f"{name} quantiles: cross (not monotone in level)")
    return issues


def check_forecast(**kwargs) -> None:
    """Run the output gate and raise if the forecast should be withheld.

    A thin wrapper over :func:`validate_forecast` for the fit scripts, which
    must not write a nonsense artifact; the operational path uses
    ``validate_forecast`` directly to drop a single bad forecast instead.
    """
    issues = validate_forecast(**kwargs)
    if issues:
        raise GateError("output gate withheld a forecast:\n  " + "\n  ".join(issues))
