"""Shared input preparation for the Bayesian structural time-series model.

The BSTS model itself lives in :mod:`nemforecastdemand.models.innovations`: a
seasonal regression on the shared design with a stationary AR(2) error and a
heteroskedastic observation scale, fitted by ADVI and NUTS. This module holds
the standardisation the model and its prediction code share.

The target and both designs are standardised on the fitting window only, so
later data never leaks into the scalers. The mean design uses actual weather
over history, which has realised by the time a forecast is issued; horizon
designs are built per weather variant at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nemforecastdemand.config import Config
from nemforecastdemand.models.base import build_design, variance_design


@dataclass
class BstsInputs:
    """Standardised arrays for fitting and prediction.

    Scalers are stored so everything returns to megawatts.
    """

    index: pd.DatetimeIndex
    y: np.ndarray
    x_mean: np.ndarray
    x_var: np.ndarray
    y_loc: float
    y_scale: float
    x_loc: np.ndarray
    x_scale: np.ndarray
    xv_loc: np.ndarray
    xv_scale: np.ndarray
    columns: list[str]


def prepare_inputs(panel: pd.DataFrame, cfg: Config, fit_index: pd.DatetimeIndex) -> BstsInputs:
    """Standardise the target and designs on the fitting window.

    Parameters
    ----------
    panel
        Full processed panel (contiguous splits concatenated).
    cfg
        Project configuration.
    fit_index
        The window the model is fitted on; scalers come from here only, so
        later data never leaks into the standardisation.
    """
    design = build_design(panel, cfg, weather_source="actual").loc[fit_index]
    if design.isna().any().any():
        raise ValueError("fit window starts inside the demand-lag warmup")
    vdesign = variance_design(panel, cfg, weather_source="actual").loc[fit_index]
    y = panel["demand_mw"].loc[fit_index].to_numpy(dtype=np.float64)

    y_loc, y_scale = float(y.mean()), float(y.std())
    x_loc = design.mean().to_numpy()
    x_scale = design.std().replace(0.0, 1.0).to_numpy()
    xv_loc = vdesign.mean().to_numpy()
    xv_scale = vdesign.std().replace(0.0, 1.0).to_numpy()

    return BstsInputs(
        index=fit_index,
        y=((y - y_loc) / y_scale).astype(np.float32),
        x_mean=((design.to_numpy() - x_loc) / x_scale).astype(np.float32),
        x_var=((vdesign.to_numpy() - xv_loc) / xv_scale).astype(np.float32),
        y_loc=y_loc,
        y_scale=y_scale,
        x_loc=x_loc,
        x_scale=x_scale,
        xv_loc=xv_loc,
        xv_scale=xv_scale,
        columns=list(design.columns),
    )


def transform_design(inputs: BstsInputs, design: pd.DataFrame, vdesign: pd.DataFrame):
    """Standardise out-of-window designs with the stored fit-window scalers."""
    x = ((design.to_numpy() - inputs.x_loc) / inputs.x_scale).astype(np.float32)
    z = ((vdesign.to_numpy() - inputs.xv_loc) / inputs.xv_scale).astype(np.float32)
    return x, z
