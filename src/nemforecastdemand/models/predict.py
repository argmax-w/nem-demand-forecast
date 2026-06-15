"""Posterior-predictive forecasting shared by the ADVI and NUTS scripts.

Takes hyperparameter draws from either inference path and produces predictive
path samples for every test origin under every weather-input variant. The
AR(2) error is second-order Markov, so each draw only needs its regression
residual at the two steps before each origin; the horizons are then simulated
in closed form from the AR(2) impulse response.

Horizon designs for the perturbation sweep are rebuilt per origin because
the 00:00 and 12:00 horizons overlap in time with different error draws.
Only the weather and degree-day columns change, so the rebuild is column
surgery on the precomputed actual-weather design rather than a full design
pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemforecastdemand.config import Config
from nemforecastdemand.features.weather import degree_days
from nemforecastdemand.models import bsts, innovations
from nemforecastdemand.models.base import build_design, perturbation_overrides, variance_design
from nemforecastdemand.splits import horizon_index

WEATHER_DESIGN_COLUMNS = ["temp_c", "dew_c", "dni_wm2", "dhi_wm2"]


def _apply_overrides(
    block: pd.DataFrame,
    vblock: pd.DataFrame,
    overrides: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = block.copy()
    block.loc[:, overrides.columns] = overrides
    degrees = degree_days(block["temp_c"], cfg.weather.heating_base, cfg.weather.cooling_base)
    block.loc[:, ["cooling_deg", "heating_deg"]] = degrees

    vblock = vblock.copy()
    if cfg.bsts.variance_use_degree_days:
        vblock.loc[:, ["cooling_deg", "heating_deg"]] = degrees
    return block, vblock


def predict_variants_innovations(
    hyper_draws: dict[str, np.ndarray],
    inputs: bsts.BstsInputs,
    panel: pd.DataFrame,
    cfg: Config,
    origins: pd.DatetimeIndex,
    perturbations: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Predictive path samples per variant for the innovations-form model.

    The AR(2) error is second-order Markov, so each draw only needs its
    regression residuals at the two steps before each origin; the horizons
    then follow in closed form from the AR(2) impulse response.
    """
    history = panel.index[
        (panel.index >= inputs.index[0]) & (panel.index <= origins[-1] + pd.Timedelta("23.5h"))
    ]
    design_actual = build_design(panel, cfg, weather_source="actual").loc[history]
    vdesign_actual = variance_design(panel, cfg, weather_source="actual").loc[history]
    design_forecast = build_design(panel, cfg, weather_source="forecast").loc[history]
    vdesign_forecast = variance_design(panel, cfg, weather_source="forecast").loc[history]

    x_hist, _ = bsts.transform_design(inputs, design_actual, vdesign_actual)
    y_hist = ((panel["demand_mw"].loc[history].to_numpy() - inputs.y_loc) / inputs.y_scale).astype(
        np.float32
    )
    positions = history.get_indexer(origins)
    e_origin = innovations.origin_residuals(hyper_draws, y_hist, x_hist, positions)

    horizons = [horizon_index(origin, cfg.horizon) for origin in origins]
    y_true = np.stack(
        [panel["demand_mw"].loc[index].to_numpy(dtype=np.float32) for index in horizons]
    )

    def simulate(blocks: list[pd.DataFrame], vblocks: list[pd.DataFrame], seed: int) -> np.ndarray:
        transformed = [
            bsts.transform_design(inputs, b, v) for b, v in zip(blocks, vblocks, strict=True)
        ]
        x_future = np.stack([x for x, _ in transformed])
        z_future = np.stack([z for _, z in transformed])
        paths = innovations.simulate_horizon_paths(
            hyper_draws, e_origin, x_future, z_future, cfg.bsts, seed=seed
        )
        return (paths * inputs.y_scale + inputs.y_loc).astype(np.float32)

    variants: dict[str, np.ndarray] = {}
    variants["forecast"] = simulate(
        [design_forecast.loc[index] for index in horizons],
        [vdesign_forecast.loc[index] for index in horizons],
        seed=cfg.seed + 1,
    )
    variants["actual"] = simulate(
        [design_actual.loc[index] for index in horizons],
        [vdesign_actual.loc[index] for index in horizons],
        seed=cfg.seed + 2,
    )
    for j, multiplier in enumerate(cfg.perturbation.sweep_multipliers):
        if multiplier == 0:
            continue
        blocks, vblocks = [], []
        for index in horizons:
            overrides = perturbation_overrides(panel, index, perturbations, multiplier, cfg.seed)
            block, vblock = _apply_overrides(
                design_actual.loc[index], vdesign_actual.loc[index], overrides, cfg
            )
            blocks.append(block)
            vblocks.append(vblock)
        variants[f"perturb_{multiplier:g}"] = simulate(blocks, vblocks, seed=cfg.seed + 3 + j)
    return variants, y_true


def variance_decomposition_innovations(
    hyper_draws: dict[str, np.ndarray],
    inputs: bsts.BstsInputs,
    panel: pd.DataFrame,
    cfg: Config,
    origins: pd.DatetimeIndex,
    weather_source: str = "forecast",
) -> dict[str, np.ndarray]:
    """Aleatoric and epistemic split for the innovations-form model, in MW².

    Two exact components: ``parameter`` (epistemic; the origin residuals are
    observed, so there is no state term) and ``innovation`` (aleatoric; the
    accumulated AR-carried noise).
    """
    history = panel.index[
        (panel.index >= inputs.index[0]) & (panel.index <= origins[-1] + pd.Timedelta("23.5h"))
    ]
    design_actual = build_design(panel, cfg, weather_source="actual").loc[history]
    vdesign_actual = variance_design(panel, cfg, weather_source="actual").loc[history]
    x_hist, _ = bsts.transform_design(inputs, design_actual, vdesign_actual)
    y_hist = ((panel["demand_mw"].loc[history].to_numpy() - inputs.y_loc) / inputs.y_scale).astype(
        np.float32
    )
    positions = history.get_indexer(origins)
    e_origin = innovations.origin_residuals(hyper_draws, y_hist, x_hist, positions)

    horizons = [horizon_index(origin, cfg.horizon) for origin in origins]
    design_h = build_design(panel, cfg, weather_source=weather_source).loc[history]
    vdesign_h = variance_design(panel, cfg, weather_source=weather_source).loc[history]
    transformed = [
        bsts.transform_design(inputs, design_h.loc[index], vdesign_h.loc[index])
        for index in horizons
    ]
    x_future = np.stack([x for x, _ in transformed])
    z_future = np.stack([z for _, z in transformed])

    parts = innovations.decompose_horizon_variance(
        hyper_draws, e_origin, x_future, z_future, cfg.bsts
    )
    return {name: value * inputs.y_scale**2 for name, value in parts.items()}


PERTURBATION_VARIABLES = (
    ("temp_c", "temp_fc_c", False),
    ("dew_c", "dew_fc_c", False),
    ("dni_wm2", "dni_fc_wm2", True),
    ("dhi_wm2", "dhi_fc_wm2", True),
)


def fit_perturbation_models(panel: pd.DataFrame, train_index: pd.DatetimeIndex) -> dict:
    """Calibrate the perturbation models on training data only.

    Each variable uses the training rows where a genuine archived forecast
    exists, that is where the forecast column differs from the actual. The
    early training period predates the forecast archive and carries actuals
    in the forecast columns; including it would understate the day-ahead
    error and flatten the robustness sweep, so it is dropped here.
    """
    from nemforecastdemand.features.weather import fit_perturbation

    models = {}
    for actual_col, fc_col, nonnegative in PERTURBATION_VARIABLES:
        actual = panel[actual_col].loc[train_index]
        forecast = panel[fc_col].loc[train_index]
        genuine = actual.to_numpy() != forecast.to_numpy()
        models[actual_col] = fit_perturbation(
            actual[genuine], forecast[genuine], nonnegative=nonnegative
        )
    return models
