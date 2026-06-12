"""Posterior-predictive forecasting shared by the ADVI and NUTS scripts.

Takes hyperparameter draws from either inference path and produces predictive
path samples for every test origin under every weather-input variant, using
the Kalman machinery in :mod:`nemforecastdemand.models.bsts`. The expensive
filter pass runs once per posterior; each variant then reuses the filtered
states and only re-simulates the horizons.

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


def predict_variants(
    hyper_draws: dict[str, np.ndarray],
    inputs: bsts.BstsInputs,
    panel: pd.DataFrame,
    cfg: Config,
    origins: pd.DatetimeIndex,
    perturbations: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Predictive path samples per variant, in megawatts.

    Parameters
    ----------
    hyper_draws
        Hyperparameter draws with leading draw dimension.
    inputs
        The fitting inputs (scalers and fit window).
    panel
        Full processed panel.
    cfg
        Project configuration.
    origins
        Forecast origins, all later than the fit window start.
    perturbations
        Fitted perturbation models keyed by canonical weather column.

    Returns
    -------
    tuple
        ``(paths, y_true)``: a mapping from variant name to predictive paths
        of shape ``(S, O, 48)`` in MW, and the observed paths ``(O, 48)``.
    """
    history = panel.index[
        (panel.index >= inputs.index[0]) & (panel.index <= origins[-1] + pd.Timedelta("23.5h"))
    ]
    design_actual = build_design(panel, cfg, weather_source="actual").loc[history]
    vdesign_actual = variance_design(panel, cfg, weather_source="actual").loc[history]
    design_forecast = build_design(panel, cfg, weather_source="forecast").loc[history]
    vdesign_forecast = variance_design(panel, cfg, weather_source="forecast").loc[history]

    x_hist, z_hist = bsts.transform_design(inputs, design_actual, vdesign_actual)
    y_hist = ((panel["demand_mw"].loc[history].to_numpy() - inputs.y_loc) / inputs.y_scale).astype(
        np.float32
    )
    filtered_mean, filtered_cov = bsts.kalman_filter_states(
        hyper_draws, y_hist, x_hist, z_hist, cfg.bsts
    )
    positions = history.get_indexer(origins)

    horizons = [horizon_index(origin, cfg.horizon) for origin in origins]
    y_true = np.stack(
        [panel["demand_mw"].loc[index].to_numpy(dtype=np.float32) for index in horizons]
    )

    def simulate(blocks: list[pd.DataFrame], vblocks: list[pd.DataFrame], seed: int) -> np.ndarray:
        x_future = np.stack(
            [bsts.transform_design(inputs, b, v)[0] for b, v in zip(blocks, vblocks, strict=True)]
        )
        z_future = np.stack(
            [bsts.transform_design(inputs, b, v)[1] for b, v in zip(blocks, vblocks, strict=True)]
        )
        paths = bsts.simulate_horizon_paths(
            hyper_draws,
            filtered_mean,
            filtered_cov,
            positions,
            x_future,
            z_future,
            cfg.bsts,
            seed=seed,
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


def variance_decomposition(
    hyper_draws: dict[str, np.ndarray],
    inputs: bsts.BstsInputs,
    panel: pd.DataFrame,
    cfg: Config,
    origins: pd.DatetimeIndex,
    weather_source: str = "forecast",
) -> dict[str, np.ndarray]:
    """Aleatoric and epistemic split of the predictive variance, in MW².

    Wraps :func:`bsts.decompose_horizon_variance`: filters the history once
    per draw, then splits each origin's horizon variance into ``parameter``
    and ``state`` (epistemic: posterior uncertainty about hyperparameters
    and the latent state, which more data shrinks) and ``process`` and
    ``observation`` (aleatoric: future trend innovations and measurement
    noise, irreducible under the model). Components sum to the predictive
    variance of the simulated paths up to Monte Carlo error.

    Returns
    -------
    dict of numpy.ndarray
        The four components, each ``(O, H)`` in megawatts squared.
    """
    history = panel.index[
        (panel.index >= inputs.index[0]) & (panel.index <= origins[-1] + pd.Timedelta("23.5h"))
    ]
    design_actual = build_design(panel, cfg, weather_source="actual").loc[history]
    vdesign_actual = variance_design(panel, cfg, weather_source="actual").loc[history]
    x_hist, z_hist = bsts.transform_design(inputs, design_actual, vdesign_actual)
    y_hist = ((panel["demand_mw"].loc[history].to_numpy() - inputs.y_loc) / inputs.y_scale).astype(
        np.float32
    )
    filtered_mean, filtered_cov = bsts.kalman_filter_states(
        hyper_draws, y_hist, x_hist, z_hist, cfg.bsts
    )
    positions = history.get_indexer(origins)

    horizons = [horizon_index(origin, cfg.horizon) for origin in origins]
    design_h = build_design(panel, cfg, weather_source=weather_source).loc[history]
    vdesign_h = variance_design(panel, cfg, weather_source=weather_source).loc[history]
    transformed = [
        bsts.transform_design(inputs, design_h.loc[index], vdesign_h.loc[index])
        for index in horizons
    ]
    x_future = np.stack([x for x, _ in transformed])
    z_future = np.stack([z for _, z in transformed])

    parts = bsts.decompose_horizon_variance(
        hyper_draws, filtered_mean, filtered_cov, positions, x_future, z_future, cfg.bsts
    )
    return {name: value * inputs.y_scale**2 for name, value in parts.items()}


def predict_variants_innovations(
    hyper_draws: dict[str, np.ndarray],
    inputs: bsts.BstsInputs,
    panel: pd.DataFrame,
    cfg: Config,
    origins: pd.DatetimeIndex,
    perturbations: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Predictive path samples per variant for the innovations-form model.

    The AR(1) error is first-order Markov, so instead of a filter pass over
    the history each draw only needs its regression residual at the step
    before each origin; everything else mirrors :func:`predict_variants`.
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

    Two exact components instead of the trend models' four: ``parameter``
    (epistemic — the origin residual is observed, so there is no state
    term) and ``innovation`` (aleatoric — the accumulated AR-carried noise
    that the trend models split between process and observation).
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


def fit_perturbation_models(panel: pd.DataFrame, train_index: pd.DatetimeIndex) -> dict:
    """Calibrate the perturbation models on training data only."""
    from nemforecastdemand.features.weather import fit_perturbation

    return {
        "temp_c": fit_perturbation(
            panel["temp_c"].loc[train_index], panel["temp_fc_c"].loc[train_index]
        ),
        "dew_c": fit_perturbation(
            panel["dew_c"].loc[train_index], panel["dew_fc_c"].loc[train_index]
        ),
        "dni_wm2": fit_perturbation(
            panel["dni_wm2"].loc[train_index],
            panel["dni_fc_wm2"].loc[train_index],
            nonnegative=True,
        ),
        "dhi_wm2": fit_perturbation(
            panel["dhi_wm2"].loc[train_index],
            panel["dhi_fc_wm2"].loc[train_index],
            nonnegative=True,
        ),
    }
