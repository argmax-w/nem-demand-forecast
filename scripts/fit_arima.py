"""Fit the classical baseline and write its forecasts to artifacts.

Order selection on the validation set, a seasonal-basis assessment
(trigonometric against periodic RBF) at the chosen order, a final refit on
the full history before the test boundary and rolling-origin test
forecasts under every weather-input variant. Heavy steps land in
``artifacts/`` so the notebooks only read results.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from nemforecastdemand import gates
from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_panel, load_splits
from nemforecastdemand.evaluation.metrics import crps_gaussian
from nemforecastdemand.features.weather import fit_perturbation
from nemforecastdemand.models.arima import DynamicHarmonicRegression
from nemforecastdemand.models.base import Forecast, SeasonalNaive, run_variants
from nemforecastdemand.splits import rolling_origins
from nemforecastdemand.utils import save_artifact, timed

warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")


def mean_crps(forecasts: list[Forecast], panel: pd.DataFrame) -> float:
    scores = [
        crps_gaussian(panel["demand_mw"].loc[fc.index].to_numpy(), fc.mean, fc.sd).mean()
        for fc in forecasts
    ]
    return float(np.mean(scores))


def stack(forecasts: list[Forecast], field: str) -> np.ndarray:
    return np.stack([getattr(fc, field) for fc in forecasts]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_panel(cfg.paths.processed)
    gates.validate_inputs(panel)  # fail hard before fitting on poisoned data
    splits = load_splits(cfg.paths.processed)
    max_lag = max(cfg.features.demand_lags)

    # Selection trains on the training block and scores on validation; the
    # final fit uses the same training block, since the evaluation windows
    # are held out. The lag warmup is trimmed so the design has no missing
    # demand lags.
    train_fit = splits["train"].index[max_lag:]
    final_fit = train_fit
    val_origins = rolling_origins(
        splits["validation"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    timings: dict[str, float] = {}

    # Residual order selection on validation, headline weather inputs.
    selection = []
    for order in cfg.arima.candidate_orders:
        with timed(f"fit_{order}") as fit_time:
            model = DynamicHarmonicRegression(cfg, order).fit(panel, train_fit)
        forecasts = [model.forecast(panel, origin, "forecast") for origin in val_origins]
        selection.append(
            {
                "order": list(order),
                "val_crps_mw": mean_crps(forecasts, panel),
                "fit_seconds": fit_time.seconds,
                "llf": float(model.results.llf),
                "aic": float(model.results.aic),
            }
        )
        print(f"order {order}: validation CRPS {selection[-1]['val_crps_mw']:.1f} MW")
    chosen = tuple(min(selection, key=lambda row: row["val_crps_mw"])["order"])

    # Seasonal basis assessment at the chosen order, confirming the model-free
    # comparison in notebook 01. The trigonometric basis is retained unless
    # the RBF basis wins by more than one percent: differences inside that
    # band are noise, and the harmonics are the more parsimonious default.
    basis_scores = {}
    for basis in ("fourier", "rbf"):
        basis_cfg = replace(cfg, features=replace(cfg.features, seasonal_basis=basis))
        model = DynamicHarmonicRegression(basis_cfg, chosen).fit(panel, train_fit)
        forecasts = [model.forecast(panel, origin, "forecast") for origin in val_origins]
        basis_scores[basis] = mean_crps(forecasts, panel)
        print(f"basis {basis}: validation CRPS {basis_scores[basis]:.1f} MW")
    chosen_basis = "rbf" if basis_scores["rbf"] < 0.99 * basis_scores["fourier"] else "fourier"
    cfg = replace(cfg, features=replace(cfg.features, seasonal_basis=chosen_basis))

    # Final fit on the window ending at the test boundary.
    with timed("final_fit", timings):
        model = DynamicHarmonicRegression(cfg, chosen).fit(panel, final_fit)
    naive = SeasonalNaive().fit(panel, final_fit)

    # Validation forecasts from the chosen model, for calibration checks.
    val_model = DynamicHarmonicRegression(cfg, chosen).fit(panel, train_fit)
    val_forecasts = [val_model.forecast(panel, origin, "forecast") for origin in val_origins]

    # Perturbation models are calibrated on training data only.
    train_index = splits["train"].index
    perturbations = {
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

    with timed("test_forecasts", timings):
        variants = run_variants(
            model, panel, test_origins, perturbations, cfg.perturbation.sweep_multipliers, cfg.seed
        )
    with timed("naive_forecasts", timings):
        naive_forecasts = [naive.forecast(panel, origin) for origin in test_origins]

    arrays = {
        "origins_val": val_origins.asi8,
        "origins_test": test_origins.asi8,
        "y_val": np.stack(
            [panel["demand_mw"].loc[fc.index].to_numpy() for fc in val_forecasts]
        ).astype(np.float32),
        "y_test": np.stack(
            [panel["demand_mw"].loc[fc.index].to_numpy() for fc in variants["forecast"]]
        ).astype(np.float32),
        "val_mean": stack(val_forecasts, "mean"),
        "val_sd": stack(val_forecasts, "sd"),
        "naive_mean": stack(naive_forecasts, "mean"),
        "naive_sd": stack(naive_forecasts, "sd"),
    }
    for name, forecasts in variants.items():
        arrays[f"{name}_mean"] = stack(forecasts, "mean")
        arrays[f"{name}_sd"] = stack(forecasts, "sd")
    gates.check_forecast(mean=arrays["forecast_mean"], sd=arrays["forecast_sd"])

    meta = {
        "selection": selection,
        "chosen_order": list(chosen),
        "basis_scores_mw": basis_scores,
        "chosen_basis": chosen_basis,
        "timings_seconds": timings,
        "forecast_seconds_per_origin": timings["test_forecasts"]
        / (len(test_origins) * len(variants)),
        "naive_train_mae_mw": naive.train_mae,
        "n_exog": len(model.results.model.exog_names),
        "fit_window": [str(final_fit[0]), str(final_fit[-1])],
        "perturbation": {
            column: {"rho": p.rho, "sigma_mean": float(p.sigma_by_step.mean())}
            for column, p in perturbations.items()
        },
    }
    save_artifact(cfg.paths.artifacts / "arima", arrays, meta)
    print(f"chosen order {chosen} with {chosen_basis} basis; artifacts written")


if __name__ == "__main__":
    main()
