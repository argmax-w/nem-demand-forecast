"""Fit the LightGBM quantile benchmark and write its forecasts to artifacts.

Heads are trained on origin blocks over the training split (the shared
design plus the origin-anchored recency features, mirroring the
operational setting) with early stopping on the validation split's
blocks (which the trees otherwise never see), then produce rolling-origin
test forecasts under every weather-input variant. Each model is run the
way a practitioner would run it.
"""

from __future__ import annotations

import argparse

import numpy as np

from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_panel, load_splits
from nemforecastdemand.evaluation.metrics import crps_from_quantiles
from nemforecastdemand.models.base import Forecast, run_variants
from nemforecastdemand.models.gbdt import LightGbmQuantile
from nemforecastdemand.models.predict import fit_perturbation_models
from nemforecastdemand.splits import rolling_origins
from nemforecastdemand.utils import save_artifact, timed


def stack_quantiles(forecasts: list[Forecast]) -> np.ndarray:
    return np.stack([fc.quantile_values for fc in forecasts]).astype(np.float32)


def stack_mean(forecasts: list[Forecast]) -> np.ndarray:
    return np.stack([fc.mean_estimate for fc in forecasts]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_panel(cfg.paths.processed)
    splits = load_splits(cfg.paths.processed)
    max_lag = max(cfg.features.demand_lags)
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    # The week-ago recency deviation reads one step behind the longest lag,
    # hence max_lag + 1 when qualifying training origins.
    train_origins = rolling_origins(
        splits["train"].index, panel.index, cfg.origins, cfg.horizon, max_lag + 1
    )
    validation_origins = rolling_origins(
        splits["validation"].index, panel.index, cfg.origins, cfg.horizon, max_lag + 1
    )

    timings: dict[str, float] = {}
    model = LightGbmQuantile(cfg)
    with timed("fit", timings):
        model.fit(panel, train_origins, validation_origins=validation_origins)

    perturbations = fit_perturbation_models(panel, splits["train"].index)
    with timed("test_forecasts", timings):
        variants = run_variants(
            model, panel, test_origins, perturbations, cfg.perturbation.sweep_multipliers, cfg.seed
        )

    y_test = np.stack(
        [panel["demand_mw"].loc[fc.index].to_numpy() for fc in variants["forecast"]]
    ).astype(np.float32)
    arrays = {"origins_test": test_origins.asi8, "y_test": y_test}
    for name, forecasts in variants.items():
        arrays[f"{name}_quantiles"] = stack_quantiles(forecasts)
        arrays[f"{name}_mean"] = stack_mean(forecasts)

    levels = np.asarray(model.quantiles)
    headline = np.mean(
        [
            crps_from_quantiles(y_test[i], arrays["forecast_quantiles"][i], levels).mean()
            for i in range(y_test.shape[0])
        ]
    )
    meta = {
        "quantile_levels": list(model.quantiles),
        "best_iterations": {f"{k:g}": v for k, v in model.best_iterations.items()},
        "mean_best_iteration": model.mean_best_iteration,
        "timings_seconds": timings,
        "headline_test_crps_mw": float(headline),
        "train_origins": len(train_origins),
        "train_rows": len(train_origins) * cfg.horizon,
        "lightgbm_note": (
            "trained on origin blocks (shared design plus recency features) "
            "over the training split, early stopped on validation blocks"
        ),
    }
    save_artifact(cfg.paths.artifacts / "gbdt", arrays, meta)
    print(
        f"lightgbm: fit {timings['fit']:.0f}s, forecasts {timings['test_forecasts']:.0f}s, "
        f"headline test CRPS {headline:.1f} MW"
    )


if __name__ == "__main__":
    main()
