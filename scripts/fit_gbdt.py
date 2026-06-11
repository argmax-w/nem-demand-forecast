"""Fit the LightGBM quantile benchmark and write its forecasts to artifacts.

Heads are trained on the full training split with early stopping on the
validation split (which the trees otherwise never see), then produce
rolling-origin test forecasts under every weather-input variant. Unlike the
window-limited time-series models, the trees use all available history;
that asymmetry is deliberate and documented: each model is run the way a
practitioner would run it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.evaluation.metrics import crps_from_quantiles
from nemforecastdemand.models.base import Forecast, run_variants
from nemforecastdemand.models.gbdt import LightGbmQuantile
from nemforecastdemand.models.predict import fit_perturbation_models
from nemforecastdemand.splits import rolling_origins
from nemforecastdemand.utils import save_artifact, timed


def stack_quantiles(forecasts: list[Forecast]) -> np.ndarray:
    return np.stack([fc.quantile_values for fc in forecasts]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_splits(cfg.paths.processed)
    panel = pd.concat([splits["train"], splits["validation"], splits["test"]])
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max(cfg.features.demand_lags)
    )

    timings: dict[str, float] = {}
    model = LightGbmQuantile(cfg)
    with timed("fit", timings):
        model.fit(panel, splits["train"].index, validation_index=splits["validation"].index)

    # Window ablation: the same protocol restricted to the 56-day tail the
    # time-series models use, separating the contribution of model class
    # from training-data quantity in the comparison.
    ablation = LightGbmQuantile(cfg)
    with timed("ablation_fit", timings):
        ablation.fit(
            panel,
            splits["train"].index[-cfg.bsts.train_days * 48 :],
            validation_index=splits["validation"].index,
        )

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

    levels = np.asarray(model.quantiles)
    headline = np.mean(
        [
            crps_from_quantiles(y_test[i], arrays["forecast_quantiles"][i], levels).mean()
            for i in range(y_test.shape[0])
        ]
    )
    ablation_scores = []
    for i, origin in enumerate(test_origins):
        fc = ablation.forecast(panel, origin, "forecast")
        ablation_scores.append(crps_from_quantiles(y_test[i], fc.quantile_values, levels).mean())
    meta = {
        "window_ablation_crps_mw": float(np.mean(ablation_scores)),
        "window_ablation_days": cfg.bsts.train_days,
        "quantile_levels": list(model.quantiles),
        "best_iterations": {f"{k:g}": v for k, v in model.best_iterations.items()},
        "timings_seconds": timings,
        "headline_test_crps_mw": float(headline),
        "train_rows": len(splits["train"]),
        "lightgbm_note": "trained on the full training split, early stopped on validation",
    }
    save_artifact(cfg.paths.artifacts / "gbdt", arrays, meta)
    print(
        f"lightgbm: fit {timings['fit']:.0f}s, forecasts {timings['test_forecasts']:.0f}s, "
        f"headline test CRPS {headline:.1f} MW"
    )


if __name__ == "__main__":
    main()
