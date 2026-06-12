"""Fit BART on the full pre-test history and write predictive draws.

Bayesian additive regression trees over the shared design matrix, the
Bayesian counterpart to the LightGBM benchmark: a sum-of-trees prior with
posterior uncertainty over the regression function. Tree structures are
discrete, so neither ADVI nor NUTS applies; the model is fitted by its
native particle-Gibbs sampler (``pymc-bart``) with the Gaussian noise
scale sampled alongside. Like the gradient-boosted benchmark this is a
direct regression: with a 48-step horizon every demand lag at every
horizon step is realised before the origin, so no recursion is needed.

Posterior predictive draws share one function draw across a horizon, so
the 48-step paths carry coherent function uncertainty plus independent
observation noise, and the energy score is well defined.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.features.weather import degree_days
from nemforecastdemand.models.base import build_design, perturbation_overrides
from nemforecastdemand.models.predict import fit_perturbation_models
from nemforecastdemand.splits import horizon_index, rolling_origins
from nemforecastdemand.utils import save_artifact

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--tune", type=int, default=None, help="override warmup iterations")
    parser.add_argument("--draws", type=int, default=None, help="override posterior draws")
    parser.add_argument("--keep-draws", type=int, default=500, help="thinned predictive draws")
    args = parser.parse_args()

    import pymc as pm
    import pymc_bart as pmb

    cfg = load_config(args.config)
    bart = cfg.bart
    tune = args.tune if args.tune is not None else bart.tune
    draws = args.draws if args.draws is not None else bart.draws

    splits = load_splits(cfg.paths.processed)
    panel = pd.concat([splits["train"], splits["validation"], splits["test"]])
    max_lag = max(cfg.features.demand_lags)
    fit_index = panel.index[panel.index < splits["test"].index[0]][max_lag:]
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    horizons = [horizon_index(origin, cfg.horizon) for origin in test_origins]
    perturbations = fit_perturbation_models(panel, splits["train"].index)

    design_actual = build_design(panel, cfg, weather_source="actual")
    design_forecast = build_design(panel, cfg, weather_source="forecast")
    x_train = design_actual.loc[fit_index].to_numpy(dtype=np.float32)
    y_train = panel["demand_mw"].loc[fit_index].to_numpy(dtype=np.float64)
    y_test = np.stack(
        [panel["demand_mw"].loc[index].to_numpy(dtype=np.float32) for index in horizons]
    )
    print(f"fit rows {len(fit_index)}, features {x_train.shape[1]}, origins {len(test_origins)}")

    timings: dict[str, float] = {}
    with pm.Model() as model:
        data_x = pm.Data("X", x_train)
        mu = pmb.BART("mu", data_x, y_train, m=bart.trees)
        sigma = pm.HalfNormal("sigma", float(y_train.std()))
        pm.Normal("y", mu, sigma, observed=y_train, shape=mu.shape)
        start = time.perf_counter()
        idata = pm.sample(
            tune=tune,
            draws=draws,
            chains=bart.chains,
            cores=bart.chains,
            random_seed=cfg.seed,
            progressbar=True,
        )
        timings["fit_seconds"] = time.perf_counter() - start

    # Convergence on the continuous parameter; the function itself is
    # summarised by the worst R-hat across a thinned set of training rows.
    import arviz as az

    sigma_summary = {
        "sigma_rhat": float(az.rhat(idata.posterior["sigma"]).values),
        "sigma_bulk_ess": float(az.ess(idata.posterior["sigma"]).values),
    }
    mu_sub = idata.posterior["mu"].isel(mu_dim_0=slice(0, None, 200))
    sigma_summary["mu_max_rhat_sampled"] = float(az.rhat(mu_sub).max().values)
    print(
        f"fit {timings['fit_seconds']:.0f}s, sigma R-hat {sigma_summary['sigma_rhat']:.4f}, "
        f"mu max R-hat (sampled rows) {sigma_summary['mu_max_rhat_sampled']:.4f}"
    )

    total = bart.chains * draws
    step = max(total // args.keep_draws, 1)
    thinned = idata.sel(draw=slice(None, None, max(step // bart.chains, 1)))

    def predict_rows(design_block: np.ndarray, seed_offset: int) -> np.ndarray:
        """Posterior predictive draws on new rows, trees re-evaluated."""
        with model:
            pm.set_data({"X": design_block.astype(np.float32)})
            post = pm.sample_posterior_predictive(
                thinned,
                sample_vars=["mu", "y"],
                predictions=True,
                progressbar=False,
                random_seed=cfg.seed + seed_offset,
            )
        values = post.predictions["y"].values
        return values.reshape(-1, values.shape[-1]).astype(np.float32)

    def stacked_design(blocks: list[pd.DataFrame]) -> np.ndarray:
        return np.concatenate([block.to_numpy(dtype=np.float32) for block in blocks])

    variants: dict[str, np.ndarray] = {}
    n_origins, horizon = len(test_origins), cfg.horizon

    with_timing = time.perf_counter()
    variants["forecast"] = predict_rows(
        stacked_design([design_forecast.loc[index] for index in horizons]), 1
    )
    variants["actual"] = predict_rows(
        stacked_design([design_actual.loc[index] for index in horizons]), 2
    )
    for j, multiplier in enumerate(cfg.perturbation.sweep_multipliers):
        if multiplier == 0:
            continue
        blocks = []
        for index in horizons:
            overrides = perturbation_overrides(panel, index, perturbations, multiplier, cfg.seed)
            block = design_actual.loc[index].copy()
            block.loc[:, overrides.columns] = overrides
            degrees = degree_days(
                block["temp_c"], cfg.weather.heating_base, cfg.weather.cooling_base
            )
            block.loc[:, ["cooling_deg", "heating_deg"]] = degrees
            blocks.append(block)
        variants[f"perturb_{multiplier:g}"] = predict_rows(stacked_design(blocks), 3 + j)
    timings["predict_seconds"] = time.perf_counter() - with_timing

    arrays = {"origins_test": test_origins.asi8, "y_test": y_test}
    for name, paths in variants.items():
        arrays[f"{name}_paths"] = paths.reshape(-1, n_origins, horizon)
        print(f"variant {name}: paths {arrays[f'{name}_paths'].shape}")

    meta = {
        "sampler": "PGBART (particle Gibbs) for trees, NUTS for sigma",
        "settings": {
            "trees": bart.trees,
            "tune": tune,
            "draws": draws,
            "chains": bart.chains,
            "kept_draws": int(arrays["forecast_paths"].shape[0]),
        },
        "timings_seconds": timings,
        "diagnostics": sigma_summary,
        "fit_window": [str(fit_index[0]), str(fit_index[-1])],
        "fit_rows": len(fit_index),
        "n_features": int(x_train.shape[1]),
    }
    save_artifact(cfg.paths.artifacts / "bart", arrays, meta)
    print(f"bart: artifacts written, fit {timings['fit_seconds']:.0f}s")


if __name__ == "__main__":
    main()
