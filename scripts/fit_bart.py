"""Fit heteroskedastic BART on origin blocks and write predictive draws.

Bayesian additive regression trees, the Bayesian counterpart to the
LightGBM benchmark: where LightGBM's quantile heads each learn their own
function of the features, BART here learns two, a sum-of-trees mean and a
sum-of-trees log scale over the same design, so the predictive spread
adapts to the covariates exactly as freely as the quantile heads do. Tree
structures are discrete, so neither ADVI nor NUTS applies; the model is
fitted by its native particle-Gibbs sampler (``pymc-bart``).

Training rows are origin blocks (the shared design plus the origin-anchored
recency features), mirroring the operational setting, and the tree count is
selected on the validation split before the final fit on the full pre-test
history, the same protocol that selects the ARIMA order. The target is
standardised so the log-scale head's exp link starts at a sane magnitude.

Posterior predictive draws share one function draw across a horizon, so the
48-step paths carry coherent function uncertainty plus covariate-dependent
noise, and the energy score is well defined.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from nemforecastdemand import gates
from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_panel, load_splits
from nemforecastdemand.evaluation.metrics import crps_samples
from nemforecastdemand.features.weather import degree_days
from nemforecastdemand.models.base import (
    build_design,
    perturbation_overrides,
    recency_features,
    stacked_origin_design,
)
from nemforecastdemand.models.predict import fit_perturbation_models
from nemforecastdemand.splits import horizon_index, rolling_origins
from nemforecastdemand.utils import save_artifact

warnings.filterwarnings("ignore", category=FutureWarning)

TREE_CANDIDATES = (50, 100, 200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--tune", type=int, default=None, help="override warmup iterations")
    parser.add_argument("--draws", type=int, default=None, help="override posterior draws")
    parser.add_argument("--keep-draws", type=int, default=1000, help="thinned predictive draws")
    args = parser.parse_args()

    import pymc as pm
    import pymc_bart as pmb

    cfg = load_config(args.config)
    bart = cfg.bart
    tune = args.tune if args.tune is not None else bart.tune
    draws = args.draws if args.draws is not None else bart.draws

    panel = load_panel(cfg.paths.processed)
    gates.validate_inputs(panel)  # fail hard before fitting on poisoned data
    splits = load_splits(cfg.paths.processed)
    max_lag = max(cfg.features.demand_lags)
    # The week-ago recency deviation reads one step behind the longest lag,
    # hence max_lag + 1 when qualifying training origins.
    train_origins = rolling_origins(
        splits["train"].index, panel.index, cfg.origins, cfg.horizon, max_lag + 1
    )
    validation_origins = rolling_origins(
        splits["validation"].index, panel.index, cfg.origins, cfg.horizon, max_lag + 1
    )
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    perturbations = fit_perturbation_models(panel, splits["train"].index)

    def fit_two_head(x: np.ndarray, y_std: np.ndarray, m: int, tune_n: int, draws_n: int):
        """Fit the mean and log-scale heads; return (model, idata)."""
        with pm.Model() as model:
            data_x = pm.Data("X", x)
            w = pmb.BART("w", data_x, y_std, m=m, shape=(2, len(y_std)), separate_trees=True)
            pm.Normal("y", w[0], pm.math.exp(w[1]), observed=y_std, shape=w[0].shape)
            idata = pm.sample(
                tune=tune_n,
                draws=draws_n,
                chains=bart.chains,
                cores=bart.chains,
                random_seed=cfg.seed,
                progressbar=True,
            )
        return model, idata

    def predict_rows(model, thinned, design_block: np.ndarray, seed_offset: int) -> np.ndarray:
        """Posterior predictive draws on new rows, trees re-evaluated."""
        with model:
            pm.set_data({"X": design_block.astype(np.float32)})
            post = pm.sample_posterior_predictive(
                thinned,
                sample_vars=["w", "y"],
                predictions=True,
                progressbar=False,
                random_seed=cfg.seed + seed_offset,
            )
        values = post.predictions["y"].values
        return values.reshape(-1, values.shape[-1]).astype(np.float32)

    # Tree-count selection on the validation split, mirroring the ARIMA
    # order-selection protocol: fit on train, score CRPS on validation.
    x_train, y_train = stacked_origin_design(panel, cfg, train_origins)
    x_val, y_val = stacked_origin_design(panel, cfg, validation_origins)
    y_loc, y_scale = float(y_train.mean()), float(y_train.std())
    print(
        f"selection: {len(train_origins)} train origins ({len(x_train)} rows), "
        f"{len(validation_origins)} validation origins",
        flush=True,
    )
    selection: dict[int, float] = {}
    for m in TREE_CANDIDATES:
        start = time.perf_counter()
        model, idata = fit_two_head(
            x_train.to_numpy(dtype=np.float32),
            ((y_train - y_loc) / y_scale).to_numpy(),
            m,
            tune_n=min(tune, 500),
            draws_n=min(draws, 500),
        )
        step = max(bart.chains * min(draws, 500) // 250, 1)
        thinned = idata.sel(draw=slice(None, None, step))
        paths = predict_rows(model, thinned, x_val.to_numpy(dtype=np.float32), 99)
        crps = float(
            crps_samples(((y_val - y_loc) / y_scale).to_numpy(), paths.astype(np.float64)).mean()
            * y_scale
        )
        selection[m] = crps
        print(
            f"  m={m}: validation CRPS {crps:.1f} MW ({time.perf_counter() - start:.0f}s)",
            flush=True,
        )
    best_m = min(selection, key=selection.get)
    print(f"selected m={best_m}", flush=True)

    # Final fit on the full pre-test history (train plus validation), with
    # scalers from the same window. The final fit uses the training origins
    # only; validation served its purpose in selecting the tree count.
    fit_origins = train_origins
    x_fit, y_fit = stacked_origin_design(panel, cfg, fit_origins)
    y_loc, y_scale = float(y_fit.mean()), float(y_fit.std())
    y_fit_std = ((y_fit - y_loc) / y_scale).to_numpy()
    print(f"final fit: {len(fit_origins)} origins, {len(x_fit)} rows", flush=True)

    timings: dict[str, float] = {}
    start = time.perf_counter()
    model, idata = fit_two_head(
        x_fit.to_numpy(dtype=np.float32), y_fit_std, best_m, tune_n=tune, draws_n=draws
    )
    timings["fit_seconds"] = time.perf_counter() - start

    # Convergence on the sampled function values: worst R-hat across a
    # thinned set of training rows, for both heads.
    import arviz as az

    w_sub = idata.posterior["w"].isel(w_dim_1=slice(0, None, 500))
    diagnostics = {
        "w_max_rhat_sampled": float(az.rhat(w_sub).max().values),
        "w_min_bulk_ess_sampled": float(az.ess(w_sub).min().values),
    }
    print(
        f"fit {timings['fit_seconds']:.0f}s, "
        f"w max R-hat (sampled rows) {diagnostics['w_max_rhat_sampled']:.4f}",
        flush=True,
    )

    # Thinning is per chain: a step of (chains * draws / keep) leaves
    # keep / chains draws in each chain, keep in total.
    step = max(bart.chains * draws // args.keep_draws, 1)
    thinned = idata.sel(draw=slice(None, None, step))

    horizons = [horizon_index(origin, cfg.horizon) for origin in test_origins]
    y_test = np.stack(
        [panel["demand_mw"].loc[index].to_numpy(dtype=np.float32) for index in horizons]
    )
    design_actual = build_design(panel, cfg, weather_source="actual")
    design_forecast = build_design(panel, cfg, weather_source="forecast")
    recency_blocks = {
        origin: recency_features(panel, origin, cfg.horizon) for origin in test_origins
    }

    def stacked_blocks(design: pd.DataFrame, override_blocks: dict | None = None) -> np.ndarray:
        blocks = []
        for origin, index in zip(test_origins, horizons, strict=True):
            block = design.loc[index]
            if override_blocks is not None:
                block = override_blocks[origin]
            block = pd.concat([block, recency_blocks[origin]], axis=1)
            blocks.append(block.to_numpy(dtype=np.float32))
        return np.concatenate(blocks)

    def in_megawatts(paths: np.ndarray) -> np.ndarray:
        return (paths * y_scale + y_loc).astype(np.float32)

    variants: dict[str, np.ndarray] = {}
    n_origins, horizon = len(test_origins), cfg.horizon

    predict_start = time.perf_counter()
    variants["forecast"] = in_megawatts(
        predict_rows(model, thinned, stacked_blocks(design_forecast), 1)
    )
    variants["actual"] = in_megawatts(
        predict_rows(model, thinned, stacked_blocks(design_actual), 2)
    )
    for j, multiplier in enumerate(cfg.perturbation.sweep_multipliers):
        if multiplier == 0:
            continue
        override_blocks = {}
        for origin, index in zip(test_origins, horizons, strict=True):
            overrides = perturbation_overrides(panel, index, perturbations, multiplier, cfg.seed)
            block = design_actual.loc[index].copy()
            block.loc[:, overrides.columns] = overrides
            degrees = degree_days(
                block["temp_c"], cfg.weather.heating_base, cfg.weather.cooling_base
            )
            block.loc[:, ["cooling_deg", "heating_deg"]] = degrees
            override_blocks[origin] = block
        variants[f"perturb_{multiplier:g}"] = in_megawatts(
            predict_rows(model, thinned, stacked_blocks(design_actual, override_blocks), 3 + j)
        )
    timings["predict_seconds"] = time.perf_counter() - predict_start

    arrays = {"origins_test": test_origins.asi8, "y_test": y_test}
    for name, paths in variants.items():
        arrays[f"{name}_paths"] = paths.reshape(-1, n_origins, horizon)
        print(f"variant {name}: paths {arrays[f'{name}_paths'].shape}", flush=True)
    gates.check_forecast(samples=arrays["forecast_paths"])  # withhold nonsense

    meta = {
        "sampler": "PGBART (particle Gibbs), two sum-of-trees heads (mean, log scale)",
        "settings": {
            "trees": best_m,
            "tree_selection_crps_mw": {str(m): v for m, v in selection.items()},
            "tune": tune,
            "draws": draws,
            "chains": bart.chains,
            "kept_draws": int(arrays["forecast_paths"].shape[0]),
        },
        "timings_seconds": timings,
        "diagnostics": diagnostics,
        "fit_origins": len(fit_origins),
        "fit_rows": len(x_fit),
        "n_features": int(x_fit.shape[1]),
        "target_standardisation": {"loc": y_loc, "scale": y_scale},
    }
    save_artifact(cfg.paths.artifacts / "bart", arrays, meta)
    print(f"bart: artifacts written, fit {timings['fit_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
