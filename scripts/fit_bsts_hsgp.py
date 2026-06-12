"""Fit the innovations model with the learned GP interaction surface.

The hand-made interaction columns carry the non-linearities the EDA could
see; this variant adds a truncated spectral Gaussian process over time of
day and temperature so the data choose the rest of the surface, with
kernel-structured shrinkage in the weight priors. Everything else matches
the innovations suite: mean-field and full-rank ADVI, then warm-started
NUTS from the full-rank guide as the reference posterior (cold chains
inherit the AR model's degenerate basins), with predictions and all
weather variants from the reference.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--time-harmonics", type=int, default=6)
    parser.add_argument("--temp-basis", type=int, default=8)
    args = parser.parse_args()

    from dataclasses import replace
    from functools import partial

    import jax.numpy as jnp
    import numpy as np
    import pandas as pd

    from nemforecastdemand.config import load_config
    from nemforecastdemand.data.loaders import load_splits
    from nemforecastdemand.models import bsts, hsgp
    from nemforecastdemand.models.inference_mcmc import fit_nuts, flatten_chains, warm_start_from_vi
    from nemforecastdemand.models.inference_vi import fit_advi
    from nemforecastdemand.models.predict import (
        fit_perturbation_models,
        predict_variants_innovations,
    )
    from nemforecastdemand.splits import rolling_origins
    from nemforecastdemand.utils import save_artifact, timed

    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        features=replace(
            cfg.features,
            hsgp_time_harmonics=args.time_harmonics,
            hsgp_temp_basis=args.temp_basis,
        ),
    )

    splits = load_splits(cfg.paths.processed)
    panel = pd.concat([splits["train"], splits["validation"], splits["test"]])
    max_lag = max(cfg.features.demand_lags)
    fit_index = panel.index[panel.index < splits["test"].index[0]][max_lag:]
    inputs = bsts.prepare_inputs(panel, cfg, fit_index)
    gp = hsgp.gp_metadata(inputs.columns, cfg.features)
    gp_jnp = {
        "n_linear": gp["n_linear"],
        "time_order": jnp.asarray(gp["time_order"]),
        "temp_omega": jnp.asarray(gp["temp_omega"]),
    }
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    perturbations = fit_perturbation_models(panel, splits["train"].index)
    print(
        f"design: {inputs.x_mean.shape[1]} columns "
        f"({gp['n_linear']} linear, {gp['time_order'].shape[0]} GP)",
        flush=True,
    )

    model_fn = partial(
        hsgp.innovations_hsgp_model,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.x_mean),
        jnp.asarray(inputs.x_var),
        gp_jnp,
        cfg.bsts,
    )
    predict_sites = ("rho", "beta", "gamma0", "gamma")
    save_sites = hsgp.GP_HYPER_SITES

    vi_fits = {}
    for kind in ("meanfield", "fullrank"):
        fit = fit_advi(model_fn, kind, cfg.vi, seed=cfg.seed)
        vi_fits[kind] = fit
        print(
            f"hsgp {kind} on {fit.device}: {fit.timings['fit_seconds']:.0f}s, "
            f"final ELBO {fit.trace.elbo[-1]:.0f}",
            flush=True,
        )
        draws = fit.posterior_draws(model_fn, seed=cfg.seed + 10, n_draws=cfg.vi.posterior_draws)
        arrays = {
            "elbo_steps": fit.trace.steps,
            "elbo": fit.trace.elbo,
            "energy": fit.trace.energy,
            "entropy": fit.trace.entropy,
        }
        for name in save_sites:
            arrays[f"draw_{name}"] = np.asarray(draws[name])
        save_artifact(
            cfg.paths.artifacts / f"bsts_hsgp_vi_{kind}",
            arrays,
            {
                "guide": kind,
                "device": fit.device,
                "timings_seconds": dict(fit.timings),
                "final_elbo": float(fit.trace.elbo[-1]),
                "gp_settings": {
                    "time_harmonics": args.time_harmonics,
                    "temp_basis": args.temp_basis,
                },
                "fit_window": [str(fit_index[0]), str(fit_index[-1])],
            },
        )

    reference_warmup = max(cfg.warm_start.reduced_warmup)
    warm = warm_start_from_vi(vi_fits["fullrank"], cfg.nuts.chains, seed=cfg.seed + 20)
    run = fit_nuts(model_fn, cfg.nuts, seed=cfg.seed + 30, warmup=reference_warmup, warm_start=warm)
    summary = run.summary().reset_index()
    print(
        f"hsgp warm reference on {run.device}: "
        f"warmup {run.timings['warmup_seconds']:.0f}s, "
        f"sampling {run.timings['sample_seconds']:.0f}s, "
        f"max rhat {summary['max_rhat'].max():.4f}",
        flush=True,
    )

    draws = flatten_chains(run.posterior)
    keep = max(draws["rho"].shape[0] // cfg.vi.posterior_draws, 1)
    thinned = {name: jnp.asarray(draws[name][::keep]) for name in predict_sites}
    timings = dict(run.timings)
    with timed("predict_seconds", timings):
        variants, y_true = predict_variants_innovations(
            thinned, inputs, panel, cfg, test_origins, perturbations
        )
    arrays = {"origins_test": test_origins.asi8, "y_test": y_true}
    for name, paths in variants.items():
        arrays[f"{name}_paths"] = paths
    for name in save_sites:
        arrays[f"post_{name}"] = np.asarray(draws[name])
    health = run.health(cfg.nuts.max_tree_depth).reset_index()
    save_artifact(
        cfg.paths.artifacts / "bsts_hsgp_nuts_reference",
        arrays,
        {
            "device": run.device,
            "timings_seconds": timings,
            "settings": run.settings,
            "site_summary": summary.to_dict("records"),
            "min_bulk_ess": float(summary["min_bulk_ess"].min()),
            "max_rhat": float(summary["max_rhat"].max()),
            "total_divergences": int(health["divergences"].sum()),
            "advi_seconds": vi_fits["fullrank"].timings["fit_seconds"],
            "advi_kind": "fullrank",
            "reduced_warmup": reference_warmup,
            "gp_settings": {
                "time_harmonics": args.time_harmonics,
                "temp_basis": args.temp_basis,
            },
            "fit_window": [str(fit_index[0]), str(fit_index[-1])],
        },
    )
    print("hsgp: artifacts written", flush=True)


if __name__ == "__main__":
    main()
