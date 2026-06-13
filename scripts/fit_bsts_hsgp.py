"""Fit the innovations model with the learned GP interaction surface.

The hand-made interaction columns carry the non-linearities the EDA could
see; this variant adds a truncated spectral Gaussian process over time of
day and temperature so the data choose the rest of the surface, with
kernel-structured shrinkage in the weight priors. It is fitted by mean-field
and full-rank ADVI, and the full-rank fit carries the predictions. Unlike
the plain AR(1) model, its NUTS posterior is multimodal (the kernel
amplitude and the basis weights form a funnel that traps chains in distinct
modes), so a NUTS run is kept only to document that, not for prediction.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--time-harmonics", type=int, default=6)
    parser.add_argument("--temp-basis", type=int, default=8)
    # The kernel hyperparameters couple with the weights more strongly
    # than anything in the plain model, so the reference run gets a full
    # warmup by default rather than the reduced schedule.
    parser.add_argument("--reference-warmup", type=int, default=None)
    args = parser.parse_args()

    from dataclasses import replace
    from functools import partial

    import jax.numpy as jnp
    import numpy as np

    from nemforecastdemand.config import load_config
    from nemforecastdemand.data.loaders import load_panel, load_splits
    from nemforecastdemand.models import bsts, hsgp
    from nemforecastdemand.models.inference_mcmc import fit_nuts, warm_start_from_vi
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

    panel = load_panel(cfg.paths.processed)
    splits = load_splits(cfg.paths.processed)
    max_lag = max(cfg.features.demand_lags)
    fit_index = splits["train"].index[max_lag:]
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

    # The GP variant is reported from full-rank ADVI. Unlike the plain AR(1)
    # model, whose warm-started NUTS reference mixes cleanly, the GP
    # posterior is multimodal: the kernel amplitude and the hundred-odd basis
    # weights form a funnel, and warm-started chains settle in distinct modes
    # (split R-hat in the single digits). The full-rank surrogate finds one
    # coherent mode and sidesteps the issue, which is the same inference-
    # geometry lesson the rest of the project turns on. A NUTS attempt is
    # still run so the notebook can show the multimodality, but its draws are
    # not used for prediction.
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
        timings = dict(fit.timings)
        arrays = {
            "elbo_steps": fit.trace.steps,
            "elbo": fit.trace.elbo,
            "energy": fit.trace.energy,
            "entropy": fit.trace.entropy,
        }
        for name in save_sites:
            arrays[f"draw_{name}"] = np.asarray(draws[name])
        meta = {
            "guide": kind,
            "device": fit.device,
            "timings_seconds": timings,
            "final_elbo": float(fit.trace.elbo[-1]),
            "gp_settings": {"time_harmonics": args.time_harmonics, "temp_basis": args.temp_basis},
            "fit_window": [str(fit_index[0]), str(fit_index[-1])],
        }
        # The full-rank fit carries the predictions for the comparison.
        if kind == "fullrank":
            thinned = {name: jnp.asarray(draws[name]) for name in predict_sites}
            with timed("predict_seconds", timings):
                variants, y_true = predict_variants_innovations(
                    thinned, inputs, panel, cfg, test_origins, perturbations
                )
            arrays["origins_test"] = test_origins.asi8
            arrays["y_test"] = y_true
            for name, paths in variants.items():
                arrays[f"{name}_paths"] = paths
        save_artifact(cfg.paths.artifacts / f"bsts_hsgp_vi_{kind}", arrays, meta)

    # Document the NUTS multimodality without using it for prediction.
    from dataclasses import replace as dc_replace

    reference_warmup = (
        args.reference_warmup if args.reference_warmup is not None else cfg.nuts.warmup
    )
    warm = warm_start_from_vi(vi_fits["fullrank"], cfg.nuts.chains, seed=cfg.seed + 20)
    warm = dc_replace(warm, freeze_mass=False)
    run = fit_nuts(model_fn, cfg.nuts, seed=cfg.seed + 30, warmup=reference_warmup, warm_start=warm)
    summary = run.summary().reset_index()
    print(
        f"hsgp NUTS (diagnostic, multimodal) on {run.device}: "
        f"max rhat {summary['max_rhat'].max():.4f}, "
        f"min bulk ESS {summary['min_bulk_ess'].min():.0f}",
        flush=True,
    )
    save_artifact(
        cfg.paths.artifacts / "bsts_hsgp_nuts_diagnostic",
        {f"post_{name}": np.asarray(run.posterior[name]) for name in predict_sites},
        {
            "device": run.device,
            "timings_seconds": dict(run.timings),
            "site_summary": summary.to_dict("records"),
            "max_rhat": float(summary["max_rhat"].max()),
            "min_bulk_ess": float(summary["min_bulk_ess"].min()),
            "note": "multimodal; reported for diagnosis only, not used for prediction",
        },
    )
    print("hsgp: artifacts written", flush=True)


if __name__ == "__main__":
    main()
