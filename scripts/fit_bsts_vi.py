"""Fit the BSTS by ADVI and write surrogates, traces and forecasts.

Fits the mean-field and full-rank surrogates on the window ending at the
test boundary, logs the ELBO decomposition while training, draws from each
posterior and produces rolling-origin test forecasts under every
weather-input variant. ``--device cpu`` forces the CPU backend and
``--benchmark`` runs a short timing-only pass, which together provide the
GPU-versus-CPU fit-time comparison.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--device", choices=["default", "cpu"], default="default")
    parser.add_argument(
        "--guides", nargs="+", default=["meanfield", "fullrank"], help="surrogate families to fit"
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="if set, run this many steps for timing only and skip predictions",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    from dataclasses import replace
    from functools import partial

    import jax.numpy as jnp
    import numpy as np
    import pandas as pd

    from nemforecastdemand.config import load_config
    from nemforecastdemand.data.loaders import load_splits
    from nemforecastdemand.models import bsts
    from nemforecastdemand.models.inference_vi import fit_advi
    from nemforecastdemand.models.predict import fit_perturbation_models, predict_variants
    from nemforecastdemand.splits import rolling_origins
    from nemforecastdemand.utils import save_artifact, timed

    cfg = load_config(args.config)
    if args.benchmark_steps:
        cfg = replace(cfg, vi=replace(cfg.vi, steps=args.benchmark_steps))
    splits = load_splits(cfg.paths.processed)
    panel = pd.concat([splits["train"], splits["validation"], splits["test"]])

    fit_index = panel.index[panel.index < splits["test"].index[0]][-cfg.bsts.train_days * 48 :]
    inputs = bsts.prepare_inputs(panel, cfg, fit_index)
    model_fn = partial(
        bsts.bsts_model,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.x_mean),
        jnp.asarray(inputs.x_var),
        cfg.bsts,
    )
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max(cfg.features.demand_lags)
    )
    perturbations = fit_perturbation_models(panel, splits["train"].index)

    # The explicit-state geometry hands the full-rank guide a Cholesky
    # factor of roughly fifteen million entries; at the shared learning
    # rate the eight-particle gradient blows it apart within two hundred
    # steps, so this model trains that family an order slower with a
    # tighter clip and a smaller initial scale.
    overrides = {"fullrank": {"lr_scale": 0.1, "clip": 0.5, "init_scale": 0.005}}

    for kind in args.guides:
        fit = fit_advi(model_fn, kind, cfg.vi, seed=cfg.seed, overrides=overrides.get(kind))
        print(
            f"{kind} on {fit.device}: {fit.timings['fit_seconds']:.0f}s "
            f"({fit.timings['steps_per_second']:.1f} steps/s), "
            f"final ELBO {fit.trace.elbo[-1]:.0f}, converged {fit.trace.converged()}"
        )

        if args.benchmark_steps:
            save_artifact(
                cfg.paths.artifacts / f"bsts_vi_bench_{kind}_{fit.device}",
                {"elbo": fit.trace.elbo},
                {"timings_seconds": fit.timings, "device": fit.device, "steps": cfg.vi.steps},
            )
            continue

        draws = fit.posterior_draws(model_fn, seed=cfg.seed + 10, n_draws=cfg.vi.posterior_draws)
        hyper = {name: jnp.asarray(draws[name]) for name in bsts.HYPER_SITES}
        levels = bsts.states_from_draws(draws, cfg.bsts)

        timings = dict(fit.timings)
        with timed("predict_seconds", timings):
            variants, y_true = predict_variants(
                hyper, inputs, panel, cfg, test_origins, perturbations
            )

        arrays = {
            "elbo_steps": fit.trace.steps,
            "elbo": fit.trace.elbo,
            "energy": fit.trace.energy,
            "entropy": fit.trace.entropy,
            "level_mean": levels.mean(axis=0),
            "level_q05": np.quantile(levels, 0.05, axis=0).astype(np.float32),
            "level_q95": np.quantile(levels, 0.95, axis=0).astype(np.float32),
            "level_paths_thinned": levels[:: max(len(levels) // 100, 1)],
            "origins_test": test_origins.asi8,
            "y_test": y_true,
        }
        for name in bsts.HYPER_SITES:
            arrays[f"draw_{name}"] = draws[name]
        for name, paths in variants.items():
            arrays[f"{name}_paths"] = paths

        meta = {
            "guide": kind,
            "device": fit.device,
            "timings_seconds": timings,
            "final_elbo": float(fit.trace.elbo[-1]),
            "final_entropy": float(fit.trace.entropy[-1]),
            "converged": bool(fit.trace.converged()),
            "vi_settings": {
                "steps": cfg.vi.steps,
                "learning_rate": cfg.vi.learning_rate,
                "num_particles": cfg.vi.num_particles,
                "posterior_draws": cfg.vi.posterior_draws,
            },
            "fit_window": [str(fit_index[0]), str(fit_index[-1])],
            "design_columns": inputs.columns,
            "standardiser": {"y_loc": inputs.y_loc, "y_scale": inputs.y_scale},
        }
        save_artifact(cfg.paths.artifacts / f"bsts_vi_{kind}", arrays, meta)
        print(f"{kind}: artifacts written")


if __name__ == "__main__":
    main()
