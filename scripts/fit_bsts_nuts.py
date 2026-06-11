"""Fit the BSTS by NUTS, cold and ADVI-warm-started, and write artifacts.

The cold run (full warmup, adaptive mass matrix) is the reference posterior
and produces the rolling-origin test forecasts. The warm runs draw their
initial positions from a fitted ADVI surrogate and freeze the inverse mass
matrix to the surrogate covariance (diagonal from mean-field, dense from
full-rank) over a grid of reduced warmup lengths. The ADVI fit time is
recorded into each warm run's accounting so the comparison in notebook 04
can be honest: total wall-clock to matched sampling quality, not raw
wall-clock at mismatched mixing.

``--mode bench`` runs a short cold chain for device timing only; with
``--device cpu`` it provides the CPU side of the GPU-versus-CPU comparison.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--device", choices=["default", "cpu"], default="default")
    parser.add_argument("--mode", choices=["full", "cold", "warm", "bench"], default="full")
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
    from nemforecastdemand.models.inference_mcmc import (
        NutsRun,
        fit_nuts,
        flatten_chains,
        warm_start_from_vi,
    )
    from nemforecastdemand.models.inference_vi import fit_advi
    from nemforecastdemand.models.predict import fit_perturbation_models, predict_variants
    from nemforecastdemand.splits import rolling_origins
    from nemforecastdemand.utils import save_artifact, timed

    cfg = load_config(args.config)
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

    def run_meta(run: NutsRun, extra_meta: dict | None = None) -> dict:
        summary = run.summary().reset_index()
        health = run.health(cfg.nuts.max_tree_depth).reset_index()
        meta = {
            "device": run.device,
            "timings_seconds": run.timings,
            "settings": run.settings,
            "site_summary": summary.to_dict("records"),
            "chain_health": health.to_dict("records"),
            "min_bulk_ess": float(summary["min_bulk_ess"].min()),
            "min_tail_ess": float(summary["min_tail_ess"].min()),
            "max_rhat": float(summary["max_rhat"].max()),
            "total_divergences": int(health["divergences"].sum()),
            "fit_window": [str(fit_index[0]), str(fit_index[-1])],
        }
        meta.update(extra_meta or {})
        return meta

    def posterior_arrays(run: NutsRun, include_level: bool = False) -> dict:
        arrays = {}
        for name in bsts.HYPER_SITES:
            arrays[f"post_{name}"] = run.posterior[name]
        if include_level:
            arrays["post_level"] = run.posterior["level"]
        for name, value in run.extra.items():
            arrays[f"extra_{name}"] = np.asarray(value)
        return arrays

    if args.mode == "bench":
        bench_cfg = replace(cfg, nuts=replace(cfg.nuts, warmup=100, samples=100))
        run = fit_nuts(model_fn, bench_cfg.nuts, seed=cfg.seed)
        leapfrogs = int(run.extra["num_steps"].sum())
        meta = run_meta(
            run,
            {
                "total_leapfrogs_sampling": leapfrogs,
                "leapfrogs_per_second": leapfrogs / run.timings["sample_seconds"],
            },
        )
        save_artifact(cfg.paths.artifacts / f"bsts_nuts_bench_{run.device}", {}, meta)
        print(
            f"bench on {run.device}: {meta['leapfrogs_per_second']:.0f} leapfrogs/s, "
            f"warmup {run.timings['warmup_seconds']:.0f}s, "
            f"sampling {run.timings['sample_seconds']:.0f}s"
        )
        return

    if args.mode in ("full", "cold"):
        run = fit_nuts(model_fn, cfg.nuts, seed=cfg.seed)
        print(
            f"cold on {run.device}: warmup {run.timings['warmup_seconds']:.0f}s, "
            f"sampling {run.timings['sample_seconds']:.0f}s, "
            f"max rhat {run.summary()['max_rhat'].max():.4f}"
        )

        draws = flatten_chains(run.posterior)
        keep = max(draws["sigma_level"].shape[0] // cfg.vi.posterior_draws, 1)
        hyper = {name: jnp.asarray(draws[name][::keep]) for name in bsts.HYPER_SITES}

        perturbations = fit_perturbation_models(panel, splits["train"].index)
        test_origins = rolling_origins(
            splits["test"].index,
            panel.index,
            cfg.origins,
            cfg.horizon,
            max(cfg.features.demand_lags),
        )
        timings: dict[str, float] = {}
        with timed("predict_seconds", timings):
            variants, y_true = predict_variants(
                hyper, inputs, panel, cfg, test_origins, perturbations
            )

        arrays = posterior_arrays(run, include_level=True)
        arrays["origins_test"] = test_origins.asi8
        arrays["y_test"] = y_true
        for name, paths in variants.items():
            arrays[f"{name}_paths"] = paths
        save_artifact(
            cfg.paths.artifacts / "bsts_nuts_cold",
            arrays,
            run_meta(run, {"predict_seconds": timings["predict_seconds"]}),
        )
        print("cold: artifacts written")

    if args.mode in ("full", "warm"):
        for kind in ("meanfield", "fullrank"):
            with timed(f"advi_{kind}") as advi_time:
                vi_fit = fit_advi(model_fn, kind, cfg.vi, seed=cfg.seed)
            warm = warm_start_from_vi(vi_fit, cfg.nuts.chains, seed=cfg.seed + 20)
            for reduced in cfg.warm_start.reduced_warmup:
                run = fit_nuts(
                    model_fn, cfg.nuts, seed=cfg.seed + reduced, warmup=reduced, warm_start=warm
                )
                meta = run_meta(
                    run,
                    {
                        "advi_seconds": advi_time.seconds,
                        "advi_kind": kind,
                        "reduced_warmup": reduced,
                    },
                )
                stem = f"bsts_nuts_warm_{kind}_w{reduced}"
                save_artifact(cfg.paths.artifacts / stem, posterior_arrays(run), meta)
                print(
                    f"warm {kind} w{reduced}: warmup {run.timings['warmup_seconds']:.0f}s, "
                    f"sampling {run.timings['sample_seconds']:.0f}s, "
                    f"max rhat {meta['max_rhat']:.4f}, "
                    f"divergences {meta['total_divergences']}"
                )


if __name__ == "__main__":
    main()
