"""Fit the innovations-form AR(1) model on the full training year.

The revision of the BSTS after the collapsed posterior diagnosed its trend:
the same regression and heteroskedastic scale, with the damped slope
replaced by a stationary AR(1) error written in innovations form. The
likelihood has no scan, so every inference route here is cheap and the
device comparison is rerun on a level playing field.

The same suite as the collapsed fits: mean-field and full-rank ADVI, cold
NUTS, warm-started NUTS over the reduced-warmup grid, plus a homoskedastic
ablation (cold NUTS) that isolates what the variance head buys relative to
an ARIMA-style constant scale. With ``--device cpu`` the suite refits for
timing only and writes ``{stem}.cpu.json`` sidecars next to the primary
artifacts.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--device", choices=["default", "cpu"], default="default")
    args = parser.parse_args()

    from dataclasses import replace

    from nemforecastdemand.config import load_config

    cfg = load_config(args.config)
    sidecar_only = args.device == "cpu"
    if sidecar_only:
        # One XLA host device per chain so chains run across cores; the
        # vectorised likelihood itself also parallelises within a chain.
        os.environ["JAX_PLATFORMS"] = "cpu"
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count={cfg.nuts.chains}"
        )
        cfg = replace(cfg, nuts=replace(cfg.nuts, chain_method="parallel"))

    from functools import partial

    import jax.numpy as jnp
    import numpy as np

    from nemforecastdemand.data.loaders import load_panel, load_splits
    from nemforecastdemand.models import bsts, innovations
    from nemforecastdemand.models.inference_mcmc import (
        NutsRun,
        fit_nuts,
        flatten_chains,
        warm_start_from_vi,
    )
    from nemforecastdemand.models.inference_vi import fit_advi
    from nemforecastdemand.models.predict import (
        fit_perturbation_models,
        predict_variants_innovations,
    )
    from nemforecastdemand.splits import rolling_origins
    from nemforecastdemand.utils import save_artifact, timed

    panel = load_panel(cfg.paths.processed)
    splits = load_splits(cfg.paths.processed)
    max_lag = max(cfg.features.demand_lags)

    fit_index = splits["train"].index[max_lag:]
    inputs = bsts.prepare_inputs(panel, cfg, fit_index)
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    perturbations = fit_perturbation_models(panel, splits["train"].index)

    def model_for(bsts_cfg):
        return partial(
            innovations.innovations_model,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.x_mean),
            jnp.asarray(inputs.x_var),
            bsts_cfg,
        )

    def sites_for(bsts_cfg) -> tuple[str, ...]:
        if bsts_cfg.heteroskedastic:
            return innovations.HYPER_SITES
        return tuple(s for s in innovations.HYPER_SITES if s != "gamma")

    def save(stem: str, arrays: dict, meta: dict) -> None:
        if sidecar_only:
            path = cfg.paths.artifacts / f"{stem}.cpu.json"
            path.write_text(json.dumps(meta, indent=2, default=str))
        else:
            save_artifact(cfg.paths.artifacts / stem, arrays, meta)

    def predict_and_pack(draws: dict[str, np.ndarray], timings: dict[str, float], bsts_cfg) -> dict:
        if sidecar_only:
            return {}
        with timed("predict_seconds", timings):
            variants, y_true = predict_variants_innovations(
                {name: jnp.asarray(value) for name, value in draws.items()},
                inputs,
                panel,
                replace(cfg, bsts=bsts_cfg),
                test_origins,
                perturbations,
            )
        arrays = {"origins_test": test_origins.asi8, "y_test": y_true}
        for name, paths in variants.items():
            arrays[f"{name}_paths"] = paths
        return arrays

    def run_meta(run: NutsRun, fit_window, extra_meta: dict | None = None) -> dict:
        summary = run.summary().reset_index()
        health = run.health(cfg.nuts.max_tree_depth).reset_index()
        meta = {
            "device": run.device,
            "timings_seconds": run.timings,
            "settings": run.settings,
            "site_summary": summary.to_dict("records"),
            "chain_health": health.to_dict("records"),
            "min_bulk_ess": float(summary["min_bulk_ess"].min()),
            "max_rhat": float(summary["max_rhat"].max()),
            "total_divergences": int(health["divergences"].sum()),
            "fit_window": [str(fit_window[0]), str(fit_window[-1])],
            "fit_steps": len(fit_window),
        }
        meta.update(extra_meta or {})
        return meta

    model_fn = model_for(cfg.bsts)
    sites = sites_for(cfg.bsts)

    vi_fits = {}
    for kind in ("meanfield", "fullrank"):
        fit = fit_advi(model_fn, kind, cfg.vi, seed=cfg.seed)
        vi_fits[kind] = fit
        print(
            f"innovations {kind} on {fit.device}: {fit.timings['fit_seconds']:.0f}s, "
            f"final ELBO {fit.trace.elbo[-1]:.0f}",
            flush=True,
        )
        draws = fit.posterior_draws(model_fn, seed=cfg.seed + 10, n_draws=cfg.vi.posterior_draws)
        timings = dict(fit.timings)
        arrays = predict_and_pack(draws, timings, cfg.bsts)
        if not sidecar_only:
            arrays.update(
                {
                    "elbo_steps": fit.trace.steps,
                    "elbo": fit.trace.elbo,
                    "energy": fit.trace.energy,
                    "entropy": fit.trace.entropy,
                }
            )
            for name in sites:
                arrays[f"draw_{name}"] = draws[name]
        save(
            f"bsts_innovations_vi_{kind}",
            arrays,
            {
                "guide": kind,
                "device": fit.device,
                "timings_seconds": timings,
                "final_elbo": float(fit.trace.elbo[-1]),
                "fit_window": [str(fit_index[0]), str(fit_index[-1])],
                "fit_steps": len(fit_index),
            },
        )

    # Cold NUTS is kept as a diagnostic, not a reference: from dispersed
    # inits the chains land in degenerate basins (an all-noise mode and the
    # near-unit-root ridge where differencing erases the regression) and
    # cannot cross between them, so its draws document the failure and its
    # predictions are deliberately not produced.
    run = fit_nuts(model_fn, cfg.nuts, seed=cfg.seed)
    print(
        f"innovations cold on {run.device}: warmup {run.timings['warmup_seconds']:.0f}s, "
        f"sampling {run.timings['sample_seconds']:.0f}s, "
        f"max rhat {run.summary()['max_rhat'].max():.4f}",
        flush=True,
    )
    arrays = {}
    if not sidecar_only:
        for name in sites:
            arrays[f"post_{name}"] = run.posterior[name]
        for name, value in run.extra.items():
            arrays[f"extra_{name}"] = np.asarray(value)
    save(
        "bsts_innovations_nuts_cold",
        arrays,
        run_meta(run, fit_index, {"note": "diagnostic only; multimodal cold chains"}),
    )

    reference_warmup = max(cfg.warm_start.reduced_warmup)
    for kind in ("meanfield", "fullrank"):
        warm = warm_start_from_vi(vi_fits[kind], cfg.nuts.chains, seed=cfg.seed + 20)
        for reduced in cfg.warm_start.reduced_warmup:
            run = fit_nuts(
                model_fn, cfg.nuts, seed=cfg.seed + reduced, warmup=reduced, warm_start=warm
            )
            meta = run_meta(
                run,
                fit_index,
                {
                    "advi_seconds": vi_fits[kind].timings["fit_seconds"],
                    "advi_kind": kind,
                    "reduced_warmup": reduced,
                },
            )
            arrays = {}
            timings = dict(run.timings)
            if not sidecar_only:
                arrays = {f"post_{name}": run.posterior[name] for name in sites}
                for name, value in run.extra.items():
                    arrays[f"extra_{name}"] = np.asarray(value)
            # The longest-warmup full-rank run is the reference posterior
            # (cold NUTS being multimodal), so it carries the predictions.
            if kind == "fullrank" and reduced == reference_warmup:
                draws = flatten_chains(run.posterior)
                keep = max(draws["rho"].shape[0] // cfg.vi.posterior_draws, 1)
                thinned = {name: draws[name][::keep] for name in sites}
                arrays.update(predict_and_pack(thinned, timings, cfg.bsts))
                meta["predict_seconds"] = timings.get("predict_seconds")
                meta["role"] = "reference posterior"
            save(f"bsts_innovations_nuts_warm_{kind}_w{reduced}", arrays, meta)
            print(
                f"innovations warm {kind} w{reduced}: "
                f"warmup {run.timings['warmup_seconds']:.0f}s, "
                f"sampling {run.timings['sample_seconds']:.0f}s, "
                f"max rhat {meta['max_rhat']:.4f}, divergences {meta['total_divergences']}",
                flush=True,
            )

    # Ablation: the same model with an ARIMA-style constant innovation
    # scale, so the comparison can attribute the heteroskedastic head's
    # contribution separately from everything else. Warm started from its
    # own mean-field fit, since cold chains suffer the same multimodality.
    homo_bsts = replace(cfg.bsts, heteroskedastic=False)
    homo_model = model_for(homo_bsts)
    homo_sites = sites_for(homo_bsts)
    homo_vi = fit_advi(homo_model, "meanfield", cfg.vi, seed=cfg.seed + 2)
    warm = warm_start_from_vi(homo_vi, cfg.nuts.chains, seed=cfg.seed + 21)
    run = fit_nuts(
        homo_model, cfg.nuts, seed=cfg.seed + 1, warmup=reference_warmup, warm_start=warm
    )
    print(
        f"innovations homoskedastic warm on {run.device}: "
        f"warmup {run.timings['warmup_seconds']:.0f}s, "
        f"sampling {run.timings['sample_seconds']:.0f}s, "
        f"max rhat {run.summary()['max_rhat'].max():.4f}",
        flush=True,
    )
    draws = flatten_chains(run.posterior)
    keep = max(draws["rho"].shape[0] // cfg.vi.posterior_draws, 1)
    thinned = {name: draws[name][::keep] for name in homo_sites}
    timings = dict(run.timings)
    arrays = predict_and_pack(thinned, timings, homo_bsts)
    if not sidecar_only:
        for name in homo_sites:
            arrays[f"post_{name}"] = run.posterior[name]
    save(
        "bsts_innovations_nuts_homoskedastic",
        arrays,
        run_meta(
            run,
            fit_index,
            {
                "ablation": "constant innovation scale",
                "advi_seconds": homo_vi.timings["fit_seconds"],
                "advi_kind": "meanfield",
            },
        ),
    )


if __name__ == "__main__":
    main()
