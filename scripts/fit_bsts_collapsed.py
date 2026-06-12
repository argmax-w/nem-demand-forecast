"""Fit the collapsed trend BSTS and write artifacts.

The collapsed model marginalises the latent states through a Kalman filter,
so ADVI (mean-field and full-rank) and NUTS (cold and ADVI-warm-started)
work in a roughly fifty-dimensional hyperparameter space over the whole
training block rather than a state space that grows with the data. The trade
is a sequential filter inside every gradient, which is hostile to the GPU.

The default run is on the CPU: the scan parallelises across cores by chain
and, on this stack, is the only device where the sampler compiles at all.
``--device gpu`` reruns the same fits on the GPU purely for the timing
comparison and writes ``{stem}.gpu.json`` sidecars; any fit the GPU backend
cannot compile is recorded as a failure rather than aborting the benchmark.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    from dataclasses import replace

    from nemforecastdemand.config import load_config

    cfg = load_config(args.config)
    benchmark = args.device == "gpu"
    if not benchmark:
        # The collapsed likelihood is a sequential scan, so vectorised
        # chains run in lockstep on one or two cores. One XLA host device
        # per chain lets the chains run in parallel across the package's
        # cores instead, which is how a practitioner would use the CPU.
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

    panel = load_panel(cfg.paths.processed)
    splits = load_splits(cfg.paths.processed)
    max_lag = max(cfg.features.demand_lags)

    fit_index = splits["train"].index[max_lag:]
    inputs = bsts.prepare_inputs(panel, cfg, fit_index)
    model_fn = partial(
        bsts.bsts_collapsed_model,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.x_mean),
        jnp.asarray(inputs.x_var),
        cfg.bsts,
    )
    sites = tuple(s for s in bsts.HYPER_SITES if s not in ("level_init", "slope_init"))
    test_origins = rolling_origins(
        splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag
    )
    perturbations = fit_perturbation_models(panel, splits["train"].index)

    def prediction_draws(draws: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
        """Add the marginalised initial-state sites as their prior mean."""
        n = draws[sites[0]].shape[0]
        full = {name: jnp.asarray(draws[name]) for name in sites}
        full["level_init"] = jnp.zeros(n)
        full["slope_init"] = jnp.zeros(n)
        return full

    def save(stem: str, arrays: dict, meta: dict) -> None:
        if benchmark:
            (cfg.paths.artifacts / f"{stem}.gpu.json").write_text(
                json.dumps(meta, indent=2, default=str)
            )
        else:
            save_artifact(cfg.paths.artifacts / stem, arrays, meta)

    def attempt(stem: str, thunk):
        """Run a fit; in benchmark mode record a GPU compile failure and skip."""
        if not benchmark:
            return thunk()
        try:
            return thunk()
        except Exception as exc:
            (cfg.paths.artifacts / f"{stem}.gpu.json").write_text(
                json.dumps({"gpu_compile_failed": str(exc)[:300]}, indent=2)
            )
            print(f"{stem}: GPU benchmark failed to compile ({type(exc).__name__})", flush=True)
            return None

    def predict_and_pack(draws: dict[str, np.ndarray], timings: dict[str, float]) -> dict:
        if benchmark:
            return {}
        with timed("predict_seconds", timings):
            variants, y_true = predict_variants(
                prediction_draws(draws), inputs, panel, cfg, test_origins, perturbations
            )
        arrays = {"origins_test": test_origins.asi8, "y_test": y_true}
        for name, paths in variants.items():
            arrays[f"{name}_paths"] = paths
        return arrays

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
            "max_rhat": float(summary["max_rhat"].max()),
            "total_divergences": int(health["divergences"].sum()),
            "fit_window": [str(fit_index[0]), str(fit_index[-1])],
            "fit_steps": len(fit_index),
        }
        meta.update(extra_meta or {})
        return meta

    vi_fits = {}
    for kind in ("meanfield", "fullrank"):
        fit = attempt(
            f"bsts_collapsed_vi_{kind}",
            lambda kind=kind: fit_advi(model_fn, kind, cfg.vi, seed=cfg.seed),
        )
        if fit is None:
            continue
        vi_fits[kind] = fit
        print(
            f"collapsed {kind} on {fit.device}: {fit.timings['fit_seconds']:.0f}s, "
            f"final ELBO {fit.trace.elbo[-1]:.0f}",
            flush=True,
        )
        draws = fit.posterior_draws(model_fn, seed=cfg.seed + 10, n_draws=cfg.vi.posterior_draws)
        timings = dict(fit.timings)
        arrays = predict_and_pack(draws, timings)
        if not benchmark:
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
            f"bsts_collapsed_vi_{kind}",
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

    run = attempt("bsts_collapsed_nuts_cold", lambda: fit_nuts(model_fn, cfg.nuts, seed=cfg.seed))
    if run is not None:
        print(
            f"collapsed cold on {run.device}: warmup {run.timings['warmup_seconds']:.0f}s, "
            f"sampling {run.timings['sample_seconds']:.0f}s, "
            f"max rhat {run.summary()['max_rhat'].max():.4f}",
            flush=True,
        )
        draws = flatten_chains(run.posterior)
        keep = max(draws["sigma_level"].shape[0] // cfg.vi.posterior_draws, 1)
        thinned = {name: draws[name][::keep] for name in sites}
        timings = dict(run.timings)
        arrays = predict_and_pack(thinned, timings)
        if not benchmark:
            for name in sites:
                arrays[f"post_{name}"] = run.posterior[name]
            for name, value in run.extra.items():
                arrays[f"extra_{name}"] = np.asarray(value)
        save(
            "bsts_collapsed_nuts_cold",
            arrays,
            run_meta(run, {"predict_seconds": timings.get("predict_seconds")}),
        )

    for kind in ("meanfield", "fullrank"):
        if kind not in vi_fits:
            continue
        warm = warm_start_from_vi(vi_fits[kind], cfg.nuts.chains, seed=cfg.seed + 20)
        for reduced in cfg.warm_start.reduced_warmup:
            stem = f"bsts_collapsed_nuts_warm_{kind}_w{reduced}"
            run = attempt(
                stem,
                lambda reduced=reduced, warm=warm: fit_nuts(
                    model_fn, cfg.nuts, seed=cfg.seed + reduced, warmup=reduced, warm_start=warm
                ),
            )
            if run is None:
                continue
            meta = run_meta(
                run,
                {
                    "advi_seconds": vi_fits[kind].timings["fit_seconds"],
                    "advi_kind": kind,
                    "reduced_warmup": reduced,
                },
            )
            arrays = {}
            if not benchmark:
                arrays = {f"post_{name}": run.posterior[name] for name in sites}
                for name, value in run.extra.items():
                    arrays[f"extra_{name}"] = np.asarray(value)
            save(stem, arrays, meta)
            print(
                f"collapsed warm {kind} w{reduced}: "
                f"warmup {run.timings['warmup_seconds']:.0f}s, "
                f"sampling {run.timings['sample_seconds']:.0f}s, "
                f"max rhat {meta['max_rhat']:.4f}, divergences {meta['total_divergences']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
