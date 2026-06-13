# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (nem-demand-forecast)
#     language: python
#     name: nem-demand-forecast
# ---

# %% [markdown]
# # 06. Bayesian AR(1) + GP against LightGBM
#
# **Goal.** The two best forecasters in the comparison win on different
# things, so neither dominates. This notebook lays the trade out in full:
# where LightGBM wins (marginal accuracy and operational simplicity),
# where the Bayesian model wins (short-lead sharpness, coherent scenarios,
# a full density and a decomposed account of uncertainty), the compute cost
# of each, and a check that the richer Bayesian output is not paid for in
# calibration. Each claim is demonstrated on the test set.
#
# The structural fact underneath every difference: LightGBM's forecast is
# fifteen independent per-step quantile heads, a set of marginal numbers
# with no joint law and no density between or beyond the quantile levels.
# The Bayesian forecast is a set of jointly sampled paths from a generative
# model. That is why the wins fall where they do.

# %%
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_panel, load_splits
from nemforecastdemand.evaluation.calibration import pit_histogram, pit_samples
from nemforecastdemand.evaluation.metrics import (
    crps_from_quantiles,
    crps_samples,
    energy_score,
    log_score_samples,
)
from nemforecastdemand.models import bsts
from nemforecastdemand.models.hsgp import gp_design
from nemforecastdemand.models.predict import variance_decomposition_innovations
from nemforecastdemand.plotting import palette, save_figure, setup_style
from nemforecastdemand.splits import rolling_origins
from nemforecastdemand.utils import load_artifact

setup_style()
cfg = load_config()
GP_TIME_HARMONICS, GP_TEMP_BASIS = 6, 8
cfg_gp = replace(
    cfg,
    features=replace(
        cfg.features, hsgp_time_harmonics=GP_TIME_HARMONICS, hsgp_temp_basis=GP_TEMP_BASIS
    ),
)

gp, gp_meta = load_artifact(cfg.paths.artifacts / "bsts_hsgp_vi_fullrank")
gbdt, gbdt_meta = load_artifact(cfg.paths.artifacts / "gbdt")
ar_meta = load_artifact(cfg.paths.artifacts / "bsts_innovations_nuts_warm_fullrank_w300")[1]
levels = np.array(gbdt_meta["quantile_levels"])

paths = gp["forecast_paths"]  # (S, O, H) coherent posterior predictive paths
quantiles = gbdt["forecast_quantiles"]  # (O, Q, H) per-step marginal quantiles
y = gp["y_test"]
n_draws, n_origins, horizon = paths.shape
hours = (np.arange(horizon) + 1) / 2
GP_BLUE, LG_GREEN = palette("demand"), "#2e7d32"

gp_origin_crps = np.array([crps_samples(y[i], paths[:, i, :]).mean() for i in range(n_origins)])
lg_origin_crps = np.array(
    [crps_from_quantiles(y[i], quantiles[i], levels).mean() for i in range(n_origins)]
)
gp_h = np.array([crps_samples(y[:, h], paths[:, :, h]).mean() for h in range(horizon)])
lg_h = np.stack(
    [crps_from_quantiles(y[i], quantiles[i], levels) for i in range(n_origins)]
).mean(axis=0)

# %% [markdown]
# ## The headline trade
#
# LightGBM has the lower CRPS overall, but the per-lead-time view shows the
# two models own different parts of the horizon: the Bayesian model is much
# sharper in the first hours and LightGBM pulls ahead once the lead grows.

# %%
crossover = next((h + 1 for h in range(horizon) if gp_h[h] > lg_h[h]), None)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(hours, gp_h, color=GP_BLUE, label=f"Bayesian AR(1) + GP (overall {gp_origin_crps.mean():.0f})")
ax.plot(hours, lg_h, color=LG_GREEN, label=f"LightGBM (overall {lg_origin_crps.mean():.0f})")
if crossover:
    ax.axvline(crossover / 2, color="grey", ls=":", lw=0.8)
ax.set_xlabel("lead time (hours)")
ax.set_ylabel("CRPS (MW)")
ax.set_title("Each model owns a different part of the horizon")
ax.legend()
save_figure(fig, "bench_horizon_crps", cfg.paths.figures)
plt.show()

# %% [markdown]
# ## Where LightGBM wins
#
# ### Marginal accuracy, and it learns interactions for free
#
# On the headline number LightGBM is clearly better: lower CRPS, lower
# point error, and the win widens with lead time. Both models see the same
# design, but the Bayesian model needed hand-built degree-day-by-time
# interactions and a learned GP surface to get this close, while the trees
# discover whatever interactions matter on their own. Once the AR anchor
# decays, that flexible non-linear mean is what carries LightGBM ahead.

# %%
gp_med = paths.mean(axis=0)
lg_med = quantiles[:, levels.tolist().index(0.5), :]
pd.DataFrame(
    {
        "Bayesian AR(1) + GP": {
            "CRPS (MW)": gp_origin_crps.mean(),
            "MAE (MW)": float(np.abs(gp_med - y).mean()),
            "CRPS at 30 min (MW)": gp_h[0],
            "CRPS at 24 h (MW)": gp_h[47],
        },
        "LightGBM": {
            "CRPS (MW)": lg_origin_crps.mean(),
            "MAE (MW)": float(np.abs(lg_med - y).mean()),
            "CRPS at 30 min (MW)": lg_h[0],
            "CRPS at 24 h (MW)": lg_h[47],
        },
    }
).round(0)

# %% [markdown]
# ### Operational simplicity: nothing to babysit
#
# This is LightGBM's deeper advantage. It fits deterministically: call it,
# get heads, done. The Bayesian forecast is the product of an inference
# apparatus that has to be watched. The GP variant here is reported from
# full-rank ADVI precisely because its NUTS posterior is multimodal, and
# the plain AR(1) needs a warm-started NUTS reference because cold NUTS
# does not mix (notebook 04). None of R-hat, warm starts, multimodality or
# surrogate adjudication exists for LightGBM. In production that is fewer
# ways to fail silently and less expertise required to run.

# %%
pd.DataFrame(
    {
        "fits deterministically": {"LightGBM": "yes", "Bayesian AR(1) + GP": "no (stochastic VI/MCMC)"},
        "convergence to monitor": {"LightGBM": "none", "Bayesian AR(1) + GP": "ELBO, R-hat, ESS"},
        "multimodality risk": {"LightGBM": "none", "Bayesian AR(1) + GP": "yes (GP NUTS is multimodal)"},
        "hand-built interactions needed": {"LightGBM": "no", "Bayesian AR(1) + GP": "yes"},
    }
).T

# %% [markdown]
# ### Compute, measured both ways
#
# Speed is not where LightGBM wins, which is worth stating plainly. On this
# machine the Bayesian likelihood is vectorised on the GPU while LightGBM
# trains fifteen separate gradient-boosted quantile heads on the CPU, so
# the Bayesian fit and prediction are in fact faster in wall-clock. The
# LightGBM advantage above is operational simplicity, not run time, and the
# numbers should make that precise rather than imply a speed edge it does
# not have.

# %%
gp_fit = gp_meta["timings_seconds"]
ar_fit = ar_meta["timings_seconds"]
compute = pd.DataFrame(
    {
        "LightGBM (15 heads, CPU)": {
            "fit (s)": gbdt_meta["timings_seconds"]["fit"],
            "forecast all origins and variants (s)": gbdt_meta["timings_seconds"]["test_forecasts"],
        },
        "Bayesian AR(1) + GP, ADVI (GPU)": {
            "fit (s)": gp_fit["fit_seconds"],
            "forecast all origins and variants (s)": gp_fit["predict_seconds"],
        },
        "Bayesian AR(1), NUTS reference (GPU)": {
            "fit (s)": ar_meta["advi_seconds"]
            + ar_fit["warmup_seconds"]
            + ar_fit["sample_seconds"],
            "forecast all origins and variants (s)": ar_meta.get("predict_seconds", float("nan")),
        },
    }
).T
compute.round(1)

# %% [markdown]
# ## Where the Bayesian model wins
#
# ### 1. Sharpest at short lead
#
# The crossover figure already showed it: at the first half hour the
# Bayesian model is more than twice as sharp. The reason is the AR(1)
# error, which carries the residual observed at the forecast origin forward
# as $\rho^{h}$, anchoring the first steps to what just happened. LightGBM
# sees the lagged-demand and recency features but not the realised error at
# issue time, so it cannot tighten the near horizon the same way.

# %%
pd.DataFrame(
    {
        "lead (h)": [0.5, 1, 2, 3, 6],
        "Bayes+GP CRPS": [gp_h[i] for i in (0, 1, 3, 5, 11)],
        "LightGBM CRPS": [lg_h[i] for i in (0, 1, 3, 5, 11)],
    }
).set_index("lead (h)").round(0)

# %% [markdown]
# ### 2. Coherent 48-step scenarios
#
# The Bayesian paths are sampled jointly, so each is one plausible
# trajectory of the whole day and any function of the 48 steps has a
# correct distribution. LightGBM's marginal heads cannot produce
# trajectories, only per-step bands.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
worst = int(np.argmax(gp_origin_crps))
idx = np.arange(horizon)
for s in range(12):
    axes[0].plot(idx, paths[s, worst, :], color=GP_BLUE, lw=0.6, alpha=0.5)
axes[0].plot(idx, y[worst], color="black", lw=1.4, label="observed")
axes[0].set_title("Bayesian model: 12 coherent sampled days")
axes[0].set_ylabel("demand (MW)")
axes[0].legend()
for q in (0.05, 0.25, 0.5, 0.75, 0.95):
    axes[1].plot(idx, quantiles[worst, levels.tolist().index(q), :], color=LG_GREEN, lw=0.8)
axes[1].plot(idx, y[worst], color="black", lw=1.4)
axes[1].set_title("LightGBM: per-step quantile bands, no trajectories")
for ax in axes:
    ax.set_xlabel("horizon step")
fig.tight_layout()
save_figure(fig, "bench_paths_vs_bands", cfg.paths.figures)
plt.show()

# %% [markdown]
# **Coherence is real and measurable.** Shuffling each step's draws
# independently across the sample axis leaves every per-step marginal
# untouched, so per-step CRPS is unchanged, but it destroys the cross-step
# dependence and the energy score, a proper score over whole paths, gets
# worse. LightGBM only ever has the marginals, so it lives in the shuffled
# world by construction.

# %%
rng = np.random.default_rng(cfg.seed)
shuffled = paths.copy()
for i in range(n_origins):
    for h in range(horizon):
        shuffled[rng.permutation(n_draws), i, h] = paths[:, i, h]
pd.DataFrame(
    {
        "coherent paths": {
            "per-step CRPS (MW)": np.mean([crps_samples(y[i], paths[:, i, :]).mean() for i in range(n_origins)]),
            "energy score (MW)": np.mean([energy_score(y[i], paths[:, i, :]) for i in range(n_origins)]),
        },
        "marginals only (shuffled)": {
            "per-step CRPS (MW)": np.mean([crps_samples(y[i], shuffled[:, i, :]).mean() for i in range(n_origins)]),
            "energy score (MW)": np.mean([energy_score(y[i], shuffled[:, i, :]) for i in range(n_origins)]),
        },
    }
).round(1)

# %% [markdown]
# **A decision that needs the joint law: the day's total energy.** A
# procurement or reserve decision over the whole day depends on the sum of
# all 48 steps, and the spread of that sum is driven by the correlation
# between steps. Coherent paths give it directly. The only thing marginal
# quantiles support is an independence assumption, adding the per-step
# variances, which ignores the strong positive correlation of a smooth
# demand curve and badly understates the risk. (The intra-day ramp is
# starker still: a step-to-step change is a function of the joint law of
# two steps, so marginals cannot give its distribution at all.)

# %%
from scipy.stats import norm

total = paths.sum(axis=2)  # (S, O): the day's total per coherent draw
coherent_sd = total.std(axis=0)
independence_sd = np.sqrt(paths.var(axis=0).sum(axis=1))  # all that marginals allow
ramp_p95 = np.quantile(np.abs(np.diff(paths, axis=2)).max(axis=2), 0.95, axis=0)
print(f"daily total (sum of 48 half hours), spread over {n_origins} test days:")
print(f"  coherent sd:              {coherent_sd.mean():.0f} MW")
print(f"  marginal-independence sd: {independence_sd.mean():.0f} MW")
print(f"  the independence assumption understates the spread "
      f"{coherent_sd.mean() / independence_sd.mean():.1f}-fold")
print(f"  intra-day ramp, coherent P95: {ramp_p95.mean():.0f} MW per half hour "
      "(marginals cannot give this)")

mean_total = paths[:, worst, :].mean(axis=0).sum()
grid = np.linspace(total[:, worst].min(), total[:, worst].max(), 200)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(total[:, worst] / 1000, bins=40, color=GP_BLUE, alpha=0.7, density=True, label="coherent paths")
ax.plot(
    grid / 1000, norm.pdf(grid, mean_total, independence_sd[worst]) * 1000,
    color="#c44536", lw=1.6, ls="--", label="marginal independence",
)
ax.set_xlabel("daily total demand (GW, sum of 48 half hours)")
ax.set_ylabel("density")
ax.set_title("Spread of the day's total energy (one test day)")
ax.legend(fontsize=8)
save_figure(fig, "bench_daily_total", cfg.paths.figures)
plt.show()

# %% [markdown]
# ### 3. A full predictive density
#
# The Bayesian posterior predictive is a density at every step, so the CDF,
# any tail probability and a log score are defined everywhere. LightGBM
# returns fifteen quantiles; between them a density must be interpolated
# and beyond the outer pair it is absent.

# %%
o, h = worst, 35
draws = paths[:, o, h]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(draws, bins=40, color=GP_BLUE, alpha=0.7, density=True)
axes[0].set_title("Bayesian: a density at this step")
axes[0].set_xlabel("demand (MW)")
axes[0].set_ylabel("density")
order = np.argsort(draws)
axes[1].plot(draws[order], np.linspace(0, 1, n_draws), color=GP_BLUE, label="Bayesian CDF")
axes[1].plot(quantiles[o, :, h], levels, "o", color=LG_GREEN, ms=5, label="LightGBM quantiles (15)")
axes[1].axhspan(0.975, 1.0, color="#c44536", alpha=0.12)
axes[1].axhspan(0.0, 0.025, color="#c44536", alpha=0.12)
axes[1].set_title("LightGBM gives 15 points; the shaded tails are undefined")
axes[1].set_xlabel("demand (MW)")
axes[1].set_ylabel("cumulative probability")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_figure(fig, "bench_density_vs_quantiles", cfg.paths.figures)
plt.show()

# %%
threshold = np.quantile(y, 0.97)
gp_exceed = (paths > threshold).mean(axis=0)
top_q = quantiles[:, levels.tolist().index(0.975), :]
beyond = 100 * (top_q < threshold).mean()
print(f"threshold (97th percentile of demand): {threshold:.0f} MW")
print(f"Bayesian: exact P(exceed) everywhere; mean {gp_exceed.mean():.3f}, max {gp_exceed.max():.2f}")
print(f"LightGBM: threshold is beyond its 97.5 quantile in {beyond:.0f}% of cells,")
print("          where it can say no more than 'below 2.5%'")
print(f"Bayesian log score (density-based): "
      f"{np.mean([log_score_samples(y[i], paths[:, i, :]).mean() for i in range(n_origins)]):.2f}; "
      f"LightGBM log score: undefined")

# %% [markdown]
# ### 4. Decomposed, interpretable, generative uncertainty
#
# The Bayesian predictive variance splits exactly into a parameter
# (epistemic) part that more data would shrink and an innovation
# (aleatoric) part that is irreducible under the model, and the structure
# producing it is inspectable. LightGBM returns calibrated quantiles with
# no such split and no generative structure to read.

# %%
panel = load_panel(cfg.paths.processed)
splits = load_splits(cfg.paths.processed)
max_lag = max(cfg.features.demand_lags)
fit_index = splits["train"].index[max_lag:]
inputs = bsts.prepare_inputs(panel, cfg_gp, fit_index)
test_origins = rolling_origins(splits["test"].index, panel.index, cfg.origins, cfg.horizon, max_lag)
parts = variance_decomposition_innovations(
    {name: gp[f"draw_{name}"] for name in ("rho", "beta", "gamma0", "gamma")},
    inputs, panel, cfg_gp, test_origins,
)
total = parts["parameter"] + parts["innovation"]
epistemic = (parts["parameter"] / total).mean(axis=0)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.fill_between(hours, 0, epistemic, color=GP_BLUE, alpha=0.5, label="epistemic (parameter)")
ax.fill_between(hours, epistemic, 1, color="#cccccc", alpha=0.7, label="aleatoric (innovation)")
ax.set_xlabel("lead time (hours)")
ax.set_ylabel("share of predictive variance")
ax.set_ylim(0, 1)
ax.set_xlim(hours[0], hours[-1])
ax.set_title("Which uncertainty more data would remove")
ax.legend(loc="center right", fontsize=8)
save_figure(fig, "bench_aleatoric_epistemic", cfg.paths.figures)
plt.show()
print(f"epistemic share over the horizon: {(parts['parameter'] / total).mean():.1%} "
      "- with two years of training the parameters are pinned down, so almost all")
print("predictive uncertainty is irreducible. LightGBM cannot make that statement.")

# %% [markdown]
# The learned interaction surface is the posterior-mean GP contribution to
# demand as a function of local time of day and temperature. It recovers,
# from the data, the effect the EDA could only point at: hot afternoons and
# evenings lift demand far above the additive fit while warm nights do not.
# It is part of the generative model and carries its own posterior
# uncertainty, where a tree ensemble offers only feature importances.

# %%
day = pd.date_range("2025-06-15 00:00", periods=horizon, freq="30min", tz="UTC")
temps = np.linspace(8.0, 38.0, 40)
gp_columns = [c for c in inputs.columns if c.startswith("gp_")]
gp_weight = gp["draw_beta"].mean(axis=0)[-len(gp_columns):]
surface = np.stack(
    [
        gp_design(day, pd.Series(np.full(horizon, t), index=day), cfg_gp.features).to_numpy() @ gp_weight
        for t in temps
    ]
)
surface_mw = surface * inputs.y_scale
local = day.tz_convert("Australia/Sydney")
local_hour = local.hour + local.minute / 60
order = np.argsort(local_hour)
fig, ax = plt.subplots(figsize=(8, 4.5))
mesh = ax.pcolormesh(
    local_hour[order], temps, surface_mw[:, order], cmap="RdBu_r",
    vmin=-np.abs(surface_mw).max(), vmax=np.abs(surface_mw).max(), shading="auto",
)
fig.colorbar(mesh, label="learned demand contribution (MW)")
ax.set_xlabel("local Sydney hour")
ax.set_ylabel("temperature (C)")
ax.set_title("The learned temperature-by-time-of-day surface")
save_figure(fig, "bench_gp_surface", cfg.paths.figures)
plt.show()

# %% [markdown]
# ## Calibration is not the price
#
# The richer Bayesian output would matter little if it were less honest.
# The PIT histograms and central coverage show the Bayesian model is at
# least as well calibrated as LightGBM, marginally better at the 50 percent
# level, so its benefits come at no calibration cost.

# %%
gp_pit = np.concatenate([pit_samples(y[i], paths[:, i, :]) for i in range(n_origins)])
lg_pit = np.array(
    [
        np.interp(y[i, j], quantiles[i, :, j], levels, left=0.0, right=1.0)
        for i in range(n_origins)
        for j in range(horizon)
    ]
)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, pit, name, colour in (
    (axes[0], gp_pit, "Bayesian AR(1) + GP", GP_BLUE),
    (axes[1], lg_pit, "LightGBM", LG_GREEN),
):
    density, edges = pit_histogram(pit, bins=20)
    ax.bar(edges[:-1], density, width=np.diff(edges), align="edge", color=colour, alpha=0.85)
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_title(name)
    ax.set_xlabel("PIT")
axes[0].set_ylabel("relative density")
fig.suptitle("PIT, test set (flat is calibrated)", y=1.04)
save_figure(fig, "bench_pit", cfg.paths.figures)
plt.show()


def coverage(get_interval) -> dict:
    out = {}
    for level in cfg.evaluation.interval_levels:
        lo, hi = get_interval(level)
        out[f"cover {level:.0%}"] = float(((y >= lo) & (y <= hi)).mean())
    return out


gp_cov = coverage(
    lambda L: (np.quantile(paths, 0.5 - L / 2, axis=0), np.quantile(paths, 0.5 + L / 2, axis=0))
)
lg_cov = coverage(
    lambda L: (
        quantiles[:, levels.tolist().index(round(0.5 - L / 2, 3)), :],
        quantiles[:, levels.tolist().index(round(0.5 + L / 2, 3)), :],
    )
)
pd.DataFrame({"Bayesian AR(1) + GP": gp_cov, "LightGBM": lg_cov}).round(2)

# %% [markdown]
# ## Summary: when to use which
#
# - **LightGBM** for the single sharpest point or quantile, particularly
#   beyond a couple of hours, and for an operationally simple model with no
#   inference to monitor and no hand-built interactions. It is the better
#   choice when the deliverable is a number and robustness matters more than
#   structure.
# - **Bayesian AR(1) + GP** when the near horizon matters most, when
#   decisions span the whole day and need coherent scenarios (the energy
#   total, the intra-day ramp, anything path-dependent), when a density or
#   tail probability is required, or when the question is not only "what
#   will demand be" but "how much of this uncertainty could more data
#   remove and what drives it".
# - Neither is paid for in calibration, and on this hardware the Bayesian
#   fit is the faster of the two; the genuine LightGBM advantages are
#   marginal accuracy and operational simplicity, not speed.
