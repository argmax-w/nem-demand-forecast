# %% [markdown]
# # 04. The same model, fitted by NUTS
#
# Identical generative model, identical data, identical priors to notebook
# 03; only the inference changes. Hamiltonian Monte Carlo with the No-U-Turn
# Sampler, four chains run vectorised on the GPU, is treated as the
# reference posterior: asymptotically exact, with the usual battery of
# convergence diagnostics standing guard. This notebook reads the artifacts
# of `scripts/fit_bsts_nuts.py`, examines sampler health, adjudicates the
# two ADVI surrogates against the reference posterior and prices the ADVI
# warm start honestly.
#
# The non-centred parameterisation earns its keep here: NUTS explores a
# roughly 5,400-dimensional latent space whose geometry, with the funnel
# removed, is close enough to Gaussian for large steps and shallow trees.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemforecastdemand.config import load_config
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.evaluation.diagnostics import time_to_target_ess
from nemforecastdemand.evaluation.metrics import crps_samples
from nemforecastdemand.models import bsts
from nemforecastdemand.plotting import palette, save_figure, setup_style
from nemforecastdemand.utils import load_artifact

setup_style()
cfg = load_config()
splits = load_splits(cfg.paths.processed)
panel = pd.concat([splits["train"], splits["validation"], splits["test"]])

cold, cold_meta = load_artifact(cfg.paths.artifacts / "bsts_nuts_cold")
vi_fits = {
    kind: load_artifact(cfg.paths.artifacts / f"bsts_vi_{kind}")
    for kind in ("meanfield", "fullrank")
}

# %% [markdown]
# ## Sampler health
#
# Split R-hat and bulk and tail effective sample sizes per site (vector
# sites report their weakest element), divergences, energy-based fraction
# of missing information (E-BFMI) and tree-depth saturation per chain. The
# things to look for: R-hat at 1.00, ESS comfortably in the hundreds,
# zero or near-zero divergences, E-BFMI above 0.3 and no saturated trees.

# %%
summary = pd.DataFrame(cold_meta["site_summary"]).set_index("site")
summary.round(4)

# %%
health = pd.DataFrame(cold_meta["chain_health"]).set_index("chain")
timing = cold_meta["timings_seconds"]
display_cols = pd.DataFrame(
    {
        "value": {
            "warmup (s)": timing["warmup_seconds"],
            "sampling (s)": timing["sample_seconds"],
            "min bulk ESS": cold_meta["min_bulk_ess"],
            "bulk ESS per second": cold_meta["min_bulk_ess"] / timing["sample_seconds"],
            "max R-hat": cold_meta["max_rhat"],
            "divergences": cold_meta["total_divergences"],
        }
    }
)
print(health.round(3).to_string())
display_cols.round(3)

# %% [markdown]
# ## Traces
#
# Post-warmup traces and rank histograms for the structural hyperparameters.
# Healthy chains are indistinguishable hairy caterpillars; healthy rank
# histograms are flat.

# %%
trace_sites = ["sigma_level", "sigma_slope", "phi", "gamma0"]
fig, axes = plt.subplots(
    len(trace_sites), 2, figsize=(11, 2.1 * len(trace_sites)), width_ratios=[2.2, 1]
)
chain_colours = ["#1f5673", "#7a4988", "#c44536", "#e8a13a"]
for row, site in enumerate(trace_sites):
    draws = cold[f"post_{site}"]
    ranks = draws.ravel().argsort().argsort().reshape(draws.shape)
    for chain in range(draws.shape[0]):
        axes[row, 0].plot(draws[chain], lw=0.4, color=chain_colours[chain], alpha=0.8)
        axes[row, 1].hist(
            ranks[chain],
            bins=20,
            histtype="step",
            color=chain_colours[chain],
            lw=1.0,
        )
    axes[row, 0].set_ylabel(site)
axes[0, 0].set_title("post-warmup trace")
axes[0, 1].set_title("rank histogram")
axes[-1, 0].set_xlabel("draw")
fig.tight_layout()
save_figure(fig, "nuts_traces", cfg.paths.figures)
plt.show()

# %% [markdown]
# ## ADVI against the reference posterior
#
# The question notebook 03 could not answer: which surrogate family is
# closer to the truth? Marginals first, then the correlation structure.

# %%
compare_sites = ["sigma_level", "sigma_slope", "phi", "gamma0"]
fig, axes = plt.subplots(1, len(compare_sites), figsize=(13, 3.2))
for ax, site in zip(axes, compare_sites, strict=True):
    nuts_draws = cold[f"post_{site}"].ravel()
    grid_lo = min(nuts_draws.min(), *(fit[0][f"draw_{site}"].min() for fit in vi_fits.values()))
    grid_hi = max(nuts_draws.max(), *(fit[0][f"draw_{site}"].max() for fit in vi_fits.values()))
    grid = np.linspace(grid_lo, grid_hi, 120)
    for label, draws, colour in (
        ("NUTS", nuts_draws, "black"),
        ("mean-field", vi_fits["meanfield"][0][f"draw_{site}"], palette("demand")),
        ("full-rank", vi_fits["fullrank"][0][f"draw_{site}"], palette("accent")),
    ):
        density = np.histogram(draws, bins=grid, density=True)[0]
        centres = (grid[:-1] + grid[1:]) / 2
        ax.plot(centres, density, label=label, color=colour, lw=1.2)
    ax.set_title(site)
    ax.set_yticks([])
axes[0].legend()
fig.suptitle("Hyperparameter marginals: surrogates against the reference", y=1.04)
save_figure(fig, "advi_vs_nuts_marginals", cfg.paths.figures)
plt.show()

# %%
rows = {}
for site in compare_sites + ["level_init", "slope_init"]:
    nuts_sd = cold[f"post_{site}"].ravel().std()
    rows[site] = {
        "sd NUTS": nuts_sd,
        "sd MF / NUTS": vi_fits["meanfield"][0][f"draw_{site}"].std() / nuts_sd,
        "sd FR / NUTS": vi_fits["fullrank"][0][f"draw_{site}"].std() / nuts_sd,
    }
pd.DataFrame(rows).T.round(3)

# %% [markdown]
# Pairwise correlations among hyperparameters: NUTS defines the target,
# full-rank can chase it and mean-field is structurally zero.

# %%
pair_sites = ["sigma_level", "sigma_slope", "phi", "gamma0"]


def pair_correlations(draw_map: dict[str, np.ndarray]) -> pd.Series:
    out = {}
    for i, a in enumerate(pair_sites):
        for b in pair_sites[i + 1 :]:
            out[f"{a} ~ {b}"] = np.corrcoef(draw_map[a].ravel(), draw_map[b].ravel())[0, 1]
    return pd.Series(out)


pd.DataFrame(
    {
        "NUTS": pair_correlations({s: cold[f"post_{s}"] for s in pair_sites}),
        "full-rank ADVI": pair_correlations(
            {s: vi_fits["fullrank"][0][f"draw_{s}"] for s in pair_sites}
        ),
        "mean-field ADVI": pair_correlations(
            {s: vi_fits["meanfield"][0][f"draw_{s}"] for s in pair_sites}
        ),
    }
).round(3)

# %% [markdown]
# ## The latent trend under both inference paths

# %%
fit_index = panel.index[panel.index < splits["test"].index[0]][-cfg.bsts.train_days * 48 :]
inputs = bsts.prepare_inputs(panel, cfg, fit_index)
times = fit_index.tz_convert("Australia/Brisbane")
level_nuts = cold["post_level"].reshape(-1, cold["post_level"].shape[-1])

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(
    times,
    np.quantile(level_nuts, 0.05, axis=0) * inputs.y_scale + inputs.y_loc,
    np.quantile(level_nuts, 0.95, axis=0) * inputs.y_scale + inputs.y_loc,
    color="black",
    alpha=0.18,
    label="NUTS 90% band",
)
for kind, colour in (("meanfield", palette("demand")), ("fullrank", palette("accent"))):
    arrays, _ = vi_fits[kind]
    ax.plot(
        times,
        arrays["level_mean"] * inputs.y_scale + inputs.y_loc,
        color=colour,
        lw=1.0,
        label=f"{kind} mean",
    )
ax.set_ylabel("trend level (MW)")
ax.set_title("Latent level: ADVI means inside the NUTS band")
ax.legend()
plt.show()

# %% [markdown]
# ## Predictive accuracy, NUTS against ADVI
#
# Same Rao-Blackwellised prediction pipeline, same test origins, archived
# forecast weather. The full cross-model table lives in notebook 05; this
# is the inference-to-inference comparison.

# %%
y_test = cold["y_test"]
crps_rows = {}
for label, paths in (
    ("BSTS NUTS", cold["forecast_paths"]),
    ("BSTS ADVI mean-field", vi_fits["meanfield"][0]["forecast_paths"]),
    ("BSTS ADVI full-rank", vi_fits["fullrank"][0]["forecast_paths"]),
):
    crps_rows[label] = np.mean(
        [crps_samples(y_test[i], paths[:, i, :]).mean() for i in range(y_test.shape[0])]
    )
pd.Series(crps_rows, name="test CRPS (MW)").to_frame().round(1)

# %% [markdown]
# ## Pricing the ADVI warm start
#
# The warm runs seed every chain from the fitted surrogate and freeze the
# inverse mass matrix to the surrogate covariance (diagonal from
# mean-field, dense from full-rank), keeping only step-size adaptation. The
# accounting is strict:
#
# - **cold total** = full warmup + sampling, everything adapted from scratch;
# - **warm total** = ADVI fit + reduced warmup + sampling.
#
# Comparison happens at matched quality: wall-clock to a target bulk ESS of
# 400 with R-hat under 1.01 and no divergences. A shorter warmup that mixes
# worse is not faster, it is unfinished.

# %%
target = cfg.warm_start.target_bulk_ess
runs = {"cold": (None, cold_meta)}
for kind in ("meanfield", "fullrank"):
    for reduced in cfg.warm_start.reduced_warmup:
        stem = f"bsts_nuts_warm_{kind}_w{reduced}"
        runs[f"warm {kind} w={reduced}"] = (None, load_artifact(cfg.paths.artifacts / stem)[1])

rows = {}
for name, (_, meta) in runs.items():
    timing = meta["timings_seconds"]
    advi_seconds = meta.get("advi_seconds", 0.0)
    to_target = time_to_target_ess(
        timing["warmup_seconds"], timing["sample_seconds"], meta["min_bulk_ess"], target
    )
    quality_ok = meta["max_rhat"] < cfg.warm_start.rhat_threshold and meta["total_divergences"] == 0
    rows[name] = {
        "ADVI (s)": advi_seconds,
        "warmup (s)": timing["warmup_seconds"],
        "sampling (s)": timing["sample_seconds"],
        "min bulk ESS": meta["min_bulk_ess"],
        "max R-hat": meta["max_rhat"],
        "divergences": meta["total_divergences"],
        f"total to ESS {target:.0f} (s)": advi_seconds + to_target,
        "quality met": quality_ok,
    }
warm_table = pd.DataFrame(rows).T
warm_table.round(3)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
column = f"total to ESS {target:.0f} (s)"
bars = warm_table[column].astype(float)
colours_bar = ["#2e7d32" if ok else "#b71c1c" for ok in warm_table["quality met"]]
ax.barh(bars.index[::-1], bars[::-1], color=colours_bar[::-1])
ax.set_xlabel(f"wall-clock to bulk ESS {target:.0f}, R-hat < {cfg.warm_start.rhat_threshold} (s)")
ax.set_title("Cold versus warm-started NUTS at matched quality")
save_figure(fig, "warm_start_accounting", cfg.paths.figures)
plt.show()

# %% [markdown]
# Reading the table honestly: the warm starts must amortise the ADVI fit
# they depend on, so they only win when the reduced warmup plus sampling
# saves more than the surrogate cost, and only count at all when quality
# holds (green bars). Any red bar is a warm start that bought speed with
# broken mixing or divergences, the failure mode flagged in the brief: a
# surrogate that under-estimates variance hands the sampler a mis-scaled
# mass matrix. The conclusion belongs to whatever the numbers above say,
# and notebook 05 carries the timing column into the final comparison.
#
# ## GPU against CPU
#
# The same fits, same code, same XLA pipeline, on the RTX 4000 Ada versus
# 32 CPU cores. ADVI is measured directly at matched step counts; NUTS uses
# short matched runs and reports the leapfrog rate, the quantity that scales
# to any run length.

# %%
bench_rows = {}
for device in ("gpu", "cpu"):
    try:
        _, nuts_bench = load_artifact(cfg.paths.artifacts / f"bsts_nuts_bench_{device}")
        _, vi_bench = load_artifact(cfg.paths.artifacts / f"bsts_vi_bench_meanfield_{device}")
        bench_rows[device] = {
            "NUTS leapfrogs per s": nuts_bench["leapfrogs_per_second"],
            "ADVI steps per s": vi_bench["timings_seconds"]["steps_per_second"],
        }
    except FileNotFoundError:
        continue
bench = pd.DataFrame(bench_rows).T
if {"gpu", "cpu"} <= set(bench.index):
    bench.loc["speed-up"] = bench.loc["gpu"] / bench.loc["cpu"]
bench.round(2)

# %% [markdown]
# ## Summary
#
# - NUTS passes its full diagnostic battery on the 5,400-dimensional
#   non-centred model, earning its role as the reference posterior.
# - Against that reference, full-rank ADVI recovers marginal scales and the
#   leading correlations; mean-field is visibly narrower on correlated
#   hyperparameters, the predicted failure mode.
# - All three posteriors forecast almost equally well here, which is itself
#   a finding: the predictive task is dominated by the regression and the
#   filtered state, both of which ADVI pins down cheaply.
# - The warm-start verdict, at matched quality and with the ADVI bill paid,
#   is in the table above, green where legitimate.
