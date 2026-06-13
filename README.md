# nem-demand-forecast

Probabilistic day-ahead forecasting of NSW1 demand in the Australian
National Electricity Market. Three years of half-hourly data, 48-step
forecasts issued twice daily, scored on a season-blocked test set that
covers every month of the evaluation year under archived weather forecasts
as issued, so there is no look-ahead anywhere.

The goal is twofold: build a Bayesian forecaster that beats a strong
classical baseline, and account honestly for what Bayesian modelling and
inference buy and cost along the way. That account includes a model that
failed, the diagnosis that caught it and the repair that followed.

## Models

| Model | Predictive form | Role |
| --- | --- | --- |
| Seasonal naive | Gaussian band from weekly-naive errors | the floor and the MASE base |
| Dynamic harmonic regression + ARIMA errors | analytic Gaussian | classical baseline |
| LightGBM, 15 quantile heads | regularised quantiles | industry benchmark |
| BART, two heads (mean and log scale) | posterior predictive draws | Bayesian trees against LightGBM |
| Bayesian AR(1) regression, heteroskedastic | posterior predictive paths | the Bayesian forecaster |

Every model consumes one shared design matrix: local-clock seasonal
harmonics, temperature, dew point, irradiance, degree days, demand lags
and holidays. The tree models additionally receive origin-anchored
recency features (last observed demand against its day-ago and week-ago
values, plus lead time), the same information the time-series models
carry through their error dynamics. The Bayesian model is also fitted
with a constant innovation scale as an ablation, so the comparison can
price its heteroskedastic variance head separately.

## How the project unfolds

1. **Data and baseline** (notebooks 01 and 02). Three years of NSW1
   demand, ERA5 reanalysis and archived ECMWF IFS day-ahead forecasts;
   cleansing, timezone verification, detection of the non-linear structure
   the models will need, and the committed season-blocked splits. The
   split keeps two-plus years as one contiguous training block and carves
   the final year into monthly validation and test windows, so both
   evaluation sets span every season and selection on validation faces the
   same seasonal mix as the test set. ARIMA order selection on validation
   gives the classical baseline.
2. **A Bayesian model, diagnosed and repaired** (notebook 03). The
   original model is a structural time series (trend plus regression)
   with its states marginalised through a Kalman filter, which keeps it
   tractable over years of data: written with explicit states, NUTS could
   not finish 2,000 iterations in seventeen hours. Fitted and scored, the
   trend model filters better than anything in the field but its 48-step
   forecast trails the classical baseline. The exact aleatoric/epistemic
   variance decomposition attributes that to the slope component (about
   85 percent of predictive variance is process noise), and the repair
   follows the baseline's lead: a stationary AR(1) error written in
   innovations form. No scan, fits in seconds, beats ARIMA.
3. **Inference** (notebook 04). Cold NUTS on the repaired model lands in
   degenerate modes and cannot leave them; chains warm-started from ADVI
   sample the data-preferred mode cleanly, so the warm start is a
   necessity here, not an economy. Surrogate adjudication against the
   certified reference, the homoskedastic ablation and a GPU-against-CPU
   benchmark of both likelihood formulations (the Kalman scan is the rare
   workload where the CPU wins; the innovations form hands the GPU back
   the lead).
4. **Comparison** (notebook 05). CRPS, log score, pinball, MASE,
   coverage, PIT and the energy score over whole 48-step paths; paired
   bootstrap significance; a weather-degradation sweep; the hardest day;
   the compute bill.
5. **Head to head** (notebook 06). The two best models, the Bayesian
   AR(1) with the GP surface and LightGBM, win on different things.
   LightGBM has the lower marginal CRPS and an operationally simpler
   deterministic fit; the Bayesian model is sharper at short lead and is
   the only one of the two that gives coherent 48-step scenarios (the
   energy total, the intra-day ramp), a full density and tail
   probabilities, and a decomposition of predictive variance into
   reducible and irreducible parts. Each claim is demonstrated, with the
   compute and calibration cost of each measured both ways.

## Results

Headline test CRPS (MW, archived forecast weather, averaged over all
twelve evaluation months), produced in
[notebook 05](notebooks/05_model_comparison.ipynb):

| seasonal naive | ARIMA | Bayesian AR(1) homoskedastic | Bayesian AR(1) | Bayesian AR(1) + GP surface | BART | LightGBM |
| --- | --- | --- | --- | --- | --- | --- |
| 490 | 292 | 285 | 272 | 268 | 305 | 207 |

Scores are larger than a single-season evaluation would report, because
the all-season test is harder; the ranking is the point. The Bayesian
AR(1) beats the classical baseline by about 20 MW (paired bootstrap
p < 0.001), the ablation attributes that to its heteroskedastic head
(about 13 MW, p = 0.009), and a learned Gaussian-process interaction
surface adds a few megawatts more, which generalises rather than
overfitting one season precisely because validation now spans all of
them. LightGBM wins marginal CRPS through point accuracy at longer lead
times, but it provides no density, no coherent 48-step paths (the
Bayesian AR(1) posts the best energy score in the field) and no
decomposition of its uncertainty, and within the first hours of each
origin the Bayesian model is the sharpest in the field.

![CRPS by lead time](reports/figures/horizon_crps_all_models.png)

![Warm-start accounting](reports/figures/warm_start_accounting.png)

## Repository layout

- `src/nemforecastdemand/`: the package. `data/` (AEMO and Open-Meteo
  loaders), `features/` (calendar, weather, perturbations), `models/`
  (shared design, ARIMA, LightGBM, BART, BSTS, innovations AR(1), the GP
  surface, ADVI and NUTS drivers, prediction), `evaluation/` (proper
  scores, calibration, sampler diagnostics), `splits.py`, `plotting.py`.
- `scripts/`: download, build and fit entry points; every fit writes
  `artifacts/{name}.npz` plus `.json` metadata that the notebooks read.
- `notebooks/`: the six-notebook narrative listed above.
- `tests/`: scoring rules (sample CRPS verified against the analytic
  Gaussian), features, splits, loaders, the innovations likelihood
  (verified against a hand computation) and a leakage audit: training
  strictly precedes every evaluation window, both evaluation sets span
  all twelve months with balanced early/late slots, design rows are
  invariant to deletion of the future, and scalers and perturbation
  calibration use the training block only.

## Task and data

- **Target:** NSW1 `TOTALDEMAND` (AEMO aggregated price-and-demand
  archive), five-minute dispatch averaged to half hours, May 2023 to May
  2026, stored in UTC, displayed in AEST.
- **Origins:** 00:00 and 12:00 AEST daily, 48 half hours each.
- **Weather:** ERA5 actuals and archived ECMWF IFS forecasts as issued
  one day earlier (Open-Meteo previous-runs API). Headline scores use
  the archived forecasts; perfect foresight and a calibrated
  error-inflation sweep are reported as variants.
- **Splits:** season-blocked. Everything before June 2025 is one
  contiguous training block; each evaluation month contributes an early
  and a late five-day window, with a balanced seeded draw assigning one to
  validation and the other to test. Validation selects model settings; the
  test set is touched once.

## Reproduction

```bash
mamba env create -f environment.yml
conda activate nem-demand-forecast
pip install -e .

python scripts/download_aemo.py        # price-and-demand CSVs -> data/raw, data/interim
python scripts/download_weather.py     # Open-Meteo ERA5 + previous-runs -> data/raw
python scripts/build_dataset.py        # processed panel + split labels (committed)
python scripts/fit_arima.py            # order selection + train-only fit
python scripts/fit_gbdt.py             # LightGBM quantile heads
python scripts/fit_bart.py             # two-head BART, tree count selected on validation
python scripts/fit_bsts_collapsed.py   # trend model (CPU): ADVI + NUTS + warm starts
python scripts/fit_bsts_innovations.py # repaired model: ADVI + NUTS + ablation
python scripts/fit_bsts_hsgp.py        # GP interaction-surface variant
```

The processed panel and split labels are committed, so the fit scripts and
notebooks run without downloads. A CUDA GPU is used automatically when JAX
sees one (`pip install "jax[cuda12]"`); the matrix-form models run fastest
there, while the Kalman-scan trend model runs on the CPU, and the notebooks
report measured timings for both. `pytest` and `ruff` run in CI on every
push.

## Data licences and attribution

- AEMO demand data are used under
  [AEMO's copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
- Weather data by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0):
  ERA5/ERA5T reanalysis (Copernicus Climate Change Service) and archived
  ECMWF IFS operational forecasts. See `data/README.md` for series,
  conventions and caveats.

## Licence

MIT for the code. Data remain under their source licences.
