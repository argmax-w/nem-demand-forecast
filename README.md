# nem-demand-forecast

Probabilistic day-ahead forecasting of NSW1 operational demand in the
Australian National Electricity Market. One year of half-hourly data,
48-step forecasts issued twice daily, every model scored on the same
held-out test origins with archived weather forecasts as issued, so there
is no look-ahead anywhere.

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

1. **Data and baseline** (notebooks 01 and 02). AEMO demand, ERA5
   reanalysis and archived ECMWF IFS day-ahead forecasts; cleansing,
   timezone verification and committed chronological 70/15/15 splits.
   ARIMA order selection on validation gives the classical baseline.
2. **A Bayesian model, diagnosed and repaired** (notebook 03). The
   original model is a structural time series (trend plus regression)
   with its states marginalised through a Kalman filter, which is what
   makes a full year tractable: written with explicit states, NUTS could
   not finish 2,000 iterations in seventeen hours. Fitted and scored, the
   trend model loses to the seasonal naive at 48 steps. The exact
   aleatoric/epistemic variance decomposition attributes the failure to
   the slope component (84 percent of predictive variance is process
   noise), and the repair follows the baseline's lead: a stationary AR(1)
   error written in innovations form. No scan, fits in seconds, beats
   ARIMA.
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

## Results

Headline test CRPS (MW, archived forecast weather), produced in
[notebook 05](notebooks/05_model_comparison.ipynb):

| seasonal naive | ARIMA | Bayesian AR(1) homoskedastic | Bayesian AR(1) | BART | LightGBM |
| --- | --- | --- | --- | --- | --- |
| 372 | 267 | 279 | 226 | 223 | 179 |

Both Bayesian models beat the classical baseline by around 15 percent
(paired bootstrap p < 0.001) and are statistically tied with each other;
the ablation attributes the Bayesian AR(1)'s gain to its heteroskedastic
head. LightGBM wins marginal CRPS through point accuracy at longer lead
times, but it provides no density, no coherent 48-step paths (the
Bayesian AR(1) posts the best energy score in the field) and no
decomposition of its uncertainty, and within the first two hours of each
origin the Bayesian model is the most accurate in the field.

![CRPS by lead time](reports/figures/horizon_crps_all_models.png)

![Warm-start accounting](reports/figures/warm_start_accounting.png)

## Repository layout

- `src/nemforecastdemand/`: the package. `data/` (NEMWeb and Open-Meteo
  loaders), `features/` (calendar, weather, perturbations), `models/`
  (shared design, ARIMA, LightGBM, BSTS, innovations AR(1), ADVI and
  NUTS drivers, prediction), `evaluation/` (proper scores, calibration,
  sampler diagnostics), `splits.py`, `plotting.py`.
- `scripts/`: download, build and fit entry points; every fit writes
  `artifacts/{name}.npz` plus `.json` metadata that the notebooks read.
- `notebooks/`: the five-notebook narrative listed above.
- `tests/`: scoring rules (sample CRPS verified against the analytic
  Gaussian), features, splits, loaders, the innovations likelihood
  (verified against a hand computation) and a leakage audit: design rows
  invariant to deletion of the future, scalers from the fit window only,
  the shortest demand lag clears the horizon and the committed splits
  cover both daylight-saving phases with test demand inside the training
  range.

## Task and data

- **Target:** NSW1 operational demand (AEMO NEMWeb `ACTUAL_HH`),
  half-hourly, May 2025 to May 2026, stored in UTC, displayed in AEST.
- **Origins:** 00:00 and 12:00 AEST daily, 48 half hours each.
- **Weather:** ERA5 actuals and archived ECMWF IFS forecasts as issued
  one day earlier (Open-Meteo previous-runs API). Headline scores use
  the archived forecasts; perfect foresight and a calibrated
  error-inflation sweep are reported as variants.
- **Splits:** chronological 70/15/15 cut at market-day boundaries;
  validation selects model settings, the test set is touched once.

## Reproduction

```bash
mamba env create -f environment.yml
conda activate nem-demand-forecast
pip install -e .

python scripts/download_aemo.py        # NEMWeb weekly archives -> data/raw, data/interim
python scripts/download_weather.py     # Open-Meteo ERA5 + previous-runs -> data/raw
python scripts/build_dataset.py        # processed train/val/test parquet (committed)
python scripts/fit_arima.py            # order selection + full-history fit
python scripts/fit_gbdt.py             # LightGBM quantile heads
python scripts/fit_bart.py             # two-head BART, tree count selected on validation
python scripts/fit_bsts_collapsed.py   # trend model: ADVI + NUTS + warm starts
python scripts/fit_bsts_innovations.py # repaired model: ADVI + NUTS + ablation
```

The processed splits are committed, so the fit scripts and notebooks run
without downloads. NEMWeb retains roughly thirteen months of demand
archives; rerunning the download later means moving the window in
`config/default.yaml`. A CUDA GPU is used automatically when JAX sees one
(`pip install "jax[cuda12]"`); everything also runs on CPU and the
notebooks report measured timings for both. `pytest` and `ruff` run in CI
on every push.

## Data licences and attribution

- AEMO operational demand data are used under
  [AEMO's copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
- Weather data by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0):
  ERA5/ERA5T reanalysis (Copernicus Climate Change Service) and archived
  ECMWF IFS operational forecasts. See `data/README.md` for series,
  conventions and caveats.

## Licence

MIT for the code. Data remain under their source licences.
