# nem-demand-forecast

Probabilistic day-ahead forecasting of NSW1 operational demand in the
Australian National Electricity Market, built as a like-for-like comparison
of inference strategies on a single Bayesian structural time-series (BSTS)
model: the same generative model is fitted by ADVI and by NUTS and both are
benchmarked against a strong classical baseline, on probabilistic accuracy,
calibration and computational cost.

## The comparison

Every model sees the same design matrix (local-clock seasonal basis,
temperature, dew point, direct and diffuse irradiance, degree days, demand
lags, holidays) and is scored on identical rolling test origins under
identical weather-input variants.

| Model | Training window | Predictive form | Role |
| --- | --- | --- | --- |
| Seasonal naive | 56 days | Gaussian band from weekly-naive errors | the floor and the MASE base |
| Dynamic harmonic regression + ARIMA errors | 56 days and full year | analytic Gaussian, homoskedastic | strong classical baseline |
| LightGBM, 15 quantile heads | full year (and a 56-day ablation) | regularised quantiles | industry point-model foil |
| BSTS, explicit states (~5,400 sampled dims) | 56 days | posterior predictive paths | mean-field ADVI only; the stress test |
| BSTS, collapsed states (~53 sampled dims) | full year | posterior predictive paths | the production formulation |

The BSTS is a stochastic local linear trend (damped slope) with static
seasonal regression, weather and lagged-demand regressors and a
heteroskedastic log-linear observation scale. It appears in two
formulations of the same generative model. The **explicit** form samples
the latent states (scan-based, non-centred innovations), the way the model
would naively be written in any probabilistic programming language; every
half hour adds two latent dimensions, so 56 days already means ~5,400 of
them. Only mean-field ADVI survives that geometry. NUTS was stopped after
seventeen hours on the GPU without completing its 2,000 iterations, and
the full-rank guide diverged twice (its dense Cholesky factor holds
roughly fifteen million entries against 2,688 observations, so it would
be underdetermined even if it optimised); both failures are reported as
findings rather than worked around. The **collapsed** form marginalises
the states analytically through a Kalman filter inside the likelihood, so
inference runs over roughly fifty hyperparameters regardless of data
length; the full training year becomes tractable, the full covariance
becomes a well-determined object and the identical sampling schedule
finishes in under an hour on CPU alone. The collapsed formulation is
fitted five ways:
mean-field ADVI (`AutoNormal`), full-rank ADVI (`AutoMultivariateNormal`),
cold NUTS (the reference posterior) and warm-started NUTS with chain
positions and the frozen inverse mass matrix taken from each surrogate
(diagonal from mean-field, dense from full-rank) over a grid of reduced
warmups.

Prediction is Rao-Blackwellised for every Bayesian fit: conditional on
hyperparameter draws the model is linear-Gaussian, so rolling-origin
forecasts run a Kalman filter per draw and simulate jointly coherent
48-step paths, with no per-origin refitting.

The axes of comparison, and where each is answered:

1. **Inference algorithm at fixed model** (mean-field against full-rank
   against NUTS): posterior fidelity on marginals and correlations,
   predictive consequences, cost. The adjudication runs on the collapsed
   model in notebook 04; notebook 03 documents why only mean-field
   survives the explicit geometry.
2. **Warm-start economics**: cold total against ADVI fit plus reduced
   warmup plus sampling, judged only at matched quality (target bulk ESS,
   clean R-hat, no divergences). Notebook 04.
3. **Formulation against geometry** (explicit ~5,400 dimensions on 56 days
   against collapsed ~53 dimensions on the full year): where NUTS and the
   full-rank guide become intractable, where mean-field still works and
   what marginalisation buys back. Notebooks 03 and 04.
4. **Model class and training window** (Bayesian against classical against
   gradient-boosted against naive): accuracy, calibration, joint-path
   coherence, statistical significance and robustness to degrading weather
   inputs, with ARIMA and LightGBM each fitted at both windows so data
   quantity separates from model class. Notebook 05.
5. **Hardware**: the collapsed suite fitted identically on the GPU and on
   32 CPU cores. Notebook 04.

## Task and data

- **Target:** NSW1 operational demand (AEMO NEMWeb `ACTUAL_HH`), half-hourly,
  May 2025 to May 2026, stored in UTC and displayed in AEST.
- **Origins:** 00:00 and 12:00 AEST daily, each forecasting 48 half hours.
- **Weather:** ERA5 reanalysis actuals and archived ECMWF IFS forecasts as
  issued one day earlier (Open-Meteo previous-runs API), so the headline
  evaluation has no look-ahead. Temperature, dew point, direct and diffuse
  irradiance.
- **Splits:** chronological 70/15/15; validation selects the ARIMA order
  and seasonal basis; the test set is touched only by the final evaluation.
- **Weather-input variants:** archived forecast (headline), ERA5 perfect
  foresight (disclosed upper bound) and a calibrated perturbation sweep
  fitted to measured forecast-minus-ERA5 errors.

## Results

Headline test-set scores (archived forecast weather) are produced in
[notebook 05](notebooks/05_model_comparison.ipynb).

![CRPS by lead time](reports/figures/horizon_crps_all_models.png)

![ELBO decomposition](reports/figures/elbo_decomposition.png)

![Warm-start accounting](reports/figures/warm_start_accounting.png)

## Notebooks

1. [`01_eda_and_cleansing`](notebooks/01_eda_and_cleansing.ipynb): timestamp
   and timezone verification (including the daylight-saving shift in the
   daily shape), demand drivers, forecast-error calibration, cleansing and
   the committed splits.
2. [`02_baseline_arima`](notebooks/02_baseline_arima.ipynb): order
   selection, the trigonometric-versus-RBF basis assessment, calibration
   and test scores for the classical baseline at the 56-day and full-year
   training windows.
3. [`03_bsts_vi`](notebooks/03_bsts_vi.ipynb): the explicit-state BSTS
   fitted by mean-field ADVI, with the ELBO decomposed into energy and
   entropy as it trains, the learned heteroskedastic variance profile,
   the aleatoric-epistemic split of predictive variance, posterior
   predictive forecasts and the documented failures of NUTS and the
   full-rank guide on this geometry.
4. [`04_bsts_nuts`](notebooks/04_bsts_nuts.ipynb): NUTS on the collapsed
   formulation with full diagnostics as the reference posterior, ADVI
   adjudicated against it, the honest cold-versus-warm-start accounting at
   matched effective sample size, the GPU-versus-CPU benchmark and the
   explicit-state intractability finding.
5. [`05_model_comparison`](notebooks/05_model_comparison.ipynb): the master
   table (CRPS, log score, pinball, MASE, coverage, energy score), paired
   bootstrap significance, PIT calibration, horizon-resolved skill, the
   weather-quality sweep, a worst-day case study and the compute table.

## Reproduction

```bash
mamba env create -f environment.yml
conda activate nem-demand-forecast
pip install -e .

python scripts/download_aemo.py      # NEMWeb weekly archives -> data/raw, data/interim
python scripts/download_weather.py   # Open-Meteo ERA5 + previous-runs -> data/raw
python scripts/build_dataset.py      # processed train/val/test parquet (committed)
python scripts/fit_arima.py          # order selection + 56-day and full-year fits -> artifacts/
python scripts/fit_gbdt.py           # LightGBM quantile heads + window ablation -> artifacts/
python scripts/fit_bsts_vi.py        # explicit-state ADVI fits + forecasts -> artifacts/
python scripts/fit_bsts_collapsed.py # collapsed full-year ADVI + NUTS + warm starts -> artifacts/
```

`scripts/fit_bsts_nuts.py` remains as the record of the explicit-state
NUTS attempt but is not part of reproduction: the cold run was stopped
after seventeen hours on the GPU without completing, which is the
intractability finding notebook 04 reports.

The processed splits are committed, so the model scripts and notebooks run
without any downloads. NEMWeb retains roughly thirteen months of demand
archives; rerunning the download later requires moving the window forward
in `config/default.yaml`. A CUDA GPU is used automatically when JAX sees
one (`pip install "jax[cuda12]"`); every script also runs on CPU and the
notebooks report measured speed-ups.

`pytest` covers the scoring rules (the sample-based CRPS is verified
against the analytic Gaussian form), feature engineering, split integrity
and loader schemas. `ruff` handles lint and formatting; CI runs both plus
the tests on every push.

## Data licences and attribution

- AEMO operational demand data are used under
  [AEMO's copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
- Weather data by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0):
  ERA5/ERA5T reanalysis (Copernicus Climate Change Service) and archived
  ECMWF IFS operational forecasts. See `data/README.md` for the exact
  series, conventions and caveats, including the deliberate train/serve
  mismatch between reanalysis-trained coefficients and operational
  forecast covariates.

## Licence

MIT for the code. Data remain under their source licences.
