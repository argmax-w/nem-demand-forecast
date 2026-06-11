# nem-demand-forecast

Probabilistic day-ahead forecasting of NSW1 operational demand in the
Australian National Electricity Market, built as a like-for-like comparison
of inference strategies on a single Bayesian structural time-series (BSTS)
model: the same generative model is fitted by ADVI and by NUTS and both are
benchmarked against a strong classical baseline, on probabilistic accuracy,
calibration and computational cost.

## The comparison

| Model | Inference | Predictive |
| --- | --- | --- |
| Seasonal naive | none | Gaussian band from weekly-naive errors |
| Dynamic harmonic regression + ARIMA errors | maximum likelihood (SARIMAX) | analytic Gaussian |
| BSTS | ADVI, mean-field (`AutoNormal`) | posterior predictive paths |
| BSTS | ADVI, full-rank (`AutoMultivariateNormal`) | posterior predictive paths |
| BSTS | NUTS, 4 chains (reference posterior) | posterior predictive paths |

Because the three Bayesian fits share one model, differences between them
isolate the inference algorithm rather than confounding it with model
choice. The BSTS is a stochastic local linear trend (damped slope) with
static seasonal regression on local-clock Fourier phases, weather and
lagged-demand regressors and a heteroskedastic log-linear observation
scale, written as an explicit scan-based state-space model with non-centred
innovations. Prediction is Rao-Blackwellised: conditional on hyperparameter
draws the model is linear-Gaussian, so rolling-origin forecasts run a
Kalman filter per draw and simulate jointly coherent 48-step paths.

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
   and test scores for the classical baseline.
3. [`03_bsts_vi`](notebooks/03_bsts_vi.ipynb): the BSTS fitted by ADVI,
   with the ELBO decomposed into energy and entropy as it trains, mean-field
   against full-rank, the learned heteroskedastic variance profile and
   posterior predictive forecasts.
4. [`04_bsts_nuts`](notebooks/04_bsts_nuts.ipynb): NUTS with full
   diagnostics as the reference posterior, ADVI adjudicated against it, the
   honest cold-versus-warm-start accounting at matched effective sample
   size and the GPU-versus-CPU benchmark.
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
python scripts/fit_arima.py          # order selection + test forecasts -> artifacts/
python scripts/fit_bsts_vi.py        # ADVI fits + forecasts -> artifacts/
python scripts/fit_bsts_nuts.py      # NUTS cold + warm starts -> artifacts/
```

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
