# nem-demand-forecast

Electricity has to be generated at the instant it is used, so the people who run
the grid live by the day-ahead forecast. Under-call the evening peak and the
market scrambles for expensive last-minute power; over-call it and you have paid
to hold reserves nobody needed. This project is a probabilistic day-ahead
forecaster for NSW1 demand in the Australian National Electricity Market: the
output is not a single number but the whole distribution of tomorrow's demand,
half hour by half hour.

![Day-ahead BSTS forecast for NSW1 demand](reports/figures/readme_forecast_fan.png)

One test day, called from its midnight origin off the weather forecast on hand the
day before. The black line is what arrived, the amber line the median, the bands
the central 50, 80 and 95 percent. The observed day stays inside the bands
throughout, tight overnight and widening across the 18:00 peak where the call is
hardest. An operator does not want the single most likely number for the peak,
they want to know how high the tail could reach, and a calibrated density is what
tells them.

![Coherent scenarios against independent marginals](reports/figures/readme_coherent_traces.png)

Underneath that fan sit whole days. Both panels draw scenarios for one example
day, its clock and weather features given but no outcome fixed. On the left the
BSTS samples each line jointly across all 48 half hours, so every one is a
physically plausible day. On the right is all LightGBM can offer: per-step
quantiles with no joint law, so a path is stitched from an independent draw at each
half hour and comes out jagged. The difference bites the moment a decision spans
the whole day. Sum the 48 steps for the day's total energy and the coherent draws
give it with an honest spread; the independent marginals add up half hours they
have pretended are unrelated and badly understate it, and a step-to-step quantity
like the ramp they cannot give at all. Whole-day decisions need the joint law,
which is why the headline forecaster is Bayesian.
[Notebook 06](notebooks/06_bayes_vs_lightgbm.ipynb) measures the gap.

The raw material is three years of half-hourly NSW1 demand. Forecasts run 48 steps
ahead, issued twice a day, scored on a season-blocked test set that covers every
month of the evaluation year under the weather forecasts actually on hand the day
before, with nothing allowed to peek at the future. The aim is a Bayesian
forecaster that matches a strong classical baseline on raw accuracy and beats it
where a probabilistic forecast should be judged (density quality, calibration,
short-lead sharpness, coherent whole-day scenarios), and an honest account of what
the Bayesian machinery brings and what it costs.

## The models

| Model | Predictive form | Role |
| --- | --- | --- |
| Seasonal naive | Gaussian band from weekly-naive errors | the floor and the MASE base |
| Dynamic harmonic regression + AR(1) errors | analytic Gaussian | classical baseline |
| LightGBM, 15 quantile heads | regularised quantiles | industry benchmark |
| BART, two heads (mean and log scale) | posterior predictive draws | Bayesian trees against LightGBM |
| BSTS: regression with a heteroskedastic AR(2) error | posterior predictive paths | the Bayesian forecaster |

Every model reads one shared design matrix: local-clock seasonal harmonics,
temperature, dew point, irradiance, degree days, demand lags and holidays. The
tree models also get origin-anchored recency features (the last observed demand
against its day-ago and week-ago values, the recent slope and curvature, and the
lead time), which hand them the information the BSTS carries inside its AR error
dynamics. That BSTS error is an AR([1,2]): the diagnostics show clear lag-2
structure, and where an AR(1) error carries only the residual level forward, an
AR(2) error carries a level and a slope, so the forecast near the origin tracks the
day's momentum. The ARIMA, left to choose its order on validation, settles on
AR(1), the extra lag not paying its way out of sample. The observation scale is
heteroskedastic, growing and shrinking with the hour of day, as the EDA asks.

## How the project unfolds

1. **Data and baseline** (notebooks 01 and 02). Three years of NSW1 demand, ERA5
   reanalysis and archived ECMWF IFS forecasts; cleansing, timezone checks, the
   non-linear structure, and the committed season-blocked splits. Choosing the
   ARIMA order on validation gives the classical baseline.
2. **The BSTS** (notebook 03). A seasonal regression with a stationary AR(2) error
   and a heteroskedastic scale, fitted by full-rank ADVI. The likelihood is written
   on the innovations, so there is no sequential scan and it fits in seconds, and it
   leads the field on log score and short-lead sharpness.
3. **Inference** (notebook 04). From cold starts NUTS wanders into degenerate
   modes; warm-started from the variational fit the chains settle into the mode the
   data prefer, so the warm start is what makes the sampler work at all. I check the
   variational fit against the certified reference and benchmark both devices; the
   likelihood is pure matrix arithmetic, so the GPU wins.
4. **Comparison** (notebook 05). Log score and calibration first, then CRPS,
   pinball, MASE, coverage, PIT and the energy score over whole paths, with
   paired-bootstrap significance, a weather-degradation sweep, the hardest day of
   the year and the compute bill.
5. **Head to head** (notebook 06). The two strongest models win on different
   things: LightGBM takes the lower marginal CRPS and a simpler fit; the BSTS is
   sharper at short lead and is the only one producing coherent 48-step scenarios, a
   full density, and a split of its uncertainty into the part more data would remove
   and the part it would not.
6. **Operations** (notebook 07). The BSTS as a control-room forecaster on a real
   winter peak: the live forecast, how it sharpens across intraday re-issues, the
   probability of a spike against the record, and the ramp, stress-duration and
   reserve numbers an operator reads, all from one coherent predictive.

## Results

The metrics that lead are the log score and calibration, not a single error
number. On the test set, under archived forecast weather, averaged over all twelve
evaluation months, from [notebook 05](notebooks/05_model_comparison.ipynb):

| model | log score (nats) | 50% coverage | 90% coverage | CRPS (MW) |
| --- | --- | --- | --- | --- |
| BSTS | 7.47 | 0.50 | 0.86 | 272 |
| BART | 7.52 | 0.56 | 0.93 | 298 |
| ARIMA AR(1) | 7.62 | 0.54 | 0.87 | 258 |
| seasonal naive | 8.04 | 0.56 | 0.90 | 490 |
| LightGBM | none (no density) | 0.36 | 0.77 | 200 |

The BSTS posts the best log score (7.47 against ARIMA's 7.62, paired bootstrap
p < 0.001) and the best calibration, its 50 percent interval covering almost
exactly 50 percent of outcomes, and it wins the energy score over whole paths,
which only a coherent joint predictive can earn. LightGBM takes the single-number
CRPS by a wide margin (200 MW) but carries no density, so it has no log score, and
its intervals are overconfident, the nominal 90 percent band catching only 77
percent. ARIMA edges the BSTS on overall CRPS (258 against 272, p = 0.02) by
winning the long horizon, while the BSTS is far sharper in the first hours (58
against 83 MW at 30 minutes). The absolute scores run high because handling every
season at once is hard; the ranking is the story.

![CRPS by lead time](reports/figures/horizon_crps_all_models.png)

![Warm-start accounting](reports/figures/warm_start_accounting.png)

## Repository layout

- `src/nemforecastdemand/`: the package. `data/` (AEMO and Open-Meteo loaders),
  `features/` (calendar, weather, perturbations), `models/` (shared design and
  inputs, ARIMA, LightGBM, BART, the innovations-form BSTS, the ADVI and NUTS
  drivers, prediction), `evaluation/` (proper scores, calibration, sampler
  diagnostics, posterior divergence), `splits.py`, `plotting.py`.
- `scripts/`: the download, build and fit entry points; every fit writes
  `artifacts/{name}.npz` plus a `.json` of metadata that the notebooks read.
- `notebooks/`: the seven-notebook narrative laid out above.
- `tests/`: scoring rules (sample CRPS against the analytic Gaussian), features,
  splits, loaders, the innovations likelihood (against a hand computation), the
  posterior KL primitives and a leakage audit. The audit pins down no-look-ahead:
  training precedes every evaluation window, both evaluation sets span all twelve
  months with balanced slots, design rows do not change when the future is deleted,
  and the scalers and perturbation calibration see the training block only.

## Task and data

- **Target:** NSW1 `TOTALDEMAND` (the AEMO price-and-demand archive), five-minute
  dispatch averaged to half hours, May 2023 to May 2026, stored in UTC and shown in
  AEST.
- **Origins:** 00:00 and 12:00 AEST daily, 48 half hours each.
- **Weather:** ERA5 actuals and archived ECMWF IFS forecasts as issued one day
  earlier (the Open-Meteo previous-runs API). The headline scores use the archived
  forecasts; perfect foresight and a calibrated error-inflation sweep are reported
  as variants.
- **Splits:** season-blocked. Everything before June 2025 is one contiguous
  training block; each evaluation month contributes an early and a late five-day
  window, and a balanced seeded draw sends one to validation and the other to test.
  Validation chooses settings; the test set is touched once.

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
python scripts/fit_bsts_innovations.py # the BSTS: ADVI + NUTS
```

Or chain every fit and then execute every notebook in order with
`bash scripts/run_all.sh` (pass `fits` or `notebooks` to run a single stage).

The processed panel and split labels are committed, so the fit scripts and
notebooks run with no downloads. JAX picks up a CUDA GPU automatically
(`pip install "jax[cuda12]"`); the BSTS likelihood is pure matrix arithmetic, so it
runs fastest there, and the notebooks report timings for both devices. `pytest` and
`ruff` run in CI on every push.

## Data licences and attribution

- AEMO demand data are used under [AEMO's copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
- Weather data by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0): ERA5/ERA5T
  reanalysis (Copernicus Climate Change Service) and archived ECMWF IFS
  operational forecasts. See `data/README.md` for the series, conventions and
  caveats.

## Licence

MIT for the code. The data remain under their source licences.
