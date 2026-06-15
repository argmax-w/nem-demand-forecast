# nem-demand-forecast

Electricity has to be generated at the instant it is used, so the people who run
the grid live by the day-ahead forecast. Under-call the evening peak and the
market scrambles for expensive last-minute power; over-call it and you have paid
to hold reserves nobody needed. This project builds a probabilistic day-ahead
forecaster for NSW1 demand in the Australian National Electricity Market.
Probabilistic is the operative word: the output is not a single number but the
whole distribution of what tomorrow's demand might be, half hour by half hour.

The raw material is three years of half-hourly NSW1 demand. Forecasts run 48
steps ahead, issued twice a day, and they are scored on a season-blocked test set
that covers every month of the evaluation year using the weather forecasts that
were actually on hand the day before. Nothing in the pipeline is allowed to peek
at the future.

I set out with two goals. The first is a Bayesian forecaster that matches a
strong classical baseline on raw accuracy and beats it where a probabilistic
forecast ought to be judged: the quality of its predictive density, how well
calibrated it is, how sharp it is at short lead, and whether it can produce
coherent whole-day scenarios rather than a row of disconnected guesses. The
second is to be honest about what the Bayesian machinery actually brings and what
it costs.

## The models

| Model | Predictive form | Role |
| --- | --- | --- |
| Seasonal naive | Gaussian band from weekly-naive errors | the floor and the MASE base |
| Dynamic harmonic regression + AR(1) errors | analytic Gaussian | classical baseline |
| LightGBM, 15 quantile heads | regularised quantiles | industry benchmark |
| BART, two heads (mean and log scale) | posterior predictive draws | Bayesian trees against LightGBM |
| BSTS: regression with a heteroskedastic AR(2) error | posterior predictive paths | the Bayesian forecaster |

Every model reads from one shared design matrix: local-clock seasonal harmonics,
temperature, dew point, irradiance, degree days, demand lags and holidays. The
tree models get a little extra, a set of origin-anchored recency features (the
last observed demand against its day-ago and week-ago values, the recent slope
and curvature, and the lead time), which hand them the same information the BSTS
carries inside its AR error dynamics.

That BSTS error is second order, an AR([1,2]). The residual diagnostics show
clear lag-2 structure, and where an AR(1) error carries only the residual level
forward, an AR(2) error carries a level and a slope. So the forecast near the
origin tracks the day's momentum, not just its height. The classical ARIMA, by
contrast, is left to choose its own order on validation and settles on AR(1),
because out of sample the extra lag does not pay its way for it. The observation
scale is heteroskedastic, meaning the noise grows and shrinks with the hour of
day, and the EDA asks for that directly.

## How the project unfolds

1. **Data and baseline** (notebooks 01 and 02). Three years of NSW1 demand, ERA5
   reanalysis and archived ECMWF IFS day-ahead forecasts, then cleansing,
   timezone checks, a look at the non-linear structure, and the committed
   season-blocked splits. The split holds two-plus years back as one contiguous
   training block and carves the final year into monthly validation and test
   windows, so both evaluation sets see every season. Choosing the ARIMA order on
   validation gives the classical baseline.
2. **The BSTS** (notebook 03). A seasonal regression with a stationary AR(2)
   error and a heteroskedastic scale, fitted by full-rank ADVI. The lag-2 term
   gives the error a slope at the origin that an AR(1) error cannot, which is
   where the short-lead sharpness comes from. The likelihood is written on the
   innovations, so there is no sequential scan, it fits in seconds, and it leads
   the field on log score and short-lead sharpness.
3. **Inference** (notebook 04). Run from cold, dispersed starts, NUTS wanders
   into degenerate modes and cannot climb back out. Warm-start the chains from the
   variational fit and they settle cleanly into the mode the data prefer, so the
   warm start is not a speed-up, it is what makes the sampler work at all. I then
   check the variational approximation against the certified reference and
   benchmark both devices. The likelihood is pure matrix arithmetic, so the GPU
   wins throughout.
4. **Comparison** (notebook 05). Log score and calibration first, then CRPS,
   pinball, MASE, coverage, PIT and the energy score over whole 48-step paths,
   with paired-bootstrap significance, a weather-degradation sweep, the hardest
   day of the year and the compute bill.
5. **Head to head** (notebook 06). The two strongest models, the BSTS and
   LightGBM, win on different things. LightGBM takes the lower marginal CRPS and a
   simpler deterministic fit; the BSTS is sharper at short lead and is the only
   one of the two that produces coherent 48-step scenarios, a full density with
   tail probabilities, and a split of its uncertainty into the part more data
   would remove and the part it would not.
6. **Operations** (notebook 07). The BSTS put to work as a control-room
   forecaster on a real winter peak: the live forecast, how it sharpens across
   intraday re-issues as the AR(2) error reads the day's emerging level, the
   probability of a spike against the past record, and the ramp, stress-duration
   and reserve numbers an operator actually reads, all drawn from one coherent
   predictive.

## Results

The point of a probabilistic forecaster is the whole predictive distribution, so
the metrics that lead are the log score (the predictive density evaluated at the
outcome) and calibration, not a single error number. The numbers below are on the
test set, under archived forecast weather, averaged over all twelve evaluation
months, and come from [notebook 05](notebooks/05_model_comparison.ipynb):

| model | log score (nats) | 50% coverage | 90% coverage | CRPS (MW) |
| --- | --- | --- | --- | --- |
| BSTS | 7.47 | 0.50 | 0.86 | 272 |
| BART | 7.52 | 0.56 | 0.93 | 298 |
| ARIMA AR(1) | 7.62 | 0.54 | 0.87 | 258 |
| seasonal naive | 8.04 | 0.56 | 0.90 | 490 |
| LightGBM | none (no density) | 0.36 | 0.77 | 200 |

The BSTS posts the best log score in the field (7.47 against ARIMA's 7.62, paired
bootstrap p < 0.001) and the best calibration, with its 50 percent interval
covering almost exactly 50 percent of outcomes. It also wins the energy score
over whole 48-step paths, a score only a model with a coherent joint predictive
can earn.

LightGBM takes the single-number CRPS by a wide margin (200 MW), but it carries
no predictive density at all, so it has no log score, and its quantile intervals
are overconfident, with the nominal 90 percent band catching only 77 percent of
outcomes. CRPS is one axis, and not the one to judge a probabilistic forecast on
by itself. ARIMA actually edges the BSTS on overall CRPS (258 against 272,
p = 0.02) by winning the long horizon, while the BSTS is far sharper in the first
hours (CRPS 58 against 83 MW at 30 minutes), where the AR(2) error is still
reading the residuals at the origin. The absolute scores are larger than a
single-season evaluation would report, because asking a model to handle every
season at once is harder; the ranking is the story.

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
- `tests/`: scoring rules (sample CRPS verified against the analytic Gaussian),
  features, splits, loaders, the innovations likelihood (verified against a hand
  computation), the posterior KL primitives and a leakage audit. The audit pins
  down what no-look-ahead means here: training strictly precedes every evaluation
  window, both evaluation sets span all twelve months with balanced early and
  late slots, design rows do not change when the future is deleted, and the
  scalers and perturbation calibration see the training block only.

## Task and data

- **Target:** NSW1 `TOTALDEMAND` (the AEMO aggregated price-and-demand archive),
  five-minute dispatch averaged to half hours, May 2023 to May 2026, stored in
  UTC and displayed in AEST.
- **Origins:** 00:00 and 12:00 AEST daily, 48 half hours each.
- **Weather:** ERA5 actuals and archived ECMWF IFS forecasts as issued one day
  earlier (the Open-Meteo previous-runs API). The headline scores use the
  archived forecasts; perfect foresight and a calibrated error-inflation sweep
  are reported alongside as variants.
- **Splits:** season-blocked. Everything before June 2025 is one contiguous
  training block; each evaluation month then contributes an early and a late
  five-day window, and a balanced seeded draw sends one to validation and the
  other to test. Validation chooses the model settings, and the test set is
  touched once.

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
notebooks run with no downloads. A CUDA GPU is picked up automatically when JAX
sees one (`pip install "jax[cuda12]"`); the BSTS likelihood is pure matrix
arithmetic, so it runs fastest there, and the notebooks report measured timings
for both devices. `pytest` and `ruff` run in CI on every push.

## Data licences and attribution

- AEMO demand data are used under [AEMO's copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
- Weather data by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0): ERA5/ERA5T
  reanalysis (Copernicus Climate Change Service) and archived ECMWF IFS
  operational forecasts. See `data/README.md` for the series, conventions and
  caveats.

## Licence

MIT for the code. The data remain under their source licences.
