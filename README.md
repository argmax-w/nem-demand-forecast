# nem-demand-forecast

Probabilistic day-ahead forecasting of NSW1 operational demand in the Australian
National Electricity Market, built as a like-for-like comparison of inference
strategies on a single Bayesian structural time-series (BSTS) model.

## What this project shows

One generative model, two inference algorithms, one strong classical baseline:

- **Classical baseline:** dynamic harmonic regression with ARIMA errors
  (statsmodels SARIMAX), with Fourier harmonics for the daily and weekly cycles
  and temperature-based weather regressors.
- **BSTS by ADVI:** automatic-differentiation variational inference in NumPyro,
  mean-field and full-rank, with the ELBO tracked and decomposed into its
  energy and entropy terms.
- **BSTS by NUTS:** Hamiltonian Monte Carlo with the No-U-Turn Sampler, full
  diagnostics, treated as the reference posterior, with and without an ADVI
  warm start.

Because both Bayesian fits share the same model, the comparison isolates the
inference algorithm. Models are judged on probabilistic accuracy (CRPS, log
score, pinball loss), calibration (coverage, PIT) and computational cost
(fit and forecast wall-clock, ESS per second for MCMC).

## Status

Under construction. Results, figures and reproduction steps will appear here
as the notebooks land.

## Data and attribution

- Demand: AEMO operational demand actuals for NSW1 from NEMWeb, used under
  AEMO's data terms.
- Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0), supplying ERA5
  reanalysis actuals and archived operational forecasts (ACCESS-G).

See `data/README.md` for provenance, exact series and licences.

## Licence

MIT for the code in this repository. Data remain under their source licences.
