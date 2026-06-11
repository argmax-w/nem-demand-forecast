# Data provenance

All timestamps in this project are NEM market time (AEST, UTC+10, no daylight
saving), represented by the `Australia/Brisbane` zone, in period-start
convention. The processed splits in `processed/` are committed so reviewers
can reproduce results without downloading anything.

## Demand

- **Series:** AEMO operational demand actuals, table
  `OPERATIONAL_DEMAND.ACTUAL`, column `OPERATIONAL_DEMAND`, region `NSW1`,
  half-hourly resolution.
- **Source:** NEMWeb `Operational Demand / ACTUAL_HH` reports,
  weekly archive zips at
  <https://nemweb.com.au/Reports/ARCHIVE/Operational_Demand/ACTUAL_HH/>.
- **Why this series:** operational demand measures the demand met by
  scheduled, semi-scheduled and significant non-scheduled generation, the
  quantity AEMO itself forecasts operationally. It is not the `TOTALDEMAND`
  column of `DISPATCHREGIONSUM` that most wrappers (for example NEMOSIS)
  expose, and the OPERATIONAL_DEMAND package is absent from the MMSDM monthly
  archives, so the reports are fetched directly.
- **Conventions:** AEMO stamps each half hour with its ending time; the data
  layer shifts stamps to period start. Duplicate publications are resolved by
  the latest `LASTCHANGED`.
- **Retention:** NEMWeb keeps roughly thirteen months of `ACTUAL_HH`
  archives. This binds the study window (see `config/default.yaml`); rerunning
  the download much later requires moving the window forward. The committed
  processed splits preserve the window used for the published results.
- **Licence:** AEMO data, used under the
  [AEMO copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).

## Weather

Both actuals and forecasts come from [Open-Meteo](https://open-meteo.com/)
(data CC BY 4.0, free for non-commercial use). Weather data by Open-Meteo.com.
Grid point: Sydney Observatory Hill (-33.87, 151.21), hourly resolution,
interpolated to the half-hourly grid during preprocessing.

- **Actuals (ground truth):** Historical Weather API (`/v1/archive`), model
  `era5_seamless`, which is ECMWF ERA5 reanalysis with the preliminary ERA5T
  tail for recent months. Used to train the demand-weather relationship and
  to score forecasts. Contains modelled reanalysis, not station observations.
- **Forecasts as issued:** Previous Runs API
  (`previous-runs-api.open-meteo.com/v1/forecast`), model
  `bom_access_global` (the Bureau of Meteorology's ACCESS-G), variable
  `temperature_2m_previous_day1`. Each timestamp carries the value predicted
  for it by the run initialised one day earlier, so day-ahead covariates have
  no look-ahead. Occasional missing runs appear as gaps and are handled in
  preprocessing.
- **Known mismatch:** weather coefficients are trained on ERA5 actuals while
  operational covariates come from ACCESS-G, which carries different biases.
  This mild train/serve mismatch is acknowledged in the notebooks; bias
  correction onto ERA5 is noted as future work.

## Directory layout

- `raw/` (gitignored): weekly NEMWeb zips and Open-Meteo pulls as fetched.
- `interim/` (gitignored): parsed half-hourly demand and the aligned panel.
- `processed/` (committed): validated train, validation and test parquet
  splits produced by notebook 01 / `scripts/build_dataset.py`.
