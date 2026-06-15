# Data provenance

A forecast is only as trustworthy as the data behind it, so this is the full
account of where every series comes from, how it is timestamped and what to
watch for. Everything is stored and modelled in UTC with period-start
timestamps. The display layers convert to AEST, the NEM market time, a fixed
UTC+10 with no daylight saving. AEMO publishes interval-ending market-time
stamps, so the data layer shifts and converts them on ingestion. The processed
panel and split labels in `processed/` are committed, so the results reproduce
without downloading anything.

## Demand

- **Series:** NSW1 `TOTALDEMAND` (the `DISPATCHREGIONSUM` regional demand met by
  scheduled, semi-scheduled and significant generation), at five-minute dispatch
  resolution averaged to half hours.
- **Source:** AEMO's aggregated price-and-demand archive, monthly CSVs at
  `https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_NSW1.csv`.
- **Why this series:** the evaluation split needs several years of history to
  cover every season in both validation and test. The half-hourly
  `OPERATIONAL_DEMAND.ACTUAL` reports on NEMWeb, the quantity AEMO forecasts
  operationally, keep only about thirteen months, too short for an all-season
  split. So I use the longer-running `TOTALDEMAND` series instead. Its loader and
  the operational-demand loader both live in
  `src/nemforecastdemand/data/aemo.py`. The window runs from May 2023 to May
  2026.
- **Conventions:** AEMO stamps each five-minute interval with its ending time in
  market time. Each reading is averaged into the half hour whose ending boundary
  it falls under, the ending stamp is shifted to a period start, and the
  fixed-offset market time is converted to UTC.
- **Licence:** AEMO data, used under the
  [AEMO copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).

## Weather

Both the actuals and the forecasts come from
[Open-Meteo](https://open-meteo.com/) (data CC BY 4.0, free for non-commercial
use). The grid point is Sydney Observatory Hill (-33.87, 151.21), pulled hourly
and interpolated to the half-hourly grid during preprocessing.

- **Actuals (ground truth):** the Historical Weather API (`/v1/archive`), model
  `era5_seamless`, which is ECMWF ERA5 reanalysis with the preliminary ERA5T tail
  for recent months. These train the demand-weather relationship and score the
  forecasts. Note that this is modelled reanalysis, not raw station observations.
- **Forecasts as issued:** the Previous Runs API
  (`previous-runs-api.open-meteo.com/v1/forecast`), model `ecmwf_ifs025`,
  variables `temperature_2m`, `dew_point_2m`, `direct_normal_irradiance` and
  `diffuse_radiation`, each at the `previous_day1` offset. Every timestamp
  carries the value predicted for it by the run initialised one day earlier, so
  the day-ahead covariates have no look-ahead built in. The Bureau's ACCESS-G
  (`bom_access_global`) would have been the natural operational choice, but its
  previous-runs archive is empty over this window, so I use the ECMWF IFS 0.25
  degree archive instead. That archive begins in March 2024. Earlier training
  rows carry the ERA5 actual in the forecast columns, and the perturbation error
  model is calibrated only on the rows with a genuine archived forecast.
  Occasional missing runs show up as gaps and are handled in preprocessing.
- **Known mismatch:** the weather coefficients are trained on ERA5 actuals while
  the day-ahead covariates come from the ECMWF IFS forecast, which carries
  different biases. This is a mild train/serve mismatch, acknowledged in the
  notebooks; bias-correcting the forecast onto ERA5 is noted as future work.

## Splits

The split is season-blocked. Everything before June 2025 is one contiguous
training block. Each month of the final year then contributes an early window
(days 8-12) and a late one (days 19-23), and a balanced seeded draw sends one
window of each month to validation and the other to test. Both evaluation sets
therefore span all twelve months, and the training block ends before every
evaluation window, so no fitting target ever sits behind an evaluation point. The
leakage and representativeness guarantees are exercised in
`tests/test_leakage.py`.

## Directory layout

- `raw/` (gitignored): the monthly price-and-demand CSVs and the Open-Meteo pulls.
- `interim/` (gitignored): the parsed half-hourly demand.
- `processed/` (committed): the validated `panel.parquet` and
  `split_labels.parquet` produced by notebook 01 and `scripts/build_dataset.py`.
