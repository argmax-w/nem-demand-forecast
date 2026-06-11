"""Build the processed train, validation and test splits from raw data.

Headless mirror of notebook 01: assembles the aligned half-hourly panel from
the interim demand series and raw weather pulls, applies the cleansing
documented there and writes the committed parquet splits.
"""

from __future__ import annotations

import argparse

import polars as pl

from nemforecastdemand.config import load_config
from nemforecastdemand.data import weather
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.features.preprocessing import build_panel
from nemforecastdemand.splits import chronological_split, split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    demand = pl.read_parquet(cfg.paths.interim / "demand.parquet")
    era5 = weather.load_raw(cfg.paths.raw / "weather" / "era5.parquet")
    forecast = weather.load_raw(cfg.paths.raw / "weather" / "forecast.parquet")

    panel, report = build_panel(demand, era5, forecast, cfg)
    print(report.as_frame().to_string(index=False))

    splits = chronological_split(panel.index, cfg.splits.train, cfg.splits.validation)
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    for name, index in splits.items():
        panel.loc[index].to_parquet(cfg.paths.processed / f"{name}.parquet")

    load_splits(cfg.paths.processed)
    print(split_summary(splits).to_string())


if __name__ == "__main__":
    main()
