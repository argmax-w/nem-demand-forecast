"""Build the processed panel and season-blocked split labels from raw data.

Headless mirror of notebook 01: assembles the aligned half-hourly panel from
the interim demand series and raw weather pulls, applies the cleansing
documented there and writes the committed ``panel.parquet`` and
``split_labels.parquet``.
"""

from __future__ import annotations

import argparse

import pandas as pd
import polars as pl

from nemforecastdemand.config import load_config
from nemforecastdemand.data import weather
from nemforecastdemand.data.loaders import load_splits
from nemforecastdemand.features.preprocessing import build_panel
from nemforecastdemand.splits import season_blocked_split, split_labels, split_summary


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

    splits = season_blocked_split(panel.index, cfg.splits)
    labels = split_labels(panel.index, splits)

    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cfg.paths.processed / "panel.parquet")
    pd.DataFrame({"split": labels}).to_parquet(cfg.paths.processed / "split_labels.parquet")

    load_splits(cfg.paths.processed)
    print(split_summary(splits).to_string())
    print(f"\npanel {panel.index[0]} -> {panel.index[-1]}, {len(panel)} half hours")


if __name__ == "__main__":
    main()
