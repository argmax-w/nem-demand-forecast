"""Download NSW1 operational demand actuals from NEMWeb.

Fetches the weekly ``ACTUAL_HH`` archive zips covering the configured window
into ``data/raw/aemo`` and writes the parsed half-hourly series to
``data/interim/demand.parquet`` so later steps do not re-read thousands of
inner zips.
"""

from __future__ import annotations

import argparse
from datetime import date

from nemforecastdemand.config import load_config
from nemforecastdemand.data import aemo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = date.fromisoformat(cfg.window.start)
    end = date.fromisoformat(cfg.window.end)

    demand = aemo.load_demand(start, end, cfg.region, cfg.paths.raw / "aemo")
    cfg.paths.interim.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.interim / "demand.parquet"
    demand.write_parquet(out)
    print(
        f"{cfg.region} operational demand: {demand.height} half hours, "
        f"{demand['ts'].min()} to {demand['ts'].max()} -> {out}"
    )


if __name__ == "__main__":
    main()
